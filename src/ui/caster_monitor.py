from __future__ import annotations

import sqlite3
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ui.formatting import fmt_ts
from utils.config import AppCfg, load_caster_config, resolve_caster_id


@dataclass(frozen=True)
class CasterMetrics:
    last_hour: int = 0
    last_8h: int = 0
    last_24h: int = 0
    avg_conf_8h: float = 0.0


@dataclass(frozen=True)
class CasterStatus:
    caster_key: str
    camera_type: str
    is_active: bool
    camera_status: str
    app_status: str
    plc_status: str
    database_status: str
    last_detection_ts: float | None
    latest_frame_ts: float | None
    latest_frame_age_s: float | None
    alerts: tuple[str, ...] = ()


@dataclass(frozen=True)
class CasterContext:
    caster_id: int
    caster_key: str
    cfg: AppCfg
    config_dir: Path
    db_path: Path
    latest_frame_path: Path
    log_path: Path | None


def get_all_casters(base_dir: str | Path = "config/casters") -> list[CasterContext]:
    root = Path(base_dir)
    contexts: list[CasterContext] = []
    if not root.exists():
        return contexts

    for path in sorted(root.glob("caster_*"), key=_caster_sort_key):
        if not path.is_dir():
            continue
        try:
            caster_id = resolve_caster_id(path.name)
            contexts.append(_context_from_cfg(load_caster_config(caster_id, str(path))))
        except Exception:
            continue
    return contexts


def get_caster(caster_id: int | str) -> CasterContext:
    return _context_from_cfg(load_caster_config(caster_id))


def _context_from_cfg(cfg: AppCfg) -> CasterContext:
    log_path = Path(cfg.runtime.log_path) if cfg.runtime.log_path else None
    return CasterContext(
        caster_id=cfg.caster_id,
        caster_key=cfg.caster_key,
        cfg=cfg,
        config_dir=Path(cfg.caster_config_path or ""),
        db_path=Path(cfg.runtime.db_path),
        latest_frame_path=Path(cfg.runtime.latest_jpg_path),
        log_path=log_path,
    )


def get_caster_camera(caster_id: int | str) -> dict[str, Any]:
    ctx = get_caster(caster_id)
    camera_cfg = ctx.cfg.camera_cfg
    return {
        "caster": ctx.caster_key,
        "type": camera_cfg.type if camera_cfg else "opencv",
        "source": ctx.cfg.runtime.video_source,
        "width": camera_cfg.width if camera_cfg else None,
        "height": camera_cfg.height if camera_cfg else None,
        "fps": camera_cfg.fps if camera_cfg else None,
    }


def get_caster_metrics(caster_id: int | str) -> CasterMetrics:
    ctx = get_caster(caster_id)
    return _metrics_from_db(ctx.db_path)


def get_caster_status(caster_id: int | str) -> CasterStatus:
    ctx = get_caster(caster_id)
    return _status_for_context(ctx)


def caster_statuses(contexts: list[CasterContext], active_after_s: float = 30.0) -> dict[str, CasterStatus]:
    return {ctx.caster_key: _status_for_context(ctx, active_after_s=active_after_s) for ctx in contexts}


def active_caster_contexts(contexts: list[CasterContext], active_after_s: float = 30.0) -> list[CasterContext]:
    statuses = caster_statuses(contexts, active_after_s=active_after_s)
    return [ctx for ctx in contexts if statuses[ctx.caster_key].is_active]


def fetch_recent_pipes(ctx: CasterContext, limit: int = 250) -> list[tuple]:
    if not ctx.db_path.exists():
        return []
    with _connect_readonly(ctx.db_path) as conn:
        pipe_checkpoint = "pipe_checkpoint" if _has_column(conn, "pipes", "pipe_checkpoint") else "0 AS pipe_checkpoint"
        return conn.execute(
            f"""
            SELECT pipe_uid, origin, {pipe_checkpoint}, t_origin, t_loadcell_enter, t_loadcell_exit,
                   weight, weight_quality, weight_samples,
                   avg_conf_full, avg_conf_till_gate, frames_missing, state, last_seen_ts
            FROM pipes
            ORDER BY COALESCE(t_origin, 0) DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()


def read_log_tail(ctx: CasterContext, max_lines: int = 80) -> str:
    if ctx.log_path is None or not ctx.log_path.exists():
        return ""
    lines = ctx.log_path.read_text(encoding="utf-8", errors="replace").splitlines()
    return "\n".join(lines[-max_lines:])


def aggregate_metrics(contexts: list[CasterContext]) -> CasterMetrics:
    metrics = [_metrics_from_db(ctx.db_path) for ctx in contexts]
    return CasterMetrics(
        last_hour=sum(m.last_hour for m in metrics),
        last_8h=sum(m.last_8h for m in metrics),
        last_24h=sum(m.last_24h for m in metrics),
        avg_conf_8h=_weighted_avg([m.avg_conf_8h for m in metrics if m.avg_conf_8h > 0]),
    )


def health_rows(
    contexts: list[CasterContext],
    active_after_s: float = 30.0,
    statuses: dict[str, CasterStatus] | None = None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    statuses = statuses or caster_statuses(contexts, active_after_s=active_after_s)
    for ctx in contexts:
        status = statuses[ctx.caster_key]
        rows.append(
            {
                "Caster": ctx.caster_key,
                "Active": "Yes" if status.is_active else "No",
                "Camera": f"{status.camera_type} / {status.camera_status}",
                "Status": status.app_status,
                "PLC": status.plc_status,
                "Database": status.database_status,
                "Last Detection": _age_label(status.last_detection_ts),
                "Last Frame": fmt_ts(status.latest_frame_ts),
                "Frame Age": _seconds_label(status.latest_frame_age_s),
                "Alerts": ", ".join(status.alerts),
            }
        )
    return rows


def _caster_sort_key(path: Path) -> tuple[int, str]:
    try:
        return (resolve_caster_id(path.name), path.name)
    except ValueError:
        return (10**9, path.name)


@contextmanager
def _connect_readonly(path: Path):
    conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True, timeout=5)
    try:
        yield conn
    finally:
        conn.close()


def _has_column(conn: sqlite3.Connection, table: str, column: str) -> bool:
    return any(row[1] == column for row in conn.execute(f"PRAGMA table_info({table})"))


def _metrics_from_db(path: Path) -> CasterMetrics:
    if not path.exists():
        return CasterMetrics()
    now = time.time()
    try:
        with _connect_readonly(path) as conn:
            return CasterMetrics(
                last_hour=_count_since(conn, now - 3600),
                last_8h=_count_since(conn, now - 8 * 3600),
                last_24h=_count_since(conn, now - 24 * 3600),
                avg_conf_8h=_avg_conf_since(conn, now - 8 * 3600),
            )
    except sqlite3.Error:
        return CasterMetrics()


def _count_since(conn: sqlite3.Connection, cutoff_ts: float) -> int:
    row = conn.execute(
        """
        SELECT COUNT(*) FROM pipes
        WHERE origin='caster' AND t_origin IS NOT NULL AND t_origin >= ?
        """,
        (cutoff_ts,),
    ).fetchone()
    return int(row[0] if row else 0)


def _avg_conf_since(conn: sqlite3.Connection, cutoff_ts: float) -> float:
    row = conn.execute(
        """
        SELECT AVG(avg_conf_full) FROM pipes
        WHERE origin='caster' AND t_origin IS NOT NULL
          AND t_origin >= ? AND avg_conf_full IS NOT NULL
        """,
        (cutoff_ts,),
    ).fetchone()
    value = row[0] if row else None
    return float(value) if value is not None else 0.0


def _status_for_context(ctx: CasterContext, active_after_s: float = 30.0) -> CasterStatus:
    now = time.time()
    camera_type = ctx.cfg.camera_cfg.type if ctx.cfg.camera_cfg else "opencv"
    latest_frame_ts = ctx.latest_frame_path.stat().st_mtime if ctx.latest_frame_path.exists() else None
    frame_age = (now - latest_frame_ts) if latest_frame_ts is not None else None
    database_status = _database_status(ctx.db_path)
    last_detection_ts = _last_detection_ts(ctx.db_path)
    is_active = frame_age is not None and frame_age <= active_after_s

    alerts: list[str] = []
    if frame_age is None:
        alerts.append("no frame")
    elif frame_age > active_after_s:
        alerts.append("stale frame")
    if database_status != "Healthy":
        alerts.append("db unavailable")

    camera_status = "Connected" if is_active else "No recent frame"
    app_status = "Running" if is_active else "Idle"
    plc_status = "Mock" if ctx.cfg.plc.mode.lower() == "mock" else "Configured"

    return CasterStatus(
        caster_key=ctx.caster_key,
        camera_type=camera_type,
        is_active=is_active,
        camera_status=camera_status,
        app_status=app_status,
        plc_status=plc_status,
        database_status=database_status,
        last_detection_ts=last_detection_ts,
        latest_frame_ts=latest_frame_ts,
        latest_frame_age_s=frame_age,
        alerts=tuple(alerts),
    )


def _database_status(path: Path) -> str:
    if not path.exists():
        return "Missing"
    try:
        with _connect_readonly(path) as conn:
            conn.execute("SELECT 1 FROM pipes LIMIT 1").fetchone()
        return "Healthy"
    except sqlite3.Error:
        return "Error"


def _last_detection_ts(path: Path) -> float | None:
    if not path.exists():
        return None
    try:
        with _connect_readonly(path) as conn:
            row = conn.execute("SELECT MAX(last_seen_ts) FROM pipes").fetchone()
            return float(row[0]) if row and row[0] is not None else None
    except sqlite3.Error:
        return None


def _age_label(ts: float | None) -> str:
    if ts is None:
        return "-"
    return _seconds_label(time.time() - ts)


def _seconds_label(seconds: float | None) -> str:
    if seconds is None:
        return "-"
    if seconds < 60:
        return f"{seconds:.0f}s ago"
    if seconds < 3600:
        return f"{seconds / 60:.0f}m ago"
    return f"{seconds / 3600:.1f}h ago"


def _weighted_avg(values: list[float]) -> float:
    if not values:
        return 0.0
    return sum(values) / len(values)
