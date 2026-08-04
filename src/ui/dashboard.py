import html
import os
import sqlite3
import sys
from collections.abc import Callable
from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st
from streamlit_autorefresh import st_autorefresh

# Allow imports when run as: streamlit run src/ui/dashboard.py
SRC_DIR = Path(__file__).resolve().parents[1]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from db.repo import SqliteRepo
from ui.caster_monitor import (
    CasterContext,
    CasterStatus,
    aggregate_metrics,
    caster_statuses,
    fetch_recent_pipes,
    fetch_recent_trolley_gate2_intersections,
    get_all_casters,
    health_rows,
    read_log_tail,
)
from ui.formatting import fmt_ts
from utils.camera_profiles import (
    CameraControlState,
    active_camera_profile,
    load_camera_control_state,
    save_camera_control_state,
    validate_camera_profiles,
)
from utils.config import CameraProfileCfg, resolve_caster_id


ALL_CASTERS = "All Casters"
DEFAULT_ACTIVE_WINDOW_S = 30
DEFAULT_REFRESH_MS = 5000


st.set_page_config(layout="wide", page_title="Caster Monitoring Dashboard")


@st.cache_data(ttl=30, show_spinner=False)
def _load_casters() -> list[CasterContext]:
    return get_all_casters()


def _inject_css() -> None:
    st.markdown(
        """
        <style>
        .block-container { padding-top: 1.1rem; padding-bottom: 2rem; max-width: 1540px; }
        h1, h2, h3 { letter-spacing: 0; }
        [data-testid="stMetric"] {
            background: #ffffff;
            border: 1px solid #d9e2ec;
            border-radius: 8px;
            padding: 0.85rem 1rem;
            box-shadow: 0 1px 2px rgba(16, 24, 40, 0.04);
        }
        [data-testid="stMetricLabel"] { color: #475467; }
        [data-testid="stMetricValue"] { color: #101828; }
        .dash-header {
            display: flex;
            justify-content: space-between;
            gap: 1rem;
            align-items: flex-start;
            padding: 0.3rem 0 0.85rem 0;
        }
        .dash-header h1 {
            font-size: 1.75rem;
            line-height: 2.1rem;
            margin: 0;
            color: #101828;
        }
        .dash-header p {
            margin: 0.2rem 0 0 0;
            color: #667085;
            font-size: 0.95rem;
        }
        .status-pill {
            display: inline-flex;
            align-items: center;
            min-height: 1.8rem;
            padding: 0.25rem 0.65rem;
            border-radius: 999px;
            font-size: 0.82rem;
            font-weight: 700;
            white-space: nowrap;
            border: 1px solid transparent;
        }
        .status-active { color: #067647; background: #ecfdf3; border-color: #abefc6; }
        .status-idle { color: #475467; background: #f2f4f7; border-color: #d0d5dd; }
        .status-warn { color: #b42318; background: #fef3f2; border-color: #fecdca; }
        .camera-card-head {
            display: flex;
            justify-content: space-between;
            gap: 0.75rem;
            align-items: center;
            padding: 0.65rem 0.75rem;
            margin: 0 0 0.45rem 0;
            border: 1px solid #d9e2ec;
            border-radius: 8px;
            background: #ffffff;
        }
        .camera-title { font-weight: 750; color: #101828; font-size: 1rem; }
        .camera-meta { color: #667085; font-size: 0.82rem; margin-top: 0.1rem; }
        .section-title {
            color: #101828;
            font-size: 1.05rem;
            font-weight: 760;
            margin: 0.8rem 0 0.35rem 0;
        }
        .quiet-note { color: #667085; font-size: 0.86rem; margin: 0.15rem 0 0.75rem 0; }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _initial_selection(options: list[str]) -> str:
    env_caster = os.environ.get("PIPE_DASHBOARD_CASTER")
    if not env_caster:
        return ALL_CASTERS
    try:
        key = f"caster_{resolve_caster_id(env_caster)}"
    except ValueError:
        return ALL_CASTERS
    return key if key in options else ALL_CASTERS


def _supports_fragments() -> bool:
    return callable(getattr(st, "fragment", None))


def _fragmented(func: Callable[..., None], refresh_ms: int) -> Callable[..., None]:
    fragment = getattr(st, "fragment", None)
    if not callable(fragment):
        return func
    return fragment(run_every=max(refresh_ms / 1000.0, 1.0))(func)


def _status_class(status: CasterStatus) -> str:
    if status.database_status != "Healthy":
        return "warn"
    return "active" if status.is_active else "idle"


def _status_label(status: CasterStatus) -> str:
    if status.database_status != "Healthy":
        return "Attention"
    return "Active" if status.is_active else "Idle"


def _status_chip(status: CasterStatus) -> str:
    cls = _status_class(status)
    return f'<span class="status-pill status-{cls}">{html.escape(_status_label(status))}</span>'


def _age_text(seconds: float | None) -> str:
    if seconds is None:
        return "no frame"
    if seconds < 60:
        return f"{seconds:.0f}s ago"
    if seconds < 3600:
        return f"{seconds / 60:.0f}m ago"
    return f"{seconds / 3600:.1f}h ago"


def _camera_label(ctx: CasterContext) -> tuple[str, str]:
    camera_cfg = ctx.cfg.camera_cfg
    camera_type = camera_cfg.type if camera_cfg else "opencv"
    fps = camera_cfg.fps if camera_cfg and camera_cfg.fps else "-"
    return camera_type, str(fps)


def _show_image(path: Path) -> None:
    try:
        st.image(str(path), width="stretch")
    except TypeError:
        st.image(str(path), use_column_width=True)


def _show_dataframe(df: pd.DataFrame) -> None:
    st.dataframe(df, width="stretch", hide_index=True)


def _metric_row(metrics, *, active_count: int | None = None, total_count: int | None = None) -> None:
    cols = st.columns(5 if active_count is not None else 4)
    offset = 0
    if active_count is not None and total_count is not None:
        cols[0].metric("Active casters", f"{active_count}/{total_count}")
        offset = 1
    cols[offset + 0].metric("Last hour", metrics.last_hour)
    cols[offset + 1].metric("Last 8h", metrics.last_8h)
    cols[offset + 2].metric("Last 24h", metrics.last_24h)
    cols[offset + 3].metric("Avg conf 8h", f"{metrics.avg_conf_8h:.2f}")


def _single_metric_row(metrics, status: CasterStatus) -> None:
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Caster status", _status_label(status))
    c2.metric("Last hour", metrics.last_hour)
    c3.metric("Last 8h", metrics.last_8h)
    c4.metric("Last 24h", metrics.last_24h)
    c5.metric("Avg conf 8h", f"{metrics.avg_conf_8h:.2f}")


def _camera_grid(
    contexts: list[CasterContext],
    statuses: dict[str, CasterStatus],
    columns: int = 2,
    *,
    empty_message: str = "No active casters right now.",
) -> None:
    st.markdown('<div class="section-title">Live Cameras</div>', unsafe_allow_html=True)
    if not contexts:
        st.info(empty_message)
        return

    for start in range(0, len(contexts), columns):
        cols = st.columns(columns)
        for col, ctx in zip(cols, contexts[start : start + columns]):
            with col:
                status = statuses[ctx.caster_key]
                camera_type, fps = _camera_label(ctx)
                meta = f"{camera_type} | fps={fps} | frame {_age_text(status.latest_frame_age_s)}"
                st.markdown(
                    f"""
                    <div class="camera-card-head">
                      <div>
                        <div class="camera-title">{html.escape(ctx.caster_key)}</div>
                        <div class="camera-meta">{html.escape(meta)}</div>
                      </div>
                      {_status_chip(status)}
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                if status.is_active and ctx.latest_frame_path.exists():
                    _show_image(ctx.latest_frame_path)
                elif ctx.latest_frame_path.exists():
                    st.warning(f"Latest frame is stale: {fmt_ts(status.latest_frame_ts)}")
                else:
                    st.info("Waiting for latest frame")


def _health_table(
    contexts: list[CasterContext],
    statuses: dict[str, CasterStatus],
    active_window_s: int,
    *,
    empty_message: str = "No configured casters found.",
) -> None:
    st.markdown('<div class="section-title">Caster Health</div>', unsafe_allow_html=True)
    rows = health_rows(contexts, active_after_s=active_window_s, statuses=statuses)
    if not rows:
        st.info(empty_message)
        return
    _show_dataframe(pd.DataFrame(rows))


def _pipes_dataframe(ctx: CasterContext) -> pd.DataFrame:
    rows = fetch_recent_pipes(ctx, limit=250)
    df = pd.DataFrame(
        rows,
        columns=[
            "pipe_uid",
            "origin",
            "pipe_checkpoint",
            "t_origin",
            "t_loadcell_enter",
            "t_loadcell_exit",
            "weight",
            "weight_quality",
            "weight_samples",
            "avg_conf_full",
            "avg_conf_till_gate",
            "frames_missing",
            "state",
            "last_seen_ts",
        ],
    )
    for col in ("t_origin", "t_loadcell_enter", "t_loadcell_exit", "last_seen_ts"):
        if col in df.columns:
            df[col] = df[col].apply(fmt_ts)
    for col in ("pipe_uid", "origin", "state", "t_origin", "t_loadcell_enter", "t_loadcell_exit", "last_seen_ts"):
        if col in df.columns:
            df[col] = df[col].map(lambda value: "" if value is None else str(value)).astype(object)
    return df


def _trolley_gate2_dataframe(ctx: CasterContext) -> pd.DataFrame:
    rows = fetch_recent_trolley_gate2_intersections(ctx, limit=250)
    df = pd.DataFrame(
        rows,
        columns=["id", "timestamp", "trolley_track_id", "pipe_on_trolley"],
    )
    if "timestamp" in df.columns:
        df["timestamp"] = df["timestamp"].apply(fmt_ts)
    if "pipe_on_trolley" in df.columns:
        df["pipe_on_trolley"] = df["pipe_on_trolley"].map({1: "Yes", 0: "No"}).fillna(df["pipe_on_trolley"])
    return df


def _read_gate_source(ctx: CasterContext) -> str:
    default = ctx.cfg.runtime.gate.source_default
    if not ctx.db_path.exists():
        return default
    conn = None
    try:
        conn = sqlite3.connect(f"file:{ctx.db_path}?mode=ro", uri=True, timeout=2)
        row = conn.execute("SELECT value FROM settings WHERE key='gate_source'").fetchone()
        return str(row[0]) if row and row[0] else default
    except sqlite3.Error:
        return default
    finally:
        if conn is not None:
            conn.close()


def _write_gate_source(ctx: CasterContext, gate_source: str) -> None:
    repo = SqliteRepo(str(ctx.db_path))
    try:
        repo.set_setting("gate_source", gate_source)
    finally:
        repo.close()


def _camera_profile_rows(state: CameraControlState) -> list[dict]:
    return [
        {
            "Profile": name,
            "Start": profile.start,
            "End": profile.end,
            "Exposure (µs)": profile.exposure_us,
            "Gain (dB)": profile.gain_db,
            "Gamma enabled": profile.gamma_enable,
            "Gamma": profile.gamma,
        }
        for name, profile in state.profiles.items()
    ]


def _profiles_from_editor(editor_value: pd.DataFrame) -> dict[str, CameraProfileCfg]:
    profiles: dict[str, CameraProfileCfg] = {}
    for row in editor_value.to_dict(orient="records"):
        name_value = row.get("Profile")
        name = "" if pd.isna(name_value) else str(name_value).strip()
        if not name:
            raise ValueError("Every profile row needs a unique name.")
        if name in profiles:
            raise ValueError(f"Duplicate camera profile name: {name!r}.")
        try:
            profiles[name] = CameraProfileCfg(
                start=str(row["Start"]).strip(),
                end=str(row["End"]).strip(),
                exposure_us=int(row["Exposure (µs)"]),
                gain_db=int(row["Gain (dB)"]),
                gamma_enable=bool(row["Gamma enabled"]),
                gamma=float(row["Gamma"]),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"Profile {name!r} contains an invalid camera value.") from exc
    return validate_camera_profiles(profiles)


def _clear_camera_schedule_feedback() -> None:
    for key in list(st.session_state):
        if key.startswith("camera_schedule_feedback_"):
            st.session_state.pop(key, None)


def _show_camera_schedule_feedback(ctx: CasterContext) -> None:
    feedback_key = f"camera_schedule_feedback_{ctx.caster_key}"
    feedback = st.session_state.get(feedback_key)
    if not feedback:
        return

    level, message = feedback
    if level == "success":
        st.success(message)
    else:
        st.error(message)


def _camera_profile_editor(ctx: CasterContext) -> None:
    camera_cfg = ctx.cfg.camera_cfg
    if camera_cfg is None or camera_cfg.type != "va_imaging":
        st.info("Scheduled exposure, gain, and gamma controls are available for VA Imaging cameras.")
        return

    config_path = Path(ctx.cfg.camera_cfg_path)
    try:
        state = load_camera_control_state(config_path)
    except Exception as exc:
        st.error(f"Cannot load {config_path}: {exc}")
        return

    active_name, active_profile = active_camera_profile(state.profiles, datetime.now())
    st.markdown(
        f'<div class="section-title">Camera Schedule - {html.escape(ctx.caster_key)}</div>',
        unsafe_allow_html=True,
    )
    st.caption(
        f"Active profile: {active_name or 'none'}"
        + (
            f" ({active_profile.start}–{active_profile.end})"
            if active_profile is not None
            else ""
        )
        + " · Changes are applied by the background scheduler within about 2 seconds."
    )
    _show_camera_schedule_feedback(ctx)

    try:
        config_version = config_path.stat().st_mtime_ns
    except OSError:
        config_version = 0
    editor_key = f"camera_profile_editor_{ctx.caster_key}_{config_version}"
    auto_col_1, auto_col_2 = st.columns(2)
    with auto_col_1:
        auto_exposure = st.checkbox(
            "Auto exposure",
            value=state.auto_exposure,
            key=f"camera_auto_exposure_{ctx.caster_key}_{config_version}",
            on_change=_clear_camera_schedule_feedback,
        )
    with auto_col_2:
        auto_gain = st.checkbox(
            "Auto gain",
            value=state.auto_gain,
            key=f"camera_auto_gain_{ctx.caster_key}_{config_version}",
            on_change=_clear_camera_schedule_feedback,
        )

    edited = st.data_editor(
        pd.DataFrame(_camera_profile_rows(state)),
        key=editor_key,
        width="stretch",
        hide_index=True,
        num_rows="dynamic",
        on_change=_clear_camera_schedule_feedback,
        column_config={
            "Profile": st.column_config.TextColumn("Profile", required=True),
            "Start": st.column_config.TextColumn("Start", help="24-hour HH:MM", required=True),
            "End": st.column_config.TextColumn("End", help="24-hour HH:MM", required=True),
            "Exposure (µs)": st.column_config.NumberColumn(
                "Exposure (µs)", min_value=1, step=1000, required=True
            ),
            "Gain (dB)": st.column_config.NumberColumn(
                "Gain (dB)", min_value=0, step=1, required=True
            ),
            "Gamma enabled": st.column_config.CheckboxColumn("Gamma enabled"),
            "Gamma": st.column_config.NumberColumn(
                "Gamma", min_value=0.1, max_value=10.0, step=0.1, required=True
            ),
        },
    )
    st.caption(
        "Use +/− row controls to add or remove profiles. The periods must cover all 24 hours "
        "without gaps or overlaps; overnight periods such as 20:00–06:00 are supported."
    )
    submitted = st.button("Save camera schedule", type="primary")

    if not submitted:
        return
    try:
        profiles = _profiles_from_editor(edited)
        saved = save_camera_control_state(
            config_path,
            CameraControlState(
                auto_exposure=auto_exposure,
                auto_gain=auto_gain,
                profiles=profiles,
            ),
        )
    except Exception as exc:
        message = f"Camera schedule was not saved: {exc}"
        st.session_state[f"camera_schedule_feedback_{ctx.caster_key}"] = ("error", message)
        st.error(message)
        return

    message = (
        f"Saved {len(saved.profiles)} profiles for {ctx.caster_key}. "
        "The running camera scheduler will apply the active profile automatically."
    )
    st.session_state[f"camera_schedule_feedback_{ctx.caster_key}"] = ("success", message)
    st.success(message)


def _single_caster_view(
    ctx: CasterContext,
    active_window_s: int,
    statuses: dict[str, CasterStatus] | None = None,
) -> None:
    statuses = statuses or caster_statuses([ctx], active_after_s=active_window_s)
    status = statuses[ctx.caster_key]
    metrics = aggregate_metrics([ctx])
    _single_metric_row(metrics, status)

    tab_live, tab_pipes, tab_trolley, tab_logs, tab_camera, tab_config = st.tabs(
        ["Live", "Recent Pipes", "Trolley Gate2", "Logs", "Camera Control", "Configuration"]
    )

    with tab_live:
        _health_table([ctx], statuses, active_window_s)
        _camera_grid([ctx], statuses, columns=1, empty_message="No live frame for this caster.")

    with tab_pipes:
        st.markdown(f'<div class="section-title">Recent Pipes - {html.escape(ctx.caster_key)}</div>', unsafe_allow_html=True)
        _show_dataframe(_pipes_dataframe(ctx))

    with tab_trolley:
        st.markdown(f'<div class="section-title">Trolley Gate2 - {html.escape(ctx.caster_key)}</div>', unsafe_allow_html=True)
        _show_dataframe(_trolley_gate2_dataframe(ctx))

    with tab_logs:
        st.markdown(f'<div class="section-title">Logs - {html.escape(ctx.caster_key)}</div>', unsafe_allow_html=True)
        tail = read_log_tail(ctx)
        st.code(tail or "No log file found.", language="text")

    with tab_camera:
        _camera_profile_editor(ctx)

    with tab_config:
        camera_cfg = ctx.cfg.camera_cfg
        st.markdown('<div class="section-title">Runtime Context</div>', unsafe_allow_html=True)
        st.json(
            {
                "caster": ctx.caster_key,
                "camera_type": camera_cfg.type if camera_cfg else "opencv",
                "video_source": ctx.cfg.runtime.video_source,
                "db_path": str(ctx.db_path),
                "latest_frame": str(ctx.latest_frame_path),
                "log_path": str(ctx.log_path) if ctx.log_path else None,
                "plc_mode": ctx.cfg.plc.mode,
                "model_path": ctx.cfg.runtime.model_path,
            }
        )


def _all_casters_view(
    contexts: list[CasterContext],
    active_window_s: int,
    show_idle: bool,
    statuses: dict[str, CasterStatus],
) -> None:
    active_contexts = [ctx for ctx in contexts if statuses[ctx.caster_key].is_active]
    visible_contexts = contexts if show_idle else active_contexts
    metrics = aggregate_metrics(visible_contexts)

    _metric_row(metrics, active_count=len(active_contexts), total_count=len(contexts))
    hidden_count = len(contexts) - len(active_contexts)
    if hidden_count and not show_idle:
        st.markdown(
            f'<div class="quiet-note">{hidden_count} idle caster(s) hidden because the latest frame is older than {active_window_s}s.</div>',
            unsafe_allow_html=True,
        )

    _health_table(
        visible_contexts,
        statuses,
        active_window_s,
        empty_message="No active casters in the selected frame window.",
    )
    _camera_grid(visible_contexts, statuses, columns=2)


def _dashboard_body(
    selected: str,
    casters: list[CasterContext],
    caster_by_key: dict[str, CasterContext],
    active_window_s: int,
    show_idle: bool,
) -> None:
    statuses = caster_statuses(casters, active_after_s=active_window_s)
    _render_header(selected, casters, active_window_s, statuses)

    if selected == ALL_CASTERS:
        _all_casters_view(casters, active_window_s, show_idle, statuses)
        return

    ctx = caster_by_key.get(selected)
    if ctx is None:
        st.error("Selected caster is not available.")
        return
    _single_caster_view(ctx, active_window_s, statuses)


def _render_header(
    selected: str,
    casters: list[CasterContext],
    active_window_s: int,
    statuses: dict[str, CasterStatus],
) -> None:
    active_count = sum(1 for status in statuses.values() if status.is_active)
    scope = "All Casters" if selected == ALL_CASTERS else selected
    st.markdown(
        f"""
        <div class="dash-header">
          <div>
            <h1>Caster Monitoring</h1>
            <p>{html.escape(scope)} | active {active_count}/{len(casters)} | window {active_window_s}s</p>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def main() -> None:
    _inject_css()
    casters = _load_casters()
    caster_by_key = {ctx.caster_key: ctx for ctx in casters}
    options = [ALL_CASTERS, *caster_by_key.keys()]

    with st.sidebar:
        st.header("Controls")
        selected = st.selectbox(
            "Caster",
            options,
            index=options.index(_initial_selection(options)),
            on_change=_clear_camera_schedule_feedback,
        )
        refresh_ms = st.slider(
            "Refresh interval",
            1000,
            10000,
            DEFAULT_REFRESH_MS,
            1000,
            on_change=_clear_camera_schedule_feedback,
        )
        active_window_s = st.slider(
            "Active frame window",
            10,
            300,
            DEFAULT_ACTIVE_WINDOW_S,
            5,
            on_change=_clear_camera_schedule_feedback,
        )
        show_idle = st.checkbox(
            "Show idle casters",
            value=False,
            disabled=selected != ALL_CASTERS,
            on_change=_clear_camera_schedule_feedback,
        )

        selected_ctx = caster_by_key.get(selected)
        gate_options = ["geometry", "plc", "vision"]
        current_gate_source = _read_gate_source(selected_ctx) if selected_ctx else gate_options[0]
        gate_index = gate_options.index(current_gate_source) if current_gate_source in gate_options else 0
        gate_source = st.selectbox(
            "Gate source",
            gate_options,
            index=gate_index,
            disabled=selected_ctx is None,
            on_change=_clear_camera_schedule_feedback,
        )
        if st.button(
            "Apply gate source",
            disabled=selected_ctx is None,
            on_click=_clear_camera_schedule_feedback,
        ):
            _write_gate_source(selected_ctx, gate_source)
            st.success(f"{selected_ctx.caster_key} gate source set to {gate_source}")

    if not _supports_fragments():
        st_autorefresh(interval=refresh_ms, key="dashboard_refresh")
        _dashboard_body(selected, casters, caster_by_key, active_window_s, show_idle)
        return

    _fragmented(_dashboard_body, refresh_ms)(selected, casters, caster_by_key, active_window_s, show_idle)


main()
