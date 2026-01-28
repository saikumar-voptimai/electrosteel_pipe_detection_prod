from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, Tuple, List
import yaml


Point = Tuple[int, int]
Polygon = List[Point]


@dataclass(frozen=True)
class GateRuntimeCfg:
    source_default: str
    stable_frames: int
    min_conf: float
    max_area_ratio_vs_closed: float
    max_w_over_h: float
    human_iou_occlusion: float


@dataclass(frozen=True)
class RuntimeCfg:
    video_source: int | str
    model_path: str
    tracker_yaml: str

    imgsz: int
    conf: float
    iou: float
    
    max_fps: int
    frame_skip: int

    db_path: str
    latest_jpg_path: str
    publish_fps: int
    publish_imgsz: int
    run_headless: bool
    db_flush_interval_s: float

    log_level: str
    log_path: str | None

    origin_confirm_frames: int
    loadcell_enter_confirm_frames: int
    loadcell_exit_confirm_frames: int
    stale_track_frames: int
    rearm_empty_frames: int

    gate: GateRuntimeCfg


@dataclass(frozen=True)
class PlcCfg:
    mode: str
    tags: Dict[str, str]
    pulse_ms: int
    modbus: Dict[str, Any] | None = None


@dataclass(frozen=True)
class AppCfg:
    runtime: RuntimeCfg
    rois: Dict[str, Polygon]
    plc: PlcCfg


def _load_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def load_config(runtime_path: str, rois_path: str, plc_path: str) -> AppCfg:
    r = _load_yaml(runtime_path)
    rois_raw = _load_yaml(rois_path)
    p = _load_yaml(plc_path)

    gate_raw = r.get("gate", {}) or {}
    gate = GateRuntimeCfg(
        source_default=str(gate_raw.get("source_default", "geometry")),
        stable_frames=int(gate_raw.get("stable_frames", 3)),
        min_conf=float(gate_raw.get("min_conf", 0.25)),
        max_area_ratio_vs_closed=float(gate_raw.get("max_area_ratio_vs_closed", 0.85)),
        max_w_over_h=float(gate_raw.get("max_w_over_h", 0.9)),
        human_iou_occlusion=float(gate_raw.get("human_iou_occlusion", 0.10)),
    )

    runtime = RuntimeCfg(
        video_source=r.get("video_source", 0),
        model_path=r["model_path"],
        tracker_yaml=r["tracker_yaml"],
        imgsz=int(r.get("imgsz", 640)),
        conf=float(r.get("conf", 0.25)),
        iou=float(r.get("iou", 0.5)),
        max_fps=int(r.get("max_fps", 15)),
        frame_skip=int(r.get("frame_skip", 0)),
        db_path=str(r.get("db_path", "var/pipes.db")),
        latest_jpg_path=str(r.get("latest_jpg_path", "var/latest.jpg")),
        publish_fps=int(r.get("publish_fps", 5)),
        publish_imgsz=int(r.get("publish_imgsz", 960)),
        run_headless=bool(r.get("run_headless", False)),
        db_flush_interval_s=float(r.get("db_flush_interval_s", 1.0)),
        log_level=str(r.get("log_level", "INFO")),
        log_path=(str(r["log_path"]) if r.get("log_path") else None),
        origin_confirm_frames=int(r.get("origin_confirm_frames", 2)),
        loadcell_enter_confirm_frames=int(r.get("loadcell_enter_confirm_frames", 1)),
        loadcell_exit_confirm_frames=int(r.get("loadcell_exit_confirm_frames", 2)),
        stale_track_frames=int(r.get("stale_track_frames", 45)),
        rearm_empty_frames=int(r.get("rearm_empty_frames", 10)),
        gate=gate,
    )

    plc = PlcCfg(
        mode=str(p.get("mode", "mock")),
        tags=dict(p.get("tags", {}) or {}),
        pulse_ms=int(p.get("pulse_ms", 300)),
        modbus=p.get("modbus"),
    )

    rois: Dict[str, Polygon] = {}
    for name, pts in (rois_raw or {}).items():
        rois[name] = [(int(x), int(y)) for (x, y) in pts]

    return AppCfg(runtime=runtime, rois=rois, plc=plc)
