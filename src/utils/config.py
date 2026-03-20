from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, Tuple, List
import yaml
from pathlib import Path


Point = Tuple[int, int]
Polygon = List[Point]

@dataclass(frozen=True)
class CameraCfg:
    id: int | str
    width: int
    height: int
    fps: int
    exposure_us: int
    gain_db: int
    auto_exposure: bool
    auto_gain: bool


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
    debug_mode: bool

    video_source: int | str
    model_path: str
    tracker_yaml: str

    imgsz: int
    conf: float
    iou: float
    
    max_fps: int
    frame_skip: int

    # Throttle non-inference logic (FSM updates, DB upserts, event handling).
    # 0 means run every inference frame.
    update_fps: int

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
    camera_cfg: CameraCfg | None
    weight: "WeightCfg | None" = None


@dataclass(frozen=True)
class WeightMachineCfg:
    ip: str
    trigger_byte: int | None
    trigger_bit: int | None
    weight_real_offset: int
    scale: float = 1.0
    offset: float = 0.0


@dataclass(frozen=True)
class WeightCfg:
    enabled: bool
    machine_id_default: int
    rack: int
    slot: int
    db_number: int
    pulse_ms: int
    read_duration_s: float
    read_interval_s: float
    stable_window_s: float
    min_nonzero: float
    max_std_rel: float
    max_std_abs: float
    machines: Dict[int, WeightMachineCfg]


def _load_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def load_config(
    runtime_path: str,
    rois_path: str,
    plc_path: str,
    camera_cfg_path: str = "config/camera.yaml",
    weight_cfg_path: str = "config/weight.yaml",
) -> AppCfg:
    r = _load_yaml(runtime_path)
    rois_raw = _load_yaml(rois_path)
    p = _load_yaml(plc_path)
    c_raw = _load_yaml(camera_cfg_path)

    # Weight config is optional.
    weight_raw: Dict[str, Any] = {}
    if weight_cfg_path and Path(weight_cfg_path).exists():
        weight_raw = _load_yaml(weight_cfg_path)

    cam = (c_raw.get("camera") if isinstance(c_raw, dict) else None) or (c_raw or {})
    camera_cfg: CameraCfg | None = None
    # Camera config is optional unless runtime.video_source is "gige".
    if cam:
        camera_cfg = CameraCfg(
            id=cam.get("id", 0),
            width=int(cam.get("width", 960)),
            height=int(cam.get("height", 640)),
            fps=int(cam.get("fps", 8)),
            exposure_us=int(cam.get("exposure_us", 15000)),
            gain_db=int(cam.get("gain_db", 5)),
            auto_exposure=bool(cam.get("auto_exposure", False)),
            auto_gain=bool(cam.get("auto_gain", False)),
        )

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
        debug_mode=bool(r.get("debug_mode", False)),
        video_source=r.get("video_source", 0),
        model_path=r["model_path"],
        tracker_yaml=r["tracker_yaml"],
        imgsz=int(r.get("imgsz", 640)),
        conf=float(r.get("conf", 0.25)),
        iou=float(r.get("iou", 0.5)),
        max_fps=int(r.get("max_fps", 15)),
        frame_skip=int(r.get("frame_skip", 0)),
        update_fps=int(r.get("update_fps", 0)),
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

    weight_cfg: WeightCfg | None = None
    if weight_raw:
        enabled = bool(weight_raw.get("enabled", True))
        machines_raw = dict(weight_raw.get("machines", {}) or {})
        machines: Dict[int, WeightMachineCfg] = {}
        for k, v in machines_raw.items():
            mid = int(k)
            v = v or {}
            machines[mid] = WeightMachineCfg(
                ip=str(v["ip"]),
                trigger_byte=(int(v["trigger_byte"]) if v.get("trigger_byte") is not None else None),
                trigger_bit=(int(v["trigger_bit"]) if v.get("trigger_bit") is not None else None),
                weight_real_offset=int(v["weight_real_offset"]),
                scale=float(v.get("scale", 1.0)),
                offset=float(v.get("offset", 0.0)),
            )

        weight_cfg = WeightCfg(
            enabled=enabled,
            machine_id_default=int(weight_raw.get("machine_id_default", 1)),
            rack=int(weight_raw.get("rack", 0)),
            slot=int(weight_raw.get("slot", 1)),
            db_number=int(weight_raw.get("db_number", 10)),
            pulse_ms=int(weight_raw.get("pulse_ms", 300)),
            read_duration_s=float(weight_raw.get("read_duration_s", 5.0)),
            read_interval_s=float(weight_raw.get("read_interval_s", 0.2)),
            stable_window_s=float(weight_raw.get("stable_window_s", 3.0)),
            min_nonzero=float(weight_raw.get("min_nonzero", 0.1)),
            max_std_rel=float(weight_raw.get("max_std_rel", 0.01)),
            max_std_abs=float(weight_raw.get("max_std_abs", 2.0)),
            machines=machines,
        )

    return AppCfg(runtime=runtime, rois=rois, plc=plc, camera_cfg=camera_cfg, weight=weight_cfg)
