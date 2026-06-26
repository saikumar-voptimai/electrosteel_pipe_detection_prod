from __future__ import annotations
from dataclasses import dataclass, replace
from typing import Any, Dict, Tuple, List
import yaml
from pathlib import Path
import re


Point = Tuple[int, int]
Polygon = List[Point]

@dataclass(frozen=True)
class CameraProfileCfg:
    start: str
    end: str
    exposure_us: int
    gain_db: int
    gamma_enable: bool
    gamma: float
@dataclass(frozen=True)
class CameraReconnectCfg:
    max_retries: int = 5
    sleep_s: float = 1.0


@dataclass(frozen=True)
class BaslerCameraCfg:
    device_user_id: str = ""
    serial_number: str = ""
    ip_address: str = ""
    exposure_time: float | None = None
    gain: float | None = None
    grab_strategy: str = "latest_image_only"
    timeout_ms: int = 1000
    pixel_format: str | None = None
    width: int | None = None
    height: int | None = None
    offset_x: int | None = None
    offset_y: int | None = None
    acquisition_frame_rate: float | None = None
    packet_size: int | None = None
    inter_packet_delay: int | None = None
    max_num_buffer: int = 20


@dataclass(frozen=True)
class CameraCfg:
    type: str
    id: int | str
    width: int
    height: int
    fps: int
    auto_exposure: bool
    auto_gain: bool
    va_imaging: Dict[str, Any] | None = None
    basler: BaslerCameraCfg | None = None
    profiles: Dict[str, CameraProfileCfg] | None = None
    reconnect: CameraReconnectCfg | None = None


@dataclass(frozen=True)
class GateRuntimeCfg:
    source_default: str
    stable_frames: int
    min_conf: float
    max_area_ratio_vs_closed: float
    max_w_over_h: float
    human_iou_occlusion: float

@dataclass(frozen=True)
class HistoryCfg:
    enabled: bool = False
    base_dir: str = ""
    prefix: str = "pipe"
    ext: str = "jpg"
    date_folder_format: str = "%Y-%m-%d"
    time_filename_format: str = "%H-%M-%S-%f"
    timezone: str = "Asia/Kolkata"
    shifts: list | None = None

    @classmethod
    def from_dict(cls, d: dict | None) -> "HistoryCfg":
        return cls(**(d or {}))


@dataclass(frozen=True)
class RuntimeCfg:
    debug_mode: bool
    video_source: int | str
    model_path: str
    tracker_yaml: str
    imgsz: int
    conf: float
    iou: float
    device: int | str | None
    half: bool
    max_fps: int
    frame_skip: int
    # Throttle non-inference logic (FSM updates, DB upserts, event handling).
    # 0 means run every inference frame.
    update_fps: int
    db_path: str
    latest_jpg_path: str
    publish_fps: int
    publish_imgsz: int | tuple[int, int]
    run_headless: bool
    db_flush_interval_s: float
    log_level: str
    log_path: str | None
    origin_confirm_frames: int
    loadcell_enter_confirm_frames: int
    loadcell_exit_confirm_frames: int
    stale_track_frames: int
    rearm_empty_frames: int
    min_pipe_gap_seconds: int
    loadcell_covered_per: int
    remove_pipe_id_pipe_checkpoint_not_entered: bool
    history: HistoryCfg | None
    class_name_to_id: Dict[str, int]
    publish_overlay: bool
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
    caster_id: int = 1
    caster_key: str = "caster_1"
    caster_config_path: str | None = None
    caster_storage_path: str = "var/caster_1"
    rois_path: str = "config/rois.yaml"
    camera_cfg_path: str = "config/camera.yaml"


@dataclass(frozen=True)
class CasterFileCfg:
    caster_id: int
    caster_key: str
    enabled: bool
    runtime: str
    rois: str
    camera: str
    plc: str
    weight: str
    bytetrack: str
    storage_dir: str
    overrides: Dict[str, Any]


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


def _load_yaml(path: str | Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def resolve_caster_id(raw: int | str) -> int:
    if isinstance(raw, str):
        text = raw.strip().lower()
        match = re.fullmatch(r"(?:caster[_-])?(\d+)", text)
        if not match:
            raise ValueError(f"Invalid caster id {raw!r}. Expected a positive integer or caster_<id>.")
        caster_id = int(match.group(1))
    else:
        caster_id = int(raw)
    return validate_caster_id(caster_id)


def validate_caster_id(caster_id: int) -> int:
    caster_id = int(caster_id)
    if caster_id < 1:
        raise ValueError(f"Invalid caster id {caster_id}. Expected a positive integer.")
    return caster_id


def caster_key(caster_id: int | str) -> str:
    return f"caster_{resolve_caster_id(caster_id)}"


def resolve_caster_config_dir(caster_id: int | str, base_dir: str | Path = "config/casters") -> Path:
    return Path(base_dir) / caster_key(caster_id)


def default_caster_config_path(caster_id: int | str) -> str:
    return str(resolve_caster_config_dir(caster_id))


def legacy_caster_config_path(caster_id: int | str) -> str:
    caster_id = resolve_caster_id(caster_id)
    return f"config/casters/caster_{caster_id}_config.yaml"


def resolve_caster_storage_path(caster_id: int | str, var_dir: str | Path = "var") -> Path:
    return Path(var_dir) / caster_key(caster_id)


def resolve_caster_database_path(caster_id: int | str, var_dir: str | Path = "var") -> Path:
    key = caster_key(caster_id)
    return resolve_caster_storage_path(caster_id, var_dir) / f"{key}_pipes.db"


def resolve_caster_latest_frame_path(caster_id: int | str, var_dir: str | Path = "var") -> Path:
    return resolve_caster_storage_path(caster_id, var_dir) / "latest.jpg"


def resolve_caster_log_path(caster_id: int | str, var_dir: str | Path = "var") -> Path:
    return resolve_caster_storage_path(caster_id, var_dir) / "pipe_detect.log"


def _first_existing(*paths: Path) -> Path:
    for path in paths:
        if path.exists():
            return path
    return paths[0]


def _normalize_rois(rois_raw: Dict[str, Any]) -> Dict[str, Polygon]:
    if "roi_caster_origin" not in rois_raw and "roi_caster5_origin" in rois_raw:
        rois_raw = dict(rois_raw)
        rois_raw["roi_caster_origin"] = rois_raw["roi_caster5_origin"]

    rois: Dict[str, Polygon] = {}
    for name, pts in (rois_raw or {}).items():
        rois[name] = [(int(x), int(y)) for (x, y) in pts]
    return rois


def _ensure_parent_dir(path: str | None) -> None:
    if not path:
        return
    parent = Path(path).expanduser().parent
    if str(parent):
        parent.mkdir(parents=True, exist_ok=True)


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    return float(value)


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    return int(value)


def _parse_camera_cfg(c_raw: Dict[str, Any]) -> CameraCfg | None:
    cam = (c_raw.get("camera") if isinstance(c_raw, dict) else None) or (c_raw or {})
    if not isinstance(cam, dict) or not cam:
        return None

    camera_type = str(cam.get("type", "va_imaging")).strip().lower()
    va_raw = dict(cam.get("va_imaging", {}) or {})
    basler_raw = dict(cam.get("basler", {}) or {})
    common_raw = dict(cam)
    for key in ("type", "va_imaging", "basler"):
        common_raw.pop(key, None)

    # Backward compatibility: old camera.yaml placed VA Imaging settings directly under camera.
    settings = {**common_raw, **va_raw} if camera_type == "va_imaging" else common_raw

    profiles_raw = dict(settings.get("profiles", {}) or {})
    profiles: Dict[str, CameraProfileCfg] | None = None
    if profiles_raw:
        profiles = {
            str(name): CameraProfileCfg(
                start=str(p["start"]),
                end=str(p["end"]),
                exposure_us=int(p["exposure_us"]),
                gain_db=int(p["gain_db"]),
                gamma_enable=bool(p.get("gamma_enable", True)),
                gamma=float(p.get("gamma", 1.0)),
            )
            for name, p in profiles_raw.items()
        }

    reconnect_raw = settings.get("reconnect", cam.get("reconnect", {})) or {}
    reconnect_cfg = CameraReconnectCfg(
        max_retries=int(reconnect_raw.get("max_retries", 5)),
        sleep_s=float(reconnect_raw.get("sleep_s", 1.0)),
    )

    basler_cfg = BaslerCameraCfg(
        device_user_id=str(basler_raw.get("device_user_id", "") or ""),
        serial_number=str(basler_raw.get("serial_number", "") or ""),
        ip_address=str(basler_raw.get("ip_address", "") or ""),
        exposure_time=_optional_float(basler_raw.get("exposure_time")),
        gain=_optional_float(basler_raw.get("gain")),
        grab_strategy=str(basler_raw.get("grab_strategy", "latest_image_only") or "latest_image_only"),
        timeout_ms=int(basler_raw.get("timeout_ms", 1000)),
        pixel_format=(str(basler_raw["pixel_format"]) if basler_raw.get("pixel_format") else None),
        width=_optional_int(basler_raw.get("width")),
        height=_optional_int(basler_raw.get("height")),
        offset_x=_optional_int(basler_raw.get("offset_x")),
        offset_y=_optional_int(basler_raw.get("offset_y")),
        acquisition_frame_rate=_optional_float(basler_raw.get("acquisition_frame_rate")),
        packet_size=_optional_int(basler_raw.get("packet_size")),
        inter_packet_delay=_optional_int(basler_raw.get("inter_packet_delay")),
        max_num_buffer=int(basler_raw.get("max_num_buffer", 20)),
    )

    return CameraCfg(
        type=camera_type,
        id=settings.get("id", 0),
        width=int(settings.get("width", 960)),
        height=int(settings.get("height", 640)),
        fps=int(settings.get("fps", 8)),
        auto_exposure=bool(settings.get("auto_exposure", False)),
        auto_gain=bool(settings.get("auto_gain", False)),
        va_imaging=va_raw,
        basler=basler_cfg,
        profiles=profiles,
        reconnect=reconnect_cfg,
    )


def load_caster_file_config(caster_id: int | str, caster_config_path: str | None = None) -> CasterFileCfg:
    caster_id = resolve_caster_id(caster_id)
    key = caster_key(caster_id)
    default_dir = resolve_caster_config_dir(caster_id)
    legacy_path = Path(legacy_caster_config_path(caster_id))
    path = Path(caster_config_path) if caster_config_path else (default_dir if default_dir.exists() else legacy_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Missing caster config {path}. Create it or run with --caster-config <path>."
        )

    if path.is_dir():
        raw = _load_yaml(path / "config.yaml") if (path / "config.yaml").exists() else {}
        runtime = path / "runtime.yaml"
        rois = path / "rois.yaml"
        camera = path / "camera.yaml"
        plc = path / "plc.yaml"
        weight = path / "weight.yaml"
        bytetrack = path / "bytetrack.yaml"
        defaults = {
            "runtime": str(_first_existing(runtime, Path("config/runtime.yaml"))),
            "rois": str(_first_existing(rois, Path(f"config/casters/{key}_rois.yaml"), Path("config/rois.yaml"))),
            "camera": str(_first_existing(camera, Path(f"config/casters/{key}_camera.yaml"), Path("config/camera.yaml"))),
            "plc": str(_first_existing(plc, Path("config/plc.yaml"))),
            "weight": str(_first_existing(weight, Path("config/weight.yaml"))),
            "bytetrack": str(_first_existing(bytetrack, Path("config/bytetrack.yaml"))),
        }
    else:
        raw = _load_yaml(path)
        defaults = {
            "runtime": "config/runtime.yaml",
            "rois": f"config/casters/{key}_rois.yaml",
            "camera": f"config/casters/{key}_camera.yaml",
            "plc": "config/plc.yaml",
            "weight": "config/weight.yaml",
            "bytetrack": "config/bytetrack.yaml",
        }

    file_caster_id = resolve_caster_id(raw.get("caster_id", caster_id))
    if file_caster_id != caster_id:
        raise ValueError(
            f"Caster config {path} declares caster_id={file_caster_id}, "
            f"but CLI requested caster_id={caster_id}."
        )
    if not bool(raw.get("enabled", True)):
        raise ValueError(f"Caster {caster_id} is disabled in {path}.")

    return CasterFileCfg(
        caster_id=caster_id,
        caster_key=key,
        enabled=True,
        runtime=str(raw.get("runtime", defaults["runtime"])),
        rois=str(raw.get("rois", defaults["rois"])),
        camera=str(raw.get("camera", defaults["camera"])),
        plc=str(raw.get("plc", defaults["plc"])),
        weight=str(raw.get("weight", defaults["weight"])),
        bytetrack=str(raw.get("bytetrack", defaults["bytetrack"])),
        storage_dir=str(raw.get("storage_dir", resolve_caster_storage_path(caster_id))),
        overrides=dict(raw.get("overrides", {}) or {}),
    )


def load_config(
    runtime_path: str,
    rois_path: str,
    plc_path: str,
    camera_cfg_path: str = "config/camera.yaml",
    weight_cfg_path: str = "config/weight.yaml",
    runtime_overrides: Dict[str, Any] | None = None,
) -> AppCfg:
    r = _load_yaml(runtime_path)
    if runtime_overrides:
        r.update(runtime_overrides)
    rois_raw = _load_yaml(rois_path)
    p = _load_yaml(plc_path)
    c_raw = _load_yaml(camera_cfg_path)

    # Weight config is optional.
    weight_raw: Dict[str, Any] = {}
    if weight_cfg_path and Path(weight_cfg_path).exists():
        weight_raw = _load_yaml(weight_cfg_path)

    camera_cfg = _parse_camera_cfg(c_raw)

    gate_raw = r.get("gate", {}) or {}
    gate = GateRuntimeCfg(
        source_default=str(gate_raw.get("source_default", "geometry")),
        stable_frames=int(gate_raw.get("stable_frames", 3)),
        min_conf=float(gate_raw.get("min_conf", 0.25)),
        max_area_ratio_vs_closed=float(gate_raw.get("max_area_ratio_vs_closed", 0.85)),
        max_w_over_h=float(gate_raw.get("max_w_over_h", 0.9)),
        human_iou_occlusion=float(gate_raw.get("human_iou_occlusion", 0.10)),
    )
    history_raw = r.get("history") or {}
    history = HistoryCfg(
        enabled=bool(history_raw.get("enabled", False)),
        base_dir=str(history_raw.get("base_dir", "")),
        prefix=str(history_raw.get("prefix", "pipe")),
        ext=str(history_raw.get("ext", "jpg")),
        date_folder_format=str(history_raw.get("date_folder_format", "%Y-%m-%d")),
        time_filename_format=str(history_raw.get("time_filename_format", "%H-%M-%S-%f")),
        timezone=str(history_raw.get("timezone", "Asia/Kolkata")),
        shifts=list(history_raw.get("shifts", []) or []),
    )
    class_map_raw = r.get("class_name_to_id", {}) or {}
    class_name_to_id = {str(k): int(v) for k, v in class_map_raw.items()}

    raw_imgsz = r.get("publish_imgsz", 960)
    if isinstance(raw_imgsz, (list, tuple)) and len(raw_imgsz) == 2:
        publish_imgsz = (int(raw_imgsz[0]), int(raw_imgsz[1]))
    elif isinstance(raw_imgsz, int):
        publish_imgsz = raw_imgsz
    else:
        raise ValueError("publish_imgsz must be int or [width, height]")

    raw_device = r.get("device", "auto")
    if raw_device is None:
        device = None
    elif isinstance(raw_device, int):
        device = raw_device
    else:
        device = str(raw_device)

    runtime = RuntimeCfg(
        debug_mode=bool(r.get("debug_mode", False)),
        video_source=r.get("video_source", 0),
        model_path=r["model_path"],
        tracker_yaml=r["tracker_yaml"],
        imgsz=int(r.get("imgsz", 640)),
        conf=float(r.get("conf", 0.25)),
        iou=float(r.get("iou", 0.5)),
        device=device,
        half=bool(r.get("half", False)),
        max_fps=int(r.get("max_fps", 15)),
        frame_skip=int(r.get("frame_skip", 0)),
        update_fps=int(r.get("update_fps", 0)),
        db_path=str(r.get("db_path", "var/pipes.db")),
        latest_jpg_path=str(r.get("latest_jpg_path", "var/latest.jpg")),
        publish_fps=int(r.get("publish_fps", 5)),
        publish_imgsz=publish_imgsz,
        publish_overlay=bool(r.get("publish_overlay", True)),
        run_headless=bool(r.get("run_headless", False)),
        db_flush_interval_s=float(r.get("db_flush_interval_s", 1.0)),
        log_level=str(r.get("log_level", "INFO")),
        log_path=(str(r["log_path"]) if r.get("log_path") else None),
        origin_confirm_frames=int(r.get("origin_confirm_frames", 2)),
        loadcell_enter_confirm_frames=int(r.get("loadcell_enter_confirm_frames", 1)),
        loadcell_exit_confirm_frames=int(r.get("loadcell_exit_confirm_frames", 2)),
        class_name_to_id=class_name_to_id,
        stale_track_frames=int(r.get("stale_track_frames", 45)),
        rearm_empty_frames=int(r.get("rearm_empty_frames", 10)),
        min_pipe_gap_seconds=int(r.get("min_pipe_gap_seconds", 60)),
        loadcell_covered_per=int(r.get("loadcell_covered_per", 90)),
        remove_pipe_id_pipe_checkpoint_not_entered=bool(r.get("remove_pipe_id_pipe_checkpoint_not_entered", False)),
        gate=gate,
        history=history, 
    )

    plc = PlcCfg(
        mode=str(p.get("mode", "mock")),
        tags=dict(p.get("tags", {}) or {}),
        pulse_ms=int(p.get("pulse_ms", 300)),
        modbus=p.get("modbus"),
    )

    rois = _normalize_rois(rois_raw)

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

    return AppCfg(
        runtime=runtime,
        rois=rois,
        plc=plc,
        camera_cfg=camera_cfg,
        weight=weight_cfg,
        caster_id=1,
        caster_key="caster_1",
        caster_config_path=None,
        caster_storage_path=str(resolve_caster_storage_path(1)),
        rois_path=str(rois_path),
        camera_cfg_path=str(camera_cfg_path),
    )


def load_caster_config(caster_id: int | str, caster_config_path: str | None = None) -> AppCfg:
    caster_file = load_caster_file_config(caster_id, caster_config_path)
    runtime_overrides = dict(caster_file.overrides)
    runtime_overrides.setdefault("tracker_yaml", caster_file.bytetrack)

    cfg = load_config(
        runtime_path=caster_file.runtime,
        rois_path=caster_file.rois,
        plc_path=caster_file.plc,
        camera_cfg_path=caster_file.camera,
        weight_cfg_path=caster_file.weight,
        runtime_overrides=runtime_overrides,
    )

    storage_dir = Path(caster_file.storage_dir)
    runtime = replace(
        cfg.runtime,
        db_path=str(storage_dir / f"{caster_file.caster_key}_pipes.db"),
        latest_jpg_path=str(storage_dir / "latest.jpg"),
        log_path=(str(storage_dir / "pipe_detect.log") if cfg.runtime.log_path else None),
        history=(replace(cfg.runtime.history, base_dir=str(storage_dir / "history")) if cfg.runtime.history else None),
    )
    _ensure_parent_dir(runtime.db_path)
    _ensure_parent_dir(runtime.latest_jpg_path)
    _ensure_parent_dir(runtime.log_path)

    return replace(
        cfg,
        runtime=runtime,
        caster_id=caster_file.caster_id,
        caster_key=caster_file.caster_key,
        caster_config_path=str(caster_config_path or default_caster_config_path(caster_file.caster_id)),
        caster_storage_path=str(storage_dir),
        rois_path=caster_file.rois,
        camera_cfg_path=caster_file.camera,
    )
