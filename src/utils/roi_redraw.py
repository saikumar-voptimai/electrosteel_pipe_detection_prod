from __future__ import annotations
from pathlib import Path
from typing import Any

def _load_camera_cfg(camera_cfg_path: str):
    try:
        import yaml
        from utils.config import BaslerCameraCfg, CameraCfg, CameraProfileCfg
    except Exception:
        return None

    p = Path(camera_cfg_path)
    if not p.exists():
        return None

    raw = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    cam = (raw.get("camera") if isinstance(raw, dict) else None) or (raw or {})

    if not isinstance(cam, dict) or not cam:
        return None

    camera_type = str(cam.get("type", "va_imaging")).strip().lower()
    va_raw = dict(cam.get("va_imaging", {}) or {})
    basler_raw = dict(cam.get("basler", {}) or {})
    common_raw = dict(cam)
    for key in ("type", "va_imaging", "basler"):
        common_raw.pop(key, None)
    settings = {**common_raw, **va_raw} if camera_type == "va_imaging" else common_raw

    profiles_raw = settings.get("profiles", {}) or {}

    profiles = None
    if profiles_raw:
        profiles = {
            name: CameraProfileCfg(
                start=str(p["start"]),
                end=str(p["end"]),
                exposure_us=int(p["exposure_us"]),
                gain_db=int(p["gain_db"]),
                gamma_enable=bool(p.get("gamma_enable", True)),
                gamma=float(p.get("gamma", 1.0)),
            )
            for name, p in profiles_raw.items()
        }

    return CameraCfg(
        type=camera_type,
        id=settings.get("id", 0),
        width=int(settings.get("width", 960)),
        height=int(settings.get("height", 640)),
        fps=int(settings.get("fps", 8)),
        auto_exposure=bool(settings.get("auto_exposure", False)),
        auto_gain=bool(settings.get("auto_gain", False)),
        va_imaging=va_raw,
        basler=BaslerCameraCfg(
            device_user_id=str(basler_raw.get("device_user_id", "") or ""),
            serial_number=str(basler_raw.get("serial_number", "") or ""),
            ip_address=str(basler_raw.get("ip_address", "") or ""),
            exposure_time=(float(basler_raw["exposure_time"]) if basler_raw.get("exposure_time") is not None else None),
            gain=(float(basler_raw["gain"]) if basler_raw.get("gain") is not None else None),
            grab_strategy=str(basler_raw.get("grab_strategy", "latest_image_only") or "latest_image_only"),
            timeout_ms=int(basler_raw.get("timeout_ms", 1000)),
            pixel_format=(str(basler_raw["pixel_format"]) if basler_raw.get("pixel_format") else None),
            width=(int(basler_raw["width"]) if basler_raw.get("width") is not None else None),
            height=(int(basler_raw["height"]) if basler_raw.get("height") is not None else None),
            offset_x=(int(basler_raw["offset_x"]) if basler_raw.get("offset_x") is not None else None),
            offset_y=(int(basler_raw["offset_y"]) if basler_raw.get("offset_y") is not None else None),
            acquisition_frame_rate=(
                float(basler_raw["acquisition_frame_rate"])
                if basler_raw.get("acquisition_frame_rate") is not None
                else None
            ),
            packet_size=(int(basler_raw["packet_size"]) if basler_raw.get("packet_size") is not None else None),
            inter_packet_delay=(
                int(basler_raw["inter_packet_delay"]) if basler_raw.get("inter_packet_delay") is not None else None
            ),
            max_num_buffer=int(basler_raw.get("max_num_buffer", 20)),
        ),
        profiles=profiles,
    )

def run_roi_redraw(video_source: int | str, rois_path: str, camera_cfg_path: str = "config/camera.yaml") -> None:
    """
    Adapter around your utils/roi_wizard.py.
    Tries a few known APIs so you don't have to modify your wizard.
    """
    import utils.roi_wizard as w

    first_frame = None
    cam_cfg = _load_camera_cfg(camera_cfg_path)
    source_name = str(video_source).strip().lower() if isinstance(video_source, str) else ""
    use_camera_client = cam_cfg is not None and (
        source_name.startswith("gige") or source_name in ("basler", "va_imaging")
    )
    if use_camera_client:
        try:
            from camera.capture import Capture
        except Exception:
            Capture = None

        if Capture is not None:
            cap = Capture(source=video_source, camera_cfg=cam_cfg)
            cap.open()
            item = cap.read()
            if item is None:
                raise RuntimeError(f"Failed to read first frame from {cam_cfg.type} camera for ROI redraw")
            first_frame, _ts = item
            cap.close()
    
    # Pattern 0: run_roi_wizard_and_save(video_source, out_path)
    if hasattr(w, "run_roi_wizard_and_save") and callable(getattr(w, "run_roi_wizard_and_save")):
        w.run_roi_wizard_and_save(video_source=video_source, out_path=rois_path)
        return

    # Pattern 1: function redraw_rois(video_source, rois_path)
    if hasattr(w, "redraw_rois") and callable(getattr(w, "redraw_rois")):
        w.redraw_rois(video_source=video_source, rois_path=rois_path)
        return

    # Pattern 2: class ROIRedrawWizard(cfg, specs).save()
    if all(hasattr(w, k) for k in ("ROIRedrawWizard", "ROIWizardConfig", "default_roi_specs")):
        cfg = w.ROIWizardConfig(video_source=video_source, rois_path=rois_path, first_frame=first_frame)
        wizard = w.ROIRedrawWizard(cfg=cfg, specs=w.default_roi_specs())
        wizard.save()
        return

    # Pattern 3: main() style with globals
    if hasattr(w, "main") and callable(getattr(w, "main")):
        # If your wizard reads cli args itself, just call main.
        w.main()
        return

    raise RuntimeError(
        "Could not find a compatible API in utils/roi_wizard.py. "
        "Expected one of: redraw_rois(), "
        "ROIRedrawWizard/ROIWizardConfig/default_roi_specs, or main()."
    )
