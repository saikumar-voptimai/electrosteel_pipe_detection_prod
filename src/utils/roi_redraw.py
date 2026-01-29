from __future__ import annotations
from pathlib import Path
from typing import Any

def _load_camera_cfg(camera_cfg_path: str):
    try:
        import yaml
        from utils.config import CameraCfg
    except Exception:
        return None

    p = Path(camera_cfg_path)
    if not p.exists():
        return None

    raw = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    cam = (raw.get("camera") if isinstance(raw, dict) else None) or (raw or {})
    if not isinstance(cam, dict) or not cam:
        return None

    return CameraCfg(
        id=cam.get("id", 0),
        width=int(cam.get("width", 960)),
        height=int(cam.get("height", 640)),
        fps=int(cam.get("fps", 8)),
        exposure_us=int(cam.get("exposure_us", 15000)),
        gain_db=int(cam.get("gain_db", 5)),
        auto_exposure=bool(cam.get("auto_exposure", False)),
        auto_gain=bool(cam.get("auto_gain", False)),
    )


def run_roi_redraw(video_source: int | str, rois_path: str, camera_cfg_path: str = "config/camera.yaml") -> None:
    """
    Adapter around your utils/roi_wizard.py.
    Tries a few known APIs so you don't have to modify your wizard.
    """
    import utils.roi_wizard as w

    first_frame = None
    if isinstance(video_source, str) and video_source.startswith("gige"):
        try:
            from camera.capture import Capture
        except Exception:
            Capture = None

        if Capture is not None:
            cam_cfg = _load_camera_cfg(camera_cfg_path)
            cap = Capture(source=video_source, camera_cfg=cam_cfg)
            cap.open()
            item = cap.read()
            if item is None:
                raise RuntimeError("Failed to read first frame from GigE camera for ROI redraw")
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
