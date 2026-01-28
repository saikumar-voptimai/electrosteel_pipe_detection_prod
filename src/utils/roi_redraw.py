from __future__ import annotations
from pathlib import Path
from typing import Any

def run_roi_redraw(video_source: int | str, rois_path: str) -> None:
    """
    Adapter around your utils/roi_wizard.py.
    Tries a few known APIs so you don't have to modify your wizard.
    """
    import utils.roi_wizard as w
    
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
        cfg = w.ROIWizardConfig(video_source=video_source, rois_path=rois_path)
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
