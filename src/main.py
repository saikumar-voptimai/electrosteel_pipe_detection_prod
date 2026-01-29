# entrypoint: wires everything together
from __future__ import annotations
import argparse
from pathlib import Path

from utils.config import load_config
from utils.roi_redraw import run_roi_redraw
from app import App

def parse_args() -> argparse.Namespace:
  p = argparse.ArgumentParser()
  p.add_argument("--runtime", default="config/runtime.yaml", help="Path to runtime config YAML.")
  p.add_argument("--rois", default="config/rois.yaml", help="Path to ROIs config YAML.")
  p.add_argument("--plc", default="config/plc.yaml", help="Path to PLC config YAML.")
  p.add_argument("--camera", default="config/camera.yaml", help="Path to camera config YAML (GigE/Aravis).")
  p.add_argument("--redraw", action="store_true", help="Launch ROI redraw wizard instead of main app.")
  p.add_argument("--video-source", default=None, help="Only used with --redraw")
  return p.parse_args()

def main() -> None:
  args = parse_args()

  if args.redraw:
    if args.video_source is None:
      # Default to whatever runtime.yaml is configured for (e.g. "gige")
      import yaml
      raw = yaml.safe_load(Path(args.runtime).read_text(encoding="utf-8")) or {}
      video_source = raw.get("video_source", 0)
    else:
      video_source = args.video_source
    Path("config").mkdir(exist_ok=True)
    run_roi_redraw(video_source=video_source, rois_path=args.rois, camera_cfg_path=args.camera)
    print(f"[OK] Saved ROIs to {args.rois}")
    return
  
  if not Path(args.rois).exists():
    raise SystemExit(f"Missing {args.rois} Run with --redraw to create ROIs.")
  
  cfg = load_config(args.runtime, args.rois, args.plc, args.camera)
  if not cfg.runtime.run_headless:
    # Only set if we aren't in headless mode
    import os
    os.environ["QT_QPA_PLATFORM"] = "wayland"
  App(cfg).run()

if __name__ == "__main__":
  main()