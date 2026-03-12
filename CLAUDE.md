# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Real-time pipe detection and tracking system for an industrial steel processing line. Uses YOLOv11 + ByteTrack on a live camera feed (USB or Daheng GigE), applies ROI-based finite state machine logic (pipe origin, loadcell events, gate monitoring), persists results to SQLite, and publishes an annotated frame for a Streamlit dashboard.

## Commands

```bash
# Setup
python3 -m pip install -U uv && uv venv && uv pip install -r requirements.txt

# Run main application
python src/main.py

# Run ROI wizard (interactive polygon drawing)
python src/main.py --redraw
python src/main.py --redraw --video-source 0

# Run dashboard (separate terminal)
streamlit run src/ui/dashboard.py

# Config overrides
python src/main.py --runtime config/runtime.yaml --rois config/rois.yaml --plc config/plc.yaml --camera config/camera.yaml --weight config/weight.yaml
```

Python 3.11 target (see `.python-version`). Tests exist as stubs only (`tests/test_*.py` are empty).

## Architecture

**Data pipeline:** Capture → YOLO+ByteTrack Inference → FSM Processing → DB Persistence → Overlay Visualization → Dashboard

Entry point is `src/main.py` which either launches the ROI wizard (`--redraw`) or runs `App(cfg).run()` from `src/app.py`.

### Key modules

- **`src/app.py`** — Main orchestrator loop: read frame, resize for inference, run tracker, update FSMs, commit to DB, publish annotated frame
- **`src/vision/tracker.py`** — Ultralytics YOLO + ByteTrack wrapper, returns `TrackDet` objects
- **`src/logic/pipe_fsm.py`** — Pipe origin determination (caster vs other) with debouncing, loadcell enter/exit events, stale track cleanup
- **`src/logic/gate_fsm.py`** + **`gate_sources.py`** — Gate open/close debouncing with pluggable sources (geometry/PLC/vision)
- **`src/geometry/roi.py`** — Polygon ROI storage and point-in-polygon checks (`cv2.pointPolygonTest`)
- **`src/db/repo.py`** — SQLite (WAL mode) with tables: `pipes`, `events`, `settings`
- **`src/plc/`** — Abstract `PLCClient` with Mock, Modbus, and S7 implementations
- **`src/ui/dashboard.py`** — Streamlit app reading from `var/pipes.db` and `var/latest.jpg`
- **`src/utils/roi_names.py`** — `RoiName` enum centralizing all ROI name constants

### Coordinate system (critical)

Three separate frame sizes are maintained throughout the pipeline:

1. **Capture/original size** — raw camera resolution. ROIs in `config/rois.yaml` are in this space.
2. **Inference `imgsz`** — YOLO operates on scaled frames (e.g. 640px wide). Detections are mapped back to original coords via `inv_scale_x/y`.
3. **Publish `publish_imgsz`** — visualization-only resolution for `var/latest.jpg`. Mapped via `vis_scale_x/y`.

ROIs are always defined and stored in original frame coordinates. Scaling factors are computed in `app.py` and passed to overlay/geometry functions.

### Configuration files (all YAML in `config/`)

- **`runtime.yaml`** — video source, model path, inference params, FPS limits (`max_fps`, `update_fps`, `publish_fps`), FSM debounce thresholds, gate config, logging
- **`rois.yaml`** — 4-point polygon coordinates for each ROI (loadcell, caster_origin, left/right_origin, safety_critical, gate1/2 open/closed)
- **`plc.yaml`** — PLC mode (`mock`/`modbus`), tag names, Modbus host/port/coils
- **`camera.yaml`** — Daheng GigE camera settings (resolution, fps, exposure, gain)
- **`bytetrack.yaml`** — ByteTrack tracker parameters
- **`weight.yaml`** — Siemens S7 weight machine config

### Runtime outputs (in `var/`)

- `pipes.db` — SQLite database (persists across runs, delete to reset)
- `latest.jpg` — latest annotated frame for dashboard
- `pipe_detect.log` — application logs

## Conventions

- Frozen dataclasses for immutable value types (`BBox`, `TrackDet`), regular dataclasses for mutable state (`PipeStats`, `GateFSM`)
- `RoiName` enum (`src/utils/roi_names.py`) for all ROI name references — never use raw strings
- Config loaded via typed dataclasses in `src/utils/config.py`
- Centralized logging per module: `logger = logging.getLogger(__name__)`
- PLC abstraction: all hardware communication goes through `PLCClient` interface with factory pattern (`src/plc/factory.py`)
