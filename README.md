# Electrosteel Pipe Detection

Production pipe detection and tracking for one or more caster lines. The app runs YOLO + ByteTrack on a live camera or video file, applies ROI-based business logic for origin, loadcell, and gate events, writes results to caster-specific SQLite databases, and publishes the latest annotated frame for the dashboard.

## Overview

Each caster is treated as an independent runtime unit:

- One caster has one camera.
- Each caster has its own configuration directory.
- Each caster writes to its own storage directory under `var/`.
- Each caster uses its own SQLite database.
- To run multiple casters, start one app process per caster.

Example:

```text
config/casters/caster_1/
  camera.yaml
  bytetrack.yaml
  plc.yaml
  rois.yaml
  runtime.yaml
  weight.yaml

var/caster_1/
  caster_1_pipes.db
  latest.jpg
  pipe_detect.log
```

The same pattern works for `caster_2`, `caster_3`, and any positive caster id.

## Requirements

- Python `>=3.10,<3.11`
- Linux production host, tested for Jetson/Raspberry Pi style deployments
- Camera source:
  - Daheng GigE camera through Aravis/GStreamer, or
  - Basler camera through pylon/pypylon, or
  - USB/V4L2 camera such as `/dev/video0`, or
  - video file for testing
- YOLO model in `models/yolo/`
- Optional PLC connection, or `mock` PLC mode for testing

Install common system packages:

```bash
sudo apt update
sudo apt install -y \
  python3 \
  python3-venv \
  python3-pip \
  git \
  libatlas-base-dev \
  libopenblas-dev \
  libjpeg-dev \
  libpng-dev \
  libv4l-dev \
  libgl1 \
  libglib2.0-0
```

## Setup

Clone the repository:

```bash
git clone https://github.com/saikumar-voptimai/electrosteel_pipe_detection_prod.git
cd electrosteel_pipe_detection_prod
```

Create the Python environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Or with `uv`:

```bash
python3 -m pip install -U uv
uv venv
uv pip install -r requirements.txt
source .venv/bin/activate
```

Confirm that the configured model exists. The current runtime configs use:

```text
models/yolo/yolo11n_26_16_01.engine
```

## Caster Configuration

Use one directory per caster:

```text
config/casters/caster_<id>/
  camera.yaml
  bytetrack.yaml
  plc.yaml
  rois.yaml
  runtime.yaml
  weight.yaml
```

Important files:

- `runtime.yaml`: video source, model path, inference size, logging, publish FPS, headless mode.
- `camera.yaml`: camera id/name, width, height, FPS, exposure/gain profiles.
- `rois.yaml`: ROI polygons for that camera view.
- `plc.yaml`: PLC mode, event tags, Modbus settings.
- `bytetrack.yaml`: ByteTrack settings.
- `weight.yaml`: optional S7 weight capture settings.

For a new caster, copy an existing directory and edit only the hardware-specific values:

```bash
cp -r config/casters/caster_1 config/casters/caster_3
```

Then update:

- `config/casters/caster_3/runtime.yaml`
- `config/casters/caster_3/camera.yaml`
- `config/casters/caster_3/plc.yaml`
- `config/casters/caster_3/weight.yaml`, if weight capture is enabled

The app also keeps backward compatibility with older flat files like `config/casters/caster_1_config.yaml`, but the directory layout above is the preferred format.

## Storage And Database

Storage is resolved dynamically from the caster id.

For `caster_1`:

```text
var/caster_1/caster_1_pipes.db
var/caster_1/latest.jpg
var/caster_1/pipe_detect.log
var/caster_1/history/
```

For `caster_2`:

```text
var/caster_2/caster_2_pipes.db
var/caster_2/latest.jpg
var/caster_2/pipe_detect.log
var/caster_2/history/
```

You do not need to create these directories manually. They are created when the caster config is loaded.

## How To Run

Activate the environment first:

```bash
source .venv/bin/activate
```

Run caster 1:

```bash
python src/app.py --caster 1
```

Equivalent form:

```bash
python src/app.py --caster caster_1
```

Run caster 2:

```bash
python src/app.py --caster 2
```

Run multiple casters:

```bash
python scripts/run_all_casters.py --casters 1,2,3,4
```

The launcher starts one child process per caster. Stop it with `Ctrl+C`; it will terminate the child processes.

`src/main.py` remains as a compatibility wrapper and defaults to caster 1:

```bash
python src/main.py
```

## First-Time ROI Setup

Each caster needs ROIs drawn for its actual camera view. Run the ROI wizard before production use.

For caster 1:

```bash
python src/app.py --caster 1 --redraw
```

For caster 2:

```bash
python src/app.py --caster 2 --redraw
```

Override the video source during ROI setup:

```bash
python src/app.py --caster 1 --redraw --video-source 0
python src/app.py --caster 1 --redraw --video-source test1.mp4
```

The wizard saves to:

```text
config/casters/caster_<id>/rois.yaml
```

Wizard controls:

- Left click: add a point.
- `u`: undo last point.
- `c`: clear current ROI.
- `Enter`: accept the current ROI after exactly 4 points.
- `q`: quit without saving.

Required ROIs:

- `roi_loadcell`
- `roi_caster_origin`
- `roi_left_origin`
- `roi_right_origin`
- `roi_safety_critical`
- `roi_gate1_closed`
- `roi_gate1_open`
- `roi_gate2_closed`
- `roi_gate2_open`

ROIs are stored in the original capture coordinate space, not the resized inference or publish image size.

## Dashboard

Run the centralized monitoring dashboard:

```bash
source .venv/bin/activate
streamlit run src/ui/dashboard.py
```

The sidebar has a `Caster` selector:

- `All Casters`: shows aggregated production metrics, health for every configured caster, and a camera preview grid.
- `caster_1`, `caster_2`, ...: switches all metrics, latest frame, recent pipes, logs, and configuration display to that caster.

Available casters are discovered dynamically from:

```text
config/casters/caster_<id>/
```

To open the dashboard with a caster preselected:

```bash
PIPE_DASHBOARD_CASTER=caster_1 streamlit run src/ui/dashboard.py
PIPE_DASHBOARD_CASTER=caster_2 streamlit run src/ui/dashboard.py
```

The dashboard reads each caster's runtime outputs:

```text
var/caster_<id>/caster_<id>_pipes.db
var/caster_<id>/latest.jpg
var/caster_<id>/pipe_detect.log
```

For a selected VA Imaging caster, open the `Camera Control` tab to manage its
scheduled exposure, gain, gamma, auto-exposure, and auto-gain settings. New
casters start with day/night defaults. Use the table's row controls to add or
remove any number of custom periods, including overnight periods such as
`20:00`–`06:00`. The schedule must cover all 24 hours without gaps or overlaps.

Saving updates only that caster's `camera.yaml`. Its independent background
camera scheduler notices the atomic file change within about two seconds and
applies the active profile; the inference loop does not poll configuration.

## Camera Configuration

Set the camera source in the caster runtime file:

```yaml
# config/casters/caster_1/runtime.yaml
video_source: gige
```

Common values:

```yaml
video_source: gige
video_source: basler
video_source: 0
video_source: "/dev/video0"
video_source: "test1.mp4"
```

The camera implementation is selected by `camera.type` in:

```text
config/casters/caster_<id>/camera.yaml
```

Supported camera types:

- `va_imaging`
- `basler`

### VA Imaging Example

Use `va_imaging` for the existing Daheng / VA Imaging GigE setup:

```yaml
camera:
  type: va_imaging
  va_imaging:
    id: "Daheng Imaging-MER2-630-18GC-P-FBJ24120608"
    width: 2620
    height: 1216
    fps: 8
    reconnect:
      max_retries: 3
      sleep_s: 2.0
    profiles:
      day:
        start: "06:00"
        end: "18:00"
        exposure_us: 100000
        gain_db: 5
        gamma_enable: false
        gamma: 0.8
      night:
        start: "18:00"
        end: "06:00"
        exposure_us: 150000
        gain_db: 12
        gamma_enable: false
        gamma: 1.4
    auto_exposure: false
    auto_gain: false
```

Then run:

```bash
python src/app.py --caster 1
```

### Basler Example

Use `basler` for Basler cameras through pypylon:

```yaml
camera:
  type: basler
  width: 1920
  height: 1280
  fps: 5
  basler:
    device_user_id: ""
    serial_number: ""
    ip_address: ""
    exposure_time: null
    gain: null
    grab_strategy: latest_image_only
    timeout_ms: 1000
    pixel_format: Mono8
    width: 1920
    height: 1280
    offset_x: 0
    offset_y: 0
    acquisition_frame_rate: 5
    packet_size: 1500
    inter_packet_delay: 1000
    max_num_buffer: 30
```

If `serial_number`, `device_user_id`, or `ip_address` is set, the app opens that matching Basler camera. If all are empty, it opens the first available Basler camera.

When Basler and VA Imaging GigE cameras run on the same switch or NIC, do not run the Basler at full sensor size unless the network is designed for it. For example, a 5472 x 3648 Mono8 Basler frame at roughly 5 FPS is about 100 MB/s before the VA camera traffic is added. Use `width`, `height`, `acquisition_frame_rate`, `inter_packet_delay`, and `max_num_buffer` to keep the stream stable. If you change Basler width/height, redraw that caster's ROIs because ROI coordinates are stored in the original camera coordinate space.

Basler setup:

```bash
python -m pip install pypylon
```

Install the Basler pylon runtime/SDK on the target machine before using `pypylon`. The Basler dependency is loaded only when `camera.type: basler` is selected, so VA Imaging and video-file modes can still run without `pypylon` installed.

For USB/V4L2 cameras, use an index such as `0` or a device path such as `/dev/video0`.

## PLC Configuration

For development without a PLC:

```yaml
mode: "mock"
```

For Modbus:

```yaml
mode: "modbus"
```

Each caster pulse tag is resolved dynamically by key:

```yaml
tags:
  caster_1_new: "caster_1_new"
  caster_2_new: "caster_2_new"
```

If a caster-specific tag is missing and PLC mode is `mock`, the app uses a synthetic mock tag. In real Modbus mode, the tag must also exist under `modbus.coils`.

## Headless Production Service

Set `run_headless: true` in the caster runtime config for production systems without a monitor:

```yaml
run_headless: true
```

Example `systemd` service for caster 1:

```ini
[Unit]
Description=Pipe Detection Caster 1
After=network.target

[Service]
WorkingDirectory=/home/pi/electrosteel_pipe_detection_prod
ExecStart=/home/pi/electrosteel_pipe_detection_prod/.venv/bin/python src/app.py --caster 1
Restart=always
RestartSec=2

[Install]
WantedBy=multi-user.target
```

For multiple casters, create one service per caster or run the launcher:

```ini
ExecStart=/home/pi/electrosteel_pipe_detection_prod/.venv/bin/python scripts/run_all_casters.py --casters 1,2,3,4
```

## Test Video Workflow

Set a caster runtime source to a video file:

```yaml
video_source: "test1.mp4"
```

Then run:

```bash
python src/app.py --caster 1
```

The database persists between runs. To reset caster 1 test data:

```bash
rm var/caster_1/caster_1_pipes.db
```

## Architecture

Runtime flow:

1. `Capture` reads frames from camera or video.
2. `YoloByteTrack` runs YOLO + ByteTrack.
3. `ROIManager` evaluates detections against configured ROIs.
4. `PipeFlowFSM` detects origin, loadcell enter/exit, stale tracks, and pipe lifecycle.
5. `GateFSM` detects gate open/close using geometry, PLC, or vision sources.
6. `SqliteRepo` writes pipe state and event history.
7. `LatestFramePublisher` writes the current dashboard image.

Main modules:

- `src/app.py`: application lifecycle and runtime orchestration.
- `src/utils/config.py`: config loading, caster id resolution, storage and DB path helpers.
- `src/camera/capture.py`: camera/video capture.
- `src/vision/tracker.py`: YOLO + ByteTrack wrapper.
- `src/geometry/roi.py`: ROI geometry helpers.
- `src/logic/pipe_fsm.py`: pipe lifecycle state machine.
- `src/logic/gate_fsm.py`: gate state machine.
- `src/db/repo.py`: SQLite schema and queries.
- `src/ui/dashboard.py`: Streamlit dashboard.

## Validation

Run the available unit tests:

```bash
python -m unittest discover tests
```

Run only caster config tests:

```bash
python -m unittest tests.test_caster_config
```

Compile-check the main Python files:

```bash
python -m py_compile src/utils/config.py src/app.py src/main.py src/ui/dashboard.py scripts/run_all_casters.py
```

## Troubleshooting

No camera frames:

- Check `video_source` in `config/casters/caster_<id>/runtime.yaml`.
- For V4L2 cameras, run `ls -l /dev/video*`.
- For GigE cameras, verify network, camera id, Aravis/GStreamer installation, and camera permissions.

ROI wizard does not open:

- Run it on a machine with a display.
- Avoid headless OpenCV builds when using the wizard.
- Generate `config/casters/caster_<id>/rois.yaml` and copy it to the production host.

Dashboard shows old counts:

- The SQLite DB persists across runs.
- Delete the caster DB to reset counts:

```bash
rm var/caster_1/caster_1_pipes.db
```

Too much logging:

- Set `log_level: "INFO"` in `config/casters/caster_<id>/runtime.yaml`.
- Set `log_path: null` to log only to console.

Wrong caster output path:

- Run with an explicit caster id:

```bash
python src/app.py --caster caster_1
```

- Confirm output under:

```text
var/caster_1/
```
