
# Pipe Detect (Production Model)

Runs YOLO + ByteTrack on a live camera feed (or a video file), detects/ tracks pipes, applies ROI-based business logic (origin, loadcell, gate), writes results to a local SQLite database, and publishes a continuously-updated annotated frame for a simple dashboard.

Key sizing concepts (kept intentionally separate):

- **Capture size**: whatever the camera/video provides (or, for GigE, what you request in `config/camera.yaml`). ROIs in `config/rois.yaml` are defined in this coordinate space.
- **Inference `imgsz`** (in `config/runtime.yaml`): passed to Ultralytics YOLO; YOLO resizes internally for inference but returns boxes back in the original input frame coordinates.
- **Publish `publish_imgsz`** (in `config/runtime.yaml`): affects only the rendered/published visualization frame (`var/latest.jpg` and the OpenCV window). It does not affect business logic.

This README is ordered intentionally:

1) **How to run on a Raspberry Pi with a real camera**
2) **How to use the ROI wizard**
3) **What each part of the codebase does**

---

## 1) Raspberry Pi: end-to-end setup (camera + model + dashboard)

### Hardware / OS assumptions

- Raspberry Pi 4/5 (4GB+ recommended)
- Raspberry Pi OS (Debian-based)
- A camera source:
	- **Recommended / simplest:** a USB UVC camera
	- **Pi CSI camera module:** works if the camera is exposed as a V4L2 device (`/dev/video0`). See the CSI notes below.

### Install system packages

OpenCV needs system libraries; headless setups also need video backends.

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

Notes:
- If you plan to run **without a monitor** (headless), it is better to avoid UI windows (see “Headless/Service mode” below).

### Get the code and model

```bash
git clone https://github.com/saikumar-voptimai/electrosteel_pipe_detection_prod.git
cd pipe_detect_prod_model
```

Make sure the model file exists:

- `models/yolo/best_nano_dataset0To5.pt`

### Python environment

This project targets **Python 3.11** (see `pyproject.toml`).
#### Option: use uv

If you prefer `uv` (fast installs):

```bash
python3 -m pip install -U uv
uv venv
uv pip install -r requirements.txt
```

### Configure the runtime for a camera

Edit `config/runtime.yaml`:

- `video_source`: can be
	- `0` (first camera)
	- `1` (second camera)
	- `/dev/video0` (explicit V4L2 device)
	- a path to a video file

Example for a USB camera:

```yaml
video_source: 0
```

### Configure Daheng GigE (Aravis/GStreamer)

If `video_source: "gige"`, camera settings are taken from `config/camera.yaml` (camera name/id, width/height/fps, exposure, gain, auto flags). The GigE pipeline in `src/camera/capture.py` uses these values.

Logging controls (already present in `config/runtime.yaml`):

```yaml
log_level: "INFO"        # set to DEBUG for very verbose logs
log_path: "var/pipe_detect.log"  # set to null for console-only
```

### Configure PLC mode

Edit `config/plc.yaml`:

- For development/testing on a Pi with no PLC connected, keep:
	- `mode: "mock"`
- For Modbus/OpenPLC style wiring, set `mode: "modbus"` and configure `modbus:`.

### Run the ROI wizard (mandatory first-time step)

You must define ROIs in `config/rois.yaml` for your *actual camera view*.

See the next section for detailed ROI wizard instructions.

### Run the main application

From the repo root:

```bash
source .venv/bin/activate
python src/main.py
```

What you should see:

- A window showing the annotated live feed (if OpenCV GUI is available)
- A SQLite DB file at `var/pipes.db`
- A continuously-updated image at `var/latest.jpg` (used by the dashboard)
- Logs in console and optionally in `var/pipe_detect.log` (depending on `config/runtime.yaml`)

### Run the dashboard

In another terminal:

```bash
source .venv/bin/activate
streamlit run src/ui/dashboard.py
```

Dashboard inputs/outputs:

- Reads `var/pipes.db`
- Displays `var/latest.jpg`

### Headless / service mode (recommended for production)

If your Pi will run without a monitor:

- You may want to disable the OpenCV GUI window usage. Right now the app calls `cv2.imshow(...)` in `src/app.py`.
- A common approach is running under `systemd` and relying on logs + dashboard.

Minimal `systemd` example (adjust paths):

```ini
[Unit]
Description=Pipe Detect
After=network.target

[Service]
WorkingDirectory=/home/pi/pipe_detect_prod_model
ExecStart=/home/pi/pipe_detect_prod_model/.venv/bin/python src/main.py
Restart=always
RestartSec=2

[Install]
WantedBy=multi-user.target
```

---

## CSI camera module notes (Pi Camera)

This code uses `cv2.VideoCapture(...)`, which works best when the camera is available as a V4L2 device (e.g. `/dev/video0`).

If your CSI camera does not show up as `/dev/video*`:

- Ensure the camera is enabled and working with `libcamera-hello`.
- Consider using `libcamera-v4l2` (if available for your OS) to expose a V4L2 device.

If you can see the camera as `/dev/video0`, set:

```yaml
video_source: "/dev/video0"
```

---

## 2) ROI Wizard: how to use it (roi_wizard)

The ROI wizard is an interactive OpenCV tool that captures a frame and lets you draw a set of required ROIs as **4-point polygons**.

### Launch the wizard

From repo root:

```bash
python src/main.py --redraw
```

To use a specific camera index:

```bash
python src/main.py --redraw --video-source 0
```

To use a video file (useful for ROI setup on test footage):

```bash
python src/main.py --redraw --video-source tests/videos/va_imaging_test.avi
```

Output:

- Saves to `config/rois.yaml` by default (or `--rois <path>` if provided)

### Controls

The wizard window title shows the controls; the important ones:

- **Left click**: add a point (each ROI needs exactly **4 points**)
- `u`: undo last point
- `c`: clear current ROI points
- `ENTER`: accept current ROI (only works when 4 points are selected)
- `q`: quit without saving (cancels)

### What ROIs are collected (and what they mean)

The wizard collects these ROIs in order (see `src/utils/roi_wizard.py`):

- `roi_loadcell`: “loadcell zone” for triggering the loadcell-enter event
- `roi_caster5_origin`: where a pipe must appear to be considered originating from caster
- `roi_left_origin`, `roi_right_origin`: exclusion areas; pipes originating here become `origin='other'`
- `roi_safety_critical`: used for safety/occlusion logic (e.g., humans near gates)
- `roi_gate1_closed`, `roi_gate1_open`: reference boxes for gate geometry decision
- `roi_gate2_closed`, `roi_gate2_open`: reference boxes for gate geometry decision

### Tips for reliable ROIs

- Draw tight rectangles aligned to the physical zones.
- Keep origin ROIs large enough to “catch” the pipe early, but not so large that unrelated areas are included.
- The gate “closed” ROI is used as an area baseline, so draw it consistently.

---

## 3) Running on a test video (developer workflow)

To run with the bundled test video, set in `config/runtime.yaml`:

```yaml
video_source: "tests/videos/va_imaging_test.avi"
```

Then:

```bash
python src/main.py
```

Important note about “last hour/8h/24h” metrics:

- Metrics are computed using wall-clock timestamps (`time.time()`), and the DB (`var/pipes.db`) persists across runs.
- If you re-run a 5-minute test video multiple times, you are counting data from previous runs unless you delete `var/pipes.db`.

---

## 4) What parts of the code do what

### High-level dataflow

1) **Capture** reads a frame from the camera/video.
2) **Tracker** runs YOLO + ByteTrack to produce detections with track IDs.
3) **FSMs** interpret tracks relative to ROIs:
	 - pipe origin detection
	 - loadcell enter/exit events
	 - gate open events (via geometry/PLC/vision)
4) **DB repo** upserts pipe state and inserts events.
5) **Overlay** draws ROIs + boxes and publishes `var/latest.jpg`.
6) **Dashboard** reads the DB and latest image to display status.

### Entry points

- `src/main.py`
	- `python src/main.py` runs the full app.
	- `python src/main.py --redraw` runs ROI wizard and writes `config/rois.yaml`.

### Configuration

- `config/runtime.yaml`
	- Video source, model path, tracker thresholds, FPS control, DB paths, logging.
- `config/rois.yaml`
	- ROI polygons in pixel coordinates.
- `config/plc.yaml`
	- PLC mode (mock/modbus/...) and tag names.

### Core runtime

- `src/app.py`
	- Orchestrates capture → infer → FSM updates → DB → overlay publishing.
	- Periodic DB commits.
	- Gate source can be switched at runtime via DB setting.

### Vision / tracking

- `src/vision/tracker.py`
	- Wrapper around Ultralytics YOLO tracking with ByteTrack.
	- Produces `TrackDet` entries with `track_id`.

- `src/vision/overlay.py`
	- Draws ROIs and tracking boxes.
	- Publishes the latest annotated frame as a single JPEG file.

### Geometry / ROIs

- `src/geometry/roi.py`
	- ROI management and point-in-polygon checks.

- `src/utils/roi_wizard.py`
	- Interactive ROI drawing wizard.

### Business logic (FSM)

- `src/logic/pipe_fsm.py`
	- Determines pipe origin (`caster` vs `other`), loadcell enter/exit, and “stale track” cleanup.

- `src/logic/gate_fsm.py`
	- Debounces gate open/close transitions.

- `src/logic/gate_sources.py`
	- Gate position sources:
		- geometry-based (from detections + ROIs)
		- PLC-based
		- placeholder for vision-based classifier

### Persistence

- `src/db/repo.py`
	- SQLite schema and queries:
		- `pipes` table: last known pipe state + timestamps
		- `events` table: event log
		- `settings` table: runtime switch settings (e.g. gate source)

### UI

- `src/ui/dashboard.py`
	- Streamlit dashboard reading `var/pipes.db` and `var/latest.jpg`.

---

## 5) Troubleshooting

### No camera frames / reconnect loop

- Check your `video_source` value in `config/runtime.yaml`.
- On Linux, verify the device exists: `ls -l /dev/video*`.

### ROI wizard window doesn’t open

- You likely installed a headless OpenCV build or are on a headless machine.
- Run the wizard on a machine with a display, generate `config/rois.yaml`, then copy it to the Pi.

### Dashboard shows old counts

- The DB persists across runs: delete `var/pipes.db` to reset.

### Too much logging

- Set `log_level: "INFO"` in `config/runtime.yaml`.

