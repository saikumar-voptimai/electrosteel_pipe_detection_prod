# Jetson Orin Nano Deployment Guide - JetPack 6.2.1

This project runs one YOLO + ByteTrack inference loop from `src/main.py` through
`src/app.py` and `src/vision/tracker.py`.

Target:

- NVIDIA Jetson Orin Nano
- JetPack 6.2.1 / Jetson Linux L4T 36.4.4
- Ubuntu 22.04
- CUDA 12.6.10
- TensorRT 10.3.0
- cuDNN 9.3.0
- Python 3.10
- ARM64 / aarch64

References:

- NVIDIA JetPack 6.2.1 release notes: https://docs.nvidia.com/jetson/jetpack/release-notes/index.html
- NVIDIA PyTorch for Jetson: https://docs.nvidia.com/deeplearning/frameworks/install-pytorch-jetson-platform/index.html
- Torch-TensorRT JetPack notes: https://docs.pytorch.org/TensorRT/getting_started/jetpack.html

## Important Dependency Rule

Do not install `torch` or `torchvision` from PyPI on this target.

The removed `uv.lock` and previous `requirements.txt` pinned:

- `torch==2.9.1`
- `torchvision==0.24.1`

Those are generic PyPI wheels, not Jetson CUDA wheels. They are the likely cause
of:

```bash
AssertionError: Torch not compiled with CUDA enabled
```

Install Jetson-compatible PyTorch first, then install the rest of the project
dependencies.

## System Setup

```bash
sudo apt update
sudo apt install -y \
  python3 \
  python3.10-venv \
  python3-venv \
  python3-pip \
  git \
  libopenblas-dev \
  libjpeg-dev \
  libpng-dev \
  libv4l-dev \
  libgl1 \
  libglib2.0-0 \
  python3-opencv
```

Verify JetPack and CUDA:

```bash
apt show nvidia-jetpack
nvcc --version
python3 --version
uname -m
```

Expected:

- `nvcc` reports CUDA 12.6.
- `python3 --version` reports Python 3.10.x on stock JetPack 6.2.1.
- `uname -m` reports `aarch64`.

If `python3 --version` reports Python 3.11, do not use `python3 -m venv` for
this project. The Jetson CUDA PyTorch wheel used below must be installed into a
Python 3.10 environment.

## Python Environment

From the repository root:

```bash
python3.10 -m venv --system-site-packages .venv
source .venv/bin/activate
python --version
python -m pip install --upgrade pip setuptools wheel
```

`--system-site-packages` lets the venv see JetPack-provided Python bindings such
as TensorRT and system OpenCV when they are installed through apt.

Install Jetson CUDA PyTorch and TorchVision:

```bash
python -m pip install \
  torch==2.8.0 \
  torchvision==0.23.0 \
  --index-url https://pypi.jetson-ai-lab.io/jp6/cu126
```

Do not continue if this command prints `No matching distribution found`. That
means the active venv is not using Python 3.10 or the Jetson wheel index is not
reachable.

Validate before installing the rest of the dependencies:

```bash
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
python -c "import torch; print(torch.cuda.get_device_name(0))"
```

Then install project dependencies without replacing torch:

```bash
python -m pip install -r requirements.txt
```

If pip attempts to replace `torch` or `torchvision`, stop and inspect the command.
The project requirements intentionally do not list them.

## Optional Basler Camera Dependency

Basler camera support uses the official `pypylon` Python binding. Install the
Basler pylon runtime/SDK on the Jetson first, then install:

```bash
python -m pip install pypylon
```

This dependency is optional. It is imported only when a caster camera config sets
`camera.type: basler`, so VA Imaging and video-file runs do not require pypylon.

## Repair A Failed Python 3.11 Venv

If you already created `.venv` with Python 3.11 and saw:

```text
ERROR: No matching distribution found for torch==2.8.0
AssertionError: Torch not compiled with CUDA enabled
```

recreate the environment with Python 3.10:

```bash
deactivate 2>/dev/null || true
mv .venv .venv-py311-cpu-torch
python3.10 -m venv --system-site-packages .venv
source .venv/bin/activate
python --version
python -m pip install --upgrade pip setuptools wheel
python -m pip install torch==2.8.0 torchvision==0.23.0 --index-url https://pypi.jetson-ai-lab.io/jp6/cu126
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
python -c "import torch; print(torch.cuda.get_device_name(0))"
python -m pip install -r requirements.txt
```

Only remove `.venv-py311-cpu-torch` after the new environment validates CUDA.

## CUDA Validation

```bash
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
python -c "import torch; print(torch.cuda.get_device_name(0))"
python -c "import torch; x=torch.zeros((1,), device='cuda'); print(x.device)"
```

Expected:

- `torch.cuda.is_available()` prints `True`.
- Device name is an NVIDIA Jetson Orin GPU.
- The tensor device prints `cuda:0`.

## Project GPU Validation

Use a `.pt` model for PyTorch CUDA inference:

```yaml
model_path: models/yolo/yolo11n_26_16_01.pt
device: auto
half: true
```

Run:

```bash
source .venv/bin/activate
python src/app.py --caster 1
```

Check logs:

```bash
tail -f var/pipe_detect.log
```

Expected log line:

```text
YOLO runtime resolved | requested=auto | ultralytics_device=0 | torch_device=cuda:0 | half=True
```

If the runtime config points to an `.onnx` file, PyTorch CUDA is not enough by
itself. ONNX GPU execution requires a CUDA/TensorRT-capable ONNX Runtime provider.
For production on Jetson, export the YOLO model to TensorRT `.engine` and set:

```yaml
model_path: models/yolo/<model>.engine
device: auto
half: false
```

## TensorRT Export Recommendation

On the Jetson, after the PyTorch CUDA environment is validated:

```bash
yolo export \
  model=models/yolo/yolo11n_26_16_01.pt \
  format=engine \
  imgsz=640 \
  half=True \
  device=0
```

Then update the caster runtime file, for example
`config/casters/caster_1/runtime.yaml`, to the generated `.engine` path.

## Runtime Commands

Main app:

```bash
source .venv/bin/activate
python src/app.py --caster 1
```

ROI redraw:

```bash
source .venv/bin/activate
python src/app.py --caster 1 --redraw
```

Dashboard:

```bash
source .venv/bin/activate
PIPE_DASHBOARD_CASTER=caster_1 streamlit run src/ui/dashboard.py
```

## Performance Notes For This Codebase

- `src/app.py` resizes every frame on CPU before inference. Keep `imgsz` at 640
  or 960; larger input widths increase CPU resize and GPU inference time.
- `src/vision/tracker.py` now uses `torch.inference_mode()`.
- `half: true` is applied only for CUDA `.pt` inference. TensorRT precision is
  controlled when the `.engine` is built.
- `torch.backends.cudnn.benchmark` is enabled when CUDA is active.
- The app converts detection tensors to CPU NumPy arrays after inference because
  the FSM, DB writes, and overlay code are CPU/OpenCV based. Keep that transfer
  to the current single post-inference boundary.
- OpenCV display and JPEG publishing are CPU work. Use `run_headless: true` and
  lower `publish_fps` if inference FPS is being limited by rendering.
