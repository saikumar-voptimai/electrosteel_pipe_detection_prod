# ByteTrack wrapper (per-class tracking policies)
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional
import importlib.util
import logging
import time
from pathlib import Path
import numpy as np
from ultralytics import YOLO

from utils.device import RuntimeDevice, configure_torch_backend, select_runtime_device
from vision.types import BBox, TrackDet

logger = logging.getLogger(__name__)

@dataclass
class YoloByteTrack:
  """
  YOLOv11 ByteTrack wrapper for multi-class tracking.
  """
  model_path: str
  tracker_yaml: str
  conf: float
  iou: float
  imgsz: int
  device: int | str | None = "auto"
  half: bool = False

  def __post_init__(self) -> None:
    logger.info("Loading YOLO model: %s", self.model_path)
    self._model_suffix = Path(str(self.model_path)).suffix.lower()
    self._is_tensorrt = self._model_suffix == ".engine"
    self._runtime_device: RuntimeDevice = select_runtime_device(
      requested=self.device,
      model_path=self.model_path,
      half=self.half,
    )
    self._resolved_device = self._runtime_device.ultralytics_device
    configure_torch_backend(self._runtime_device)

    logger.info(
      "YOLO runtime resolved | requested=%s | ultralytics_device=%s | torch_device=%s | half=%s | torch=%s | torch_cuda=%s | cuda_name=%s | tensorrt_model=%s",
      self._runtime_device.requested,
      self._resolved_device,
      self._runtime_device.torch_device,
      self._runtime_device.half,
      self._runtime_device.torch_version,
      self._runtime_device.torch_cuda_available,
      self._runtime_device.cuda_device_name,
      self._is_tensorrt,
    )
    if self._is_tensorrt and importlib.util.find_spec("tensorrt") is None:
      logger.warning(
        "TensorRT model configured but Python TensorRT bindings are not importable in this environment. "
        "On Jetson, run the app from a JetPack-compatible Python environment."
      )
    if not self._is_tensorrt and self._device_is_cpu(self._resolved_device):
      logger.warning(
        "YOLO is configured for CPU inference. On Jetson Orin Nano this is usually the reason FPS stays near 2-3. "
        "Install Jetson CUDA PyTorch or use a TensorRT .engine model."
      )
    if self._runtime_device.is_onnx:
      logger.warning(
        "ONNX model configured. GPU execution depends on a CUDA/TensorRT-capable ONNX Runtime provider; "
        "for Jetson production, prefer exporting this model to TensorRT .engine."
      )

    self.model = YOLO(self.model_path)
    if self._model_suffix == ".pt" and self._runtime_device.is_cuda:
      self.model.to(self._runtime_device.torch_device)
    try:
      logger.info("YOLO model loaded | names=%d", len(getattr(self.model, "names", {}) or {}))
    except Exception:
      logger.debug("YOLO model loaded (names unavailable)")

  @staticmethod
  def _device_is_cpu(device: int | str | None) -> bool:
    if device is None:
      return False
    return isinstance(device, str) and device.strip().lower() == "cpu"

  @staticmethod
  def _is_cuda_runtime_error(exc: RuntimeError) -> bool:
    message = str(exc).lower()
    markers = (
      "cuda",
      "cudacachingallocator",
      "nvml",
      "nvmap",
      "out of memory",
      "cublas",
      "cudnn",
    )
    return any(marker in message for marker in markers)

  def _track_once(self, track_kwargs: dict) -> list:
    try:
      import torch
    except Exception:
      torch = None

    if torch is not None:
      with torch.inference_mode():
        return self.model.track(**track_kwargs)
    return self.model.track(**track_kwargs)

  def _track_kwargs(self, frame: np.ndarray) -> dict:
    track_kwargs = {
      "source": frame,
      "persist": True,
      "tracker": self.tracker_yaml,
      "conf": self.conf,
      "iou": self.iou,
      "imgsz": self.imgsz,
      "verbose": False,
    }
    if self._resolved_device is not None and not self._is_tensorrt:
      track_kwargs["device"] = self._resolved_device
    if self._runtime_device.half:
      track_kwargs["half"] = True
    return track_kwargs

  def _fallback_to_cpu(self) -> None:
    logger.warning(
      "CUDA inference failed for PyTorch model; reloading YOLO on CPU for backup mode | model=%s",
      self.model_path,
    )
    try:
      del self.model
    except AttributeError:
      pass

    try:
      import torch

      if torch.cuda.is_available():
        torch.cuda.empty_cache()
    except Exception:
      logger.debug("Could not clear CUDA cache before CPU fallback", exc_info=True)

    self._runtime_device = select_runtime_device(
      requested="cpu",
      model_path=self.model_path,
      half=False,
    )
    self._resolved_device = self._runtime_device.ultralytics_device
    self.model = YOLO(self.model_path)
    logger.warning(
      "YOLO runtime switched to CPU fallback | ultralytics_device=%s | torch_device=%s | half=%s",
      self._resolved_device,
      self._runtime_device.torch_device,
      self._runtime_device.half,
    )

  def infer(self, frame: np.ndarray) -> List[TrackDet]:
    """
    Run tracking inference on a single frame.
    Returns list of TrackDet.
    """
    t0 = time.perf_counter()
    track_kwargs = self._track_kwargs(frame)

    try:
      results = self._track_once(track_kwargs)
    except RuntimeError as exc:
      can_retry_cpu = (
        not self._is_tensorrt
        and self._runtime_device.is_cuda
        and self._is_cuda_runtime_error(exc)
      )
      if not can_retry_cpu:
        raise
      self._fallback_to_cpu()
      results = self._track_once(self._track_kwargs(frame))

    dt_ms = (time.perf_counter() - t0) * 1000.0
    if not results:
      logger.debug("YOLO.track returned no results | dt_ms=%.1f", dt_ms)
      return []
    
    r0 = results[0]
    boxes = getattr(r0, 'boxes', None)
    if boxes is None or len(boxes) == 0:
      logger.debug("No boxes in result | dt_ms=%.1f", dt_ms)
      return []
    
    # Get boxes, class_ids, confs, ids from the tracking results
    xyxy = boxes.xyxy.cpu().numpy()  # (N,4)
    class_ids = boxes.cls.cpu().numpy().astype(int)  # (N,)
    confs = boxes.conf.cpu().numpy()  # (N,)
    
    ids = None
    if hasattr(boxes, 'id') and boxes.id is not None:
      ids = boxes.id.cpu().numpy().astype(int)  # (N,)

    out: List[TrackDet] = []
    for i in range(len(xyxy)):
      x1, y1, x2, y2 = map(float, xyxy[i])
      tid: Optional[int] = int(ids[i]) if ids is not None else None
      out.append(
        TrackDet(
          cls_name=self.model.names[int(class_ids[i])],
          conf=float(confs[i]),
          track_id=tid,
          bbox=BBox(x1, y1, x2, y2),
        )
      )
    logger.debug("Tracking inference | dt_ms=%.1f | dets=%d", dt_ms, len(out))
    return out
