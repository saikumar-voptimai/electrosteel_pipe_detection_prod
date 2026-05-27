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
    self._torch_cuda_available = self._check_torch_cuda()
    self._resolved_device = self._resolve_device()

    logger.info(
      "YOLO runtime requested | device=%s | resolved_device=%s | half=%s | torch_cuda=%s | tensorrt_model=%s",
      self.device,
      self._resolved_device,
      self.half,
      self._torch_cuda_available,
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

    self.model = YOLO(self.model_path)
    try:
      logger.info("YOLO model loaded | names=%d", len(getattr(self.model, "names", {}) or {}))
    except Exception:
      logger.debug("YOLO model loaded (names unavailable)")

  def _check_torch_cuda(self) -> bool:
    try:
      import torch
      return bool(torch.cuda.is_available())
    except Exception:
      return False

  def _resolve_device(self) -> int | str | None:
    requested = self.device
    if requested is None:
      return None
    if isinstance(requested, str):
      normalized = requested.strip().lower()
      if normalized in ("", "none", "default"):
        return None
      if normalized == "auto":
        if self._is_tensorrt:
          return 0
        return 0 if self._torch_cuda_available else "cpu"
      return requested
    return requested

  @staticmethod
  def _device_is_cpu(device: int | str | None) -> bool:
    if device is None:
      return False
    return isinstance(device, str) and device.strip().lower() == "cpu"

  def infer(self, frame: np.ndarray) -> List[TrackDet]:
    """
    Run tracking inference on a single frame.
    Returns list of TrackDet.
    """
    t0 = time.perf_counter()
    track_kwargs = {
      "source": frame,
      "persist": True,
      "tracker": self.tracker_yaml,
      "conf": self.conf,
      "iou": self.iou,
      "imgsz": self.imgsz,
      "verbose": False,
    }
    if self._resolved_device is not None:
      track_kwargs["device"] = self._resolved_device
    if self.half and not self._is_tensorrt and not self._device_is_cpu(self._resolved_device):
      track_kwargs["half"] = True

    results = self.model.track(**track_kwargs)
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
