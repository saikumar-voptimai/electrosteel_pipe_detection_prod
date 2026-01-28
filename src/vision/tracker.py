# ByteTrack wrapper (per-class tracking policies)
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional
import logging
import time
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

  def __post_init__(self) -> None:
    logger.info("Loading YOLO model: %s", self.model_path)
    self.model = YOLO(self.model_path)
    try:
      logger.info("YOLO model loaded | names=%d", len(getattr(self.model, "names", {}) or {}))
    except Exception:
      logger.debug("YOLO model loaded (names unavailable)")

  def infer(self, frame: np.ndarray) -> List[TrackDet]:
    """
    Run tracking inference on a single frame.
    Returns list of TrackDet.
    """
    t0 = time.perf_counter()
    results = self.model.track(
      source=frame,
      persist=True,
      tracker=self.tracker_yaml,
      conf=self.conf,
      iou=self.iou,
      imgsz=self.imgsz,
      verbose=False,
    )
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