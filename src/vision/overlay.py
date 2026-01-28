from __future__ import annotations
import os
import time
from dataclasses import dataclass
from typing import List, Tuple
from pathlib import Path
import logging

import cv2
import numpy as np

from geometry.roi import ROIManager
from vision.types import TrackDet

logger = logging.getLogger(__name__)

def draw_overlay(frame: np.ndarray, rois: ROIManager, dets: List[TrackDet]) -> np.ndarray:
  """
  Draw ROIs and tracking boxes on the frame.
  """
  out = frame.copy()

  # Draw key ROIs - Only for testing/debugging
  #TODO: Use Enums or constants for ROI names
  for name in [
    "roi_loadcell",
    "roi_caster5_origin",
  ]:
    if name not in rois.rois:
      continue
    pts = np.array(rois.rois[name], dtype=np.int32)
    cv2.polylines(out, [pts], True, (0, 255, 255), 2)
    cv2.putText(out, name, (int(pts[0][0]), int(pts[0][1])), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.6, (0,255,255), 2)
  
  # Draw Detections/Tracks
  for d in dets:
    x1, y1, x2, y2 = map(int, [d.bbox.x1, d.bbox.y1, d.bbox.x2, d.bbox.y2])
    color = (0, 255, 0) if d.cls_name == "pipe" else (255, 0, 0)
    cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
    tid = d.track_id if d.track_id is not None else -1
    cv2.putText(out, 
                f"{d.cls_name}:{tid} {d.conf:.2f}", 
                (x1, max(20, y1-5)),
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.5, 
                color, 
                2)
  return out


@dataclass
class LatestFramePublisher:
  """
  Overlay latest tracking results onto frames.
  """
  out_path: str
  fps: int = 5
  _last: float = 0.0

  def publish(self, frame_bgr: np.ndarray) -> None:
    """
    Save the latest frame to out_path at limited fps.
    """
    if self.fps <= 0:
      return

    if frame_bgr is None or not isinstance(frame_bgr, np.ndarray) or frame_bgr.size == 0:
      logger.debug("Skipping publish: empty frame")
      return

    now = time.time()
    if now - self._last < 1.0 / float(self.fps):
      return
    self._last = now

    #
    PROJECT_ROOT = Path(__file__).parent.parent.parent
    out_full_path = PROJECT_ROOT / self.out_path
    os.makedirs(out_full_path.parent, exist_ok=True)
    tmp_path = out_full_path.with_name(out_full_path.stem + "_new.jpg")

    ok = cv2.imwrite(str(tmp_path), frame_bgr)
    if not ok:
      logger.warning("Failed to write latest frame | tmp=%s", tmp_path)
      return

    # Atomic replace
    try:
      os.replace(str(tmp_path), str(out_full_path))
      logger.debug("Published latest frame | path=%s", out_full_path)
    except Exception:
      logger.exception("Failed to replace latest frame | tmp=%s -> out=%s", tmp_path, out_full_path)