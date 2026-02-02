from __future__ import annotations
import os
import time
from dataclasses import dataclass
from typing import Dict, List, Tuple
from pathlib import Path
import logging

import cv2
import numpy as np

from geometry.roi import ROIManager, PolygonROI
from vision.types import TrackDet

logger = logging.getLogger(__name__)

from datetime import datetime
import pytz

IST = pytz.timezone("Asia/Kolkata")

def ist_now_str(ts: float) -> str:
    return datetime.fromtimestamp(ts, tz=IST).strftime("%Y-%m-%d %H:%M:%S")

def scale_polygon(points, sx, sy):
    return [(int(x * sx), int(y * sy)) for (x, y) in points]

def draw_overlay(frame_vis: np.ndarray, 
                 rois: ROIManager, 
                 dets_vis: List[TrackDet], 
                 ts: float,
                 scale_x: float = 1.0,
                 scale_y: float = 1.0,
                 gate_metrics: Dict = None,
                 debug: bool = False,
                 runfps: float = 0.0) -> np.ndarray:
  """
  Draw ROIs and tracking boxes on the frame. 
  The frame is resized frame_viz using publish_imgsz.
  dets are also dets_vis hence, we take the scaling factors to map ROIs correctly.
  """
  out = frame_vis.copy()

  # Precompute scaled ROIs for checks in visualization coordinates.
  rois_scaled: Dict[str, PolygonROI] = {k: PolygonROI(k, scale_polygon(v, scale_x, scale_y)) for k, v in rois.rois.items()}

  # Draw key ROIs - Only for testing/debugging
  #TODO: Use Enums or constants for ROI names
  for name in [
    "roi_loadcell",
    "roi_caster_origin",
    "roi_gate1_open",
    "roi_gate2_open",
    "roi_right_origin",
  ]:
    if name not in rois.rois:
      continue
    if not debug:
      logger.debug("Skipping ROI drawing since debug=False | roi=%s", name)
      continue
    pts_orig = rois.rois[name]
    pts_scaled = scale_polygon(pts_orig, scale_x, scale_y)
    roi_polygon_scaled = rois_scaled[name]
    
    pts_np = np.array(pts_scaled, dtype=np.int32)
    cv2.polylines(out, [pts_np], True, (0, 255, 255), 2)                    # ROI in yellow
    (cx, cy) = roi_polygon_scaled.centroid()                            
    cv2.circle(out, (int(cx), int(cy)), radius=5, color=(0, 255, 255), thickness=-1) # Centroid in yellow
    cv2.putText(out, name, 
                (int(pts_np[0][0])+20, int(pts_np[0][1])+5),                # ROI name
                cv2.FONT_HERSHEY_SIMPLEX, 
                1, (0,255,255), 2)
  
  cv2.putText(
    out,
    f"{runfps:.2f} FPS | {ist_now_str(ts)}",
    (int(0.5 * out.shape[1]), int(0.9 * out.shape[0])),                    # Timestamp at btm-right
    cv2.FONT_HERSHEY_SIMPLEX,
    1.5,
    (255, 255, 255),
    2,
  )

  # Draw Detections/Tracks
  for d in dets_vis:
    x1, y1, x2, y2 = map(int, [d.bbox.x1, d.bbox.y1, d.bbox.x2, d.bbox.y2])
    color = (255, 0, 0)
    # Change pipe bbox color if in loadcell ROI
    if d.cls_name == "pipe":
      cx, cy = d.bbox.centroid()
      if rois_scaled["roi_loadcell"] is not None and rois_scaled["roi_loadcell"].contains(cx, cy):
        color = (0, 0, 255) # Red if in loadcell ROI
      else:
        color = (0, 255, 0) # Green for the pipe
    # Stop rendering pipe bbox if it is in roi_left_origin or roi_right_origin
    if d.cls_name == "pipe":
      cx, cy = d.bbox.centroid()
      if rois_scaled["roi_left_origin"] is not None and rois_scaled["roi_left_origin"].contains(cx, cy):
        continue
      if rois_scaled["roi_right_origin"] is not None and rois_scaled["roi_right_origin"].contains(cx, cy):
        continue
    
    if not debug:
      logger.debug("Skipping detailed bbox drawing since debug=False | det=%s", d)
      continue
    cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
    tid = d.track_id if d.track_id is not None else -1
    cv2.putText(out, 
                f"{d.cls_name}:{tid} {d.conf:.2f}", 
                (x1, max(20, y1-5)),
                cv2.FONT_HERSHEY_SIMPLEX, 
                1.5, 
                color, 
                2)
    cx, cy = d.bbox.centroid()
    cv2.circle(out, (int(cx), int(cy)), radius=5, color=color, thickness=-1)

    if d.cls_name in ("gate1", "gate2") and gate_metrics is not None and not debug:
      m = gate_metrics.get(d.cls_name) if isinstance(gate_metrics, dict) else None
      if isinstance(m, dict) and m:
        metrics_str = ", ".join([f"{k}:{float(v):.2f}" for k, v in m.items()])
        cv2.putText(
          out,
          f"Metrics: {metrics_str}",
          (x1, min(out.shape[0]-10, y2+25)),
          cv2.FONT_HERSHEY_SIMPLEX,
          0.5,
          color,
          2,
        )
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
      logger.warning("Failed to replace latest frame | tmp=%s -> out=%s", tmp_path, out_full_path)