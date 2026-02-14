from __future__ import annotations
import os
import time
from dataclasses import dataclass
from typing import Dict, List, Tuple
from pathlib import Path
import logging
from utils.config import HistoryCfg 
import cv2
import numpy as np

from geometry.roi import ROIManager, PolygonROI
from vision.types import TrackDet
from utils.roi_names import RoiName


logger = logging.getLogger(__name__)
from datetime import datetime, time as dtime
import pytz

IST = pytz.timezone("Asia/Kolkata")


def _parse_hhmm(v: str) -> dtime:
    h, m = v.split(":")
    return dtime(int(h), int(m))

def _resolve_shift(ts: datetime, shifts: list[dict]) -> str:
    if not shifts:
        return "shift_unknown"

    t = ts.timetz().replace(tzinfo=None)

    for s in shifts:
        name = str(s.get("name", "shift"))
        start = _parse_hhmm(s.get("start", "00:00"))
        end = _parse_hhmm(s.get("end", "23:59"))

        if start < end:
            if start <= t < end:
                return name
        else:
            # Overnight shift (e.g. 22:00 → 06:00)
            if t >= start or t < end:
                return name

    return str(shifts[0].get("name", "shift"))

def draw_text_bottom_right(
    img: np.ndarray,
    text: str,
    font=cv2.FONT_HERSHEY_SIMPLEX,
    font_scale: float = 1.5,
    thickness: int = 2,
    margin: int = 20,
    color: Tuple[int, int, int] = (255, 255, 255),
) -> None:
    """
    Draw text anchored to the bottom-right corner, safely inside the frame.
    """
    h, w = img.shape[:2]

    (text_w, text_h), baseline = cv2.getTextSize(
        text, font, font_scale, thickness
    )

    x = max(margin, w - text_w - margin)
    y = max(text_h + margin, h - margin)

    cv2.putText(
        img,
        text,
        (x, y),
        font,
        font_scale,
        color,
        thickness,
        cv2.LINE_AA,
    )



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
  for roi in (
    RoiName.LOADCELL,
    RoiName.CASTER_ORIGIN,
    RoiName.GATE1_OPEN,
    RoiName.GATE2_OPEN,
    RoiName.RIGHT_ORIGIN,
    ):
    name = roi.value

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
  
  status_text = f"{runfps:.2f} FPS | {ist_now_str(ts)}"
  draw_text_bottom_right(
        out,
        status_text,
        font_scale=1.5,
        thickness=2,
        margin=20,
    )
  loadcell = rois_scaled.get(RoiName.LOADCELL.value)
  left_origin = rois_scaled.get(RoiName.LEFT_ORIGIN.value)
  right_origin = rois_scaled.get(RoiName.RIGHT_ORIGIN.value)

  # Draw Detections/Tracks
  for d in dets_vis:
    x1, y1, x2, y2 = map(int, [d.bbox.x1, d.bbox.y1, d.bbox.x2, d.bbox.y2])
    color = (255, 0, 0)
    # Change pipe bbox color if in loadcell ROI
    if d.cls_name == "pipe":
      cx, cy = d.bbox.centroid()
      if loadcell is not None and loadcell.contains(cx, cy):
        color = (0, 0, 255) # Red if in loadcell ROI
      else:
        color = (0, 255, 0) # Green for the pipe
    # Stop rendering pipe bbox if it is in roi_left_origin or roi_right_origin
    if d.cls_name == "pipe":
      cx, cy = d.bbox.centroid()
      if left_origin and left_origin.contains(cx, cy):
        continue
      if right_origin and right_origin.contains(cx, cy):
        continue
    
    # if not debug:
    #   logger.debug("Skipping detailed bbox drawing since debug=False | det=%s", d)
    #   continue
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
    out_path: str
    fps: int
    history_cfg: HistoryCfg  | None
    _last: float = 0.0

    def publish(self, frame_bgr: np.ndarray) -> None:
        if (
            self.fps <= 0
            or frame_bgr is None
            or not isinstance(frame_bgr, np.ndarray)
            or frame_bgr.size == 0
        ):
            return

        now = time.time()
        if now - self._last < 1.0 / float(self.fps):
            return
        self._last = now

        project_root = Path(__file__).parent.parent.parent
        latest = project_root / self.out_path
        latest.parent.mkdir(parents=True, exist_ok=True)

        ok, enc = cv2.imencode(".jpg", frame_bgr)
        if not ok:
            logger.warning("Frame encode failed")
            return
        data = enc.tobytes()

        # ---- latest.jpg (UI, best effort) ----
        try:
            tmp = latest.with_name(f"{latest.stem}.{os.getpid()}.tmp")
            tmp.write_bytes(data)
            os.replace(tmp, latest)
        except Exception:
            logger.debug("latest.jpg locked, skipping")
        cfg = self.history_cfg
        # ---- history save (absolute OR relative path) ----
        if not cfg or not cfg.enabled:
            return

        try:
            tz = pytz.timezone(cfg.timezone)
            ts = datetime.fromtimestamp(now, tz)

            base_dir = Path(cfg.base_dir)
            if not base_dir.is_absolute():
                base_dir = project_root / base_dir

            shift = _resolve_shift(ts, cfg.shifts or [])
            date_fmt = cfg.date_folder_format

            day_dir = (
                base_dir
                / ts.strftime(date_fmt)
                / shift
            )
            day_dir.mkdir(parents=True, exist_ok=True)
            time_fmt = cfg.time_filename_format
            ts_str = ts.strftime(time_fmt)
            if "%f" in time_fmt:
                ts_str = ts_str[:-3]
            
            fname = (
                f"{cfg.prefix}_"
                f"{ts_str}."
                f"{cfg.ext}"
            )

            (day_dir / fname).write_bytes(data)

        except Exception:
            logger.exception("History image save failed (ignored)")



