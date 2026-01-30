from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Optional
import math
import numpy as np
import logging

from geometry.roi import ROIManager
from plc.client import PLCClient
from vision.types import TrackDet, BBox

logger = logging.getLogger(__name__)


def _iou(a: BBox, b: BBox) -> float:
  """
  Compute Intersection over Union of two bounding boxes.
  """
  ix1 = max(a.x1, b.x1)
  iy1 = max(a.y1, b.y1)
  ix2 = min(a.x2, b.x2)
  iy2 = min(a.y2, b.y2)
  iw = max(0.0, ix2 - ix1)
  ih = max(0.0, iy2 - iy1)
  inter = iw * ih
  if inter <= 0:
    return 0.0
  union = a.area + b.area - inter
  return inter / union if union > 0 else 0.0


class GateStatusSource(ABC):
  @abstractmethod
  def get_position(self, gate_name: str, frame: np.ndarray, dets: List[TrackDet]) -> str:
    """Get the position of the gate ("open" | "closed" | "unknown")"""
    ...


class PLCGateSource(GateStatusSource):
  """
  Gate status source using PLC client
  """ 
  plc: PLCClient

  # PLC tags expected to return open/closed state
  # e.g. {"gate1": "gate1_is_open", "gate2": "gate2_is_open"}

  open_tags: Dict[str, str]

  def get_position(self, gate_name, frame=None, dets=None) -> tuple[str, Dict[str, float]]:
    """
    Get gate position from PLC
    """
    tag = self.open_tags.get(gate_name)
    metrics: Dict[str, float] = {}
    if not tag:
      logger.debug("PLC gate source missing tag | gate=%s", gate_name)
      return "unknown", metrics
    try:
      v = self.plc.read_bool(tag)
      logger.debug("PLC gate read | gate=%s | tag=%s | value=%s", gate_name, tag, v)
      return "open" if v else "closed", metrics
    except Exception:
      logger.exception("PLC gate read failed | gate=%s | tag=%s", gate_name, tag)
      return "unknown", metrics


@dataclass
class GeometryGateSource(GateStatusSource):
  """
  Gate status source using geometry model analysis.
  """
  rois: ROIManager
  min_gate_conf: float
  max_area_ratio_vs_closed: float
  max_w_over_h: float
  human_iou_occlusion: float

  def get_position(self, gate_name: str, frame=None, dets: List[TrackDet] | None = None) -> tuple[str, dict[str, float]]:
    """
    Get gate position from geometry analysis of detections.
    Returns "open", "closed", or "unknown".
    """
    metrics: Dict[str, float] = {}
    if dets is None:
      return "unknown", metrics
    
    gate_det = self._best_det(dets, gate_name, self.min_gate_conf)

    if gate_det is None:
      logger.debug("No gate detection | gate=%s | min_conf=%.3f", gate_name, self.min_gate_conf)
      return "unknown", metrics
    
    open_roi = f"roi_{gate_name}_open"
    closed_roi = f"roi_{gate_name}_closed"

    if open_roi not in self.rois.rois or closed_roi not in self.rois.rois:
      logger.debug("Missing gate ROIs | gate=%s | open_roi=%s | closed_roi=%s", gate_name, open_roi, closed_roi)
      return "unknown", metrics
    
    gate_bbox = gate_det.bbox
    cx, cy = gate_bbox.centroid()

    # Human occlusion guard: Human in safety ROI OR overlaps gate bbox or gate closed roi
    #TODO: If human overlaps defined roi_gate1_closed or roi_gate2_closed -> closed
    for d in dets:
      if d.cls_name == "human":
        logger.info(f"Human detected | gate={gate_name} | conf={d.conf:.3f}")
      if d.cls_name.lower() not in ("human", "humans"):
        continue
      hb = d.bbox
      hx, hy = hb.centroid()
      #TODO: Use ENUMS or constants for ROI names
      if _iou(hb, gate_bbox) >= self.human_iou_occlusion:
        logger.debug("Gate occluded by human (iou) | gate=%s | iou=%.3f", gate_name, _iou(hb, gate_bbox))
        return "unknown", metrics
      if self.rois.contains(closed_roi, hx, hy):
        logger.debug("Gate occluded by human in closed ROI | gate=%s", gate_name)
        return "unknown", metrics 
      if self.rois.contains("roi_safety_critical", hx, hy) and _iou(hb, gate_bbox) >= self.human_iou_occlusion:
        logger.debug("Gate occluded by human in safety ROI | gate=%s", gate_name)
        return "unknown", metrics
      

    in_open = self.rois.contains(open_roi, cx, cy)

    closed_area = max(1.0, self.rois.roi(closed_roi).area())
    area_ratio = gate_bbox.area / closed_area
    w_over_h = gate_bbox.w / max(1.0, gate_bbox.h)
    
    metrics = {}
    metrics["closed_area"] = closed_area
    metrics["area_ratio"] = area_ratio
    metrics["w_over_h"] = w_over_h

    # Open if in open ROI AND gate looks tall/narrow AND smaller area vs closed baseline
    if in_open and (w_over_h < self.max_w_over_h) and (area_ratio < self.max_area_ratio_vs_closed):
      logger.debug(
        "Gate=open by geometry | gate=%s | conf=%.3f | in_open=%s | w/h=%.3f | area_ratio=%.3f",
        gate_name,
        gate_det.conf,
        in_open,
        w_over_h,
        area_ratio,
      )
      return "open", metrics

    logger.debug(
      "Gate=closed by geometry | gate=%s | conf=%.3f | in_open=%s | w/h=%.3f | area_ratio=%.3f",
      gate_name,
      gate_det.conf,
      in_open,
      w_over_h,
      area_ratio,
    )
    
    return "closed", metrics
  
  @staticmethod
  def _best_det(dets: List[TrackDet], cls_name: str, min_conf: float) -> Optional[TrackDet]:
    """
    Get the best detection for a given class name above min confidence.
    """
    c = [d for d in dets if d.cls_name.lower() == cls_name.lower() and d.conf >= min_conf]
    if not c:
      return None
    return max(c, key=lambda d: d.conf)
  

@dataclass
class VisionGateSource(GateStatusSource):
  def get_position(self, gate_name: str, frame: np.ndarray = None, dets=None) -> str:
    """
    Placeholder for a small classifier (ONNX/TFLITE) later.
    """
    return "unknown", {}