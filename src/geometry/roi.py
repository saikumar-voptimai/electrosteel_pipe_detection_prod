# polygon contains(), box IoU, centroid 
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Tuple, TYPE_CHECKING
import numpy as np
import cv2

if TYPE_CHECKING:
  from vision.types import BBox

Point = Tuple[int, int]

@dataclass(frozen=True)
class PolygonROI:
  """
  Polygon ROI defined by a list of (x,y) points.
  """
  #TODO: Want to include % of intersection between two polygons or a polygon and a bbox
  name: str
  points: List[Point]  # in (x,y) format

  def contains(self, x: float, y: float) -> bool:
    """
    Check if the point (x,y) is inside the polygon using cv2.pointPolygonTest.
    """
    pts = np.array(self.points, dtype=np.int32)
    #TODO: verify why float was used here
    return cv2.pointPolygonTest(pts, (float(x), float(y)), False) >= 0

  def intersects_bbox(self, bbox: "BBox") -> bool:
    """
    Check whether this polygon touches or overlaps a detection bounding box.
    """
    x1, x2 = sorted((float(bbox.x1), float(bbox.x2)))
    y1, y2 = sorted((float(bbox.y1), float(bbox.y2)))
    if x1 == x2 or y1 == y2:
      return False

    rx1, ry1, rx2, ry2 = self.bbox()
    if x2 < rx1 or x1 > rx2 or y2 < ry1 or y1 > ry2:
      return False

    rect_points = [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]
    if any(self.contains(x, y) for x, y in rect_points):
      return True

    if any(x1 <= x <= x2 and y1 <= y <= y2 for x, y in self.points):
      return True

    rect_edges = list(zip(rect_points, rect_points[1:] + rect_points[:1]))
    poly_points = [(float(x), float(y)) for x, y in self.points]
    poly_edges = list(zip(poly_points, poly_points[1:] + poly_points[:1]))
    return any(
      self._segments_intersect(a1, a2, b1, b2)
      for a1, a2 in poly_edges
      for b1, b2 in rect_edges
    )

  @staticmethod
  def _segments_intersect(
    a1: Tuple[float, float],
    a2: Tuple[float, float],
    b1: Tuple[float, float],
    b2: Tuple[float, float],
  ) -> bool:
    def orient(p: Tuple[float, float], q: Tuple[float, float], r: Tuple[float, float]) -> float:
      return (q[0] - p[0]) * (r[1] - p[1]) - (q[1] - p[1]) * (r[0] - p[0])

    def on_segment(p: Tuple[float, float], q: Tuple[float, float], r: Tuple[float, float]) -> bool:
      eps = 1e-9
      return (
        min(p[0], r[0]) - eps <= q[0] <= max(p[0], r[0]) + eps
        and min(p[1], r[1]) - eps <= q[1] <= max(p[1], r[1]) + eps
      )

    o1 = orient(a1, a2, b1)
    o2 = orient(a1, a2, b2)
    o3 = orient(b1, b2, a1)
    o4 = orient(b1, b2, a2)
    eps = 1e-9

    if abs(o1) <= eps and on_segment(a1, b1, a2):
      return True
    if abs(o2) <= eps and on_segment(a1, b2, a2):
      return True
    if abs(o3) <= eps and on_segment(b1, a1, b2):
      return True
    if abs(o4) <= eps and on_segment(b1, a2, b2):
      return True

    return (o1 > 0) != (o2 > 0) and (o3 > 0) != (o4 > 0)
  
  def area(self) -> float:
    """
    Compute the area of the polygon using cv2.contourArea.
    """
    pts = np.array(self.points, dtype=np.int32)
    return float(cv2.contourArea(pts))
  
  def bbox(self) -> Tuple[int, int, int, int]:
    """
    Compute the bounding box of the polygon as (x1, y1, x2, y2).
    """
    xs = [p[0] for p in self.points]
    ys = [p[1] for p in self.points]
    return (min(xs), min(ys), max(xs), max(ys))
  
  def centroid(self) -> Tuple[float, float]:
    """
    Compute the centroid of the polygon.
    """
    pts = np.array(self.points, dtype=np.int32)
    M = cv2.moments(pts)
    if M["m00"] == 0:
      return (0.0, 0.0)
    cx = M["m10"] / M["m00"]
    cy = M["m01"] / M["m00"]
    return (cx, cy)
  

@dataclass
class ROIManager:
  rois: Dict[str, List[Point]]

  def __post_init__(self) -> None:
    self._objs = {k: PolygonROI(k, v) for k, v in self.rois.items()}
  
  def contains(self, name: str, x: float, y: float) -> bool:
    """
    Check if point (x,y) is inside the named ROI polygon.
    """
    return self._objs[name].contains(x, y)

  def roi(self, name:str) -> PolygonROI:
    """
    Get the PolygonROI object by name.
    """
    return self._objs[name]