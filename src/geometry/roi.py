# polygon contains(), box IoU, centroid 
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Tuple
import numpy as np
import cv2

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