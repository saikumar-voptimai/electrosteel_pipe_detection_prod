from __future__ import annotations
from dataclasses import dataclass

from typing import Optional, Tuple

@dataclass(frozen=True)
class BBox:
  x1: float; y1: float; x2: float; y2: float
  
  def centroid(self) -> Tuple[float, float]:
    return ((self.x1 + self.x2)/2.0, (self.y1 + self.y2)/2.0)

  def intersection_area(self, other: "BBox") -> float:
    ix1 = max(self.x1, other.x1)
    iy1 = max(self.y1, other.y1)
    ix2 = min(self.x2, other.x2)
    iy2 = min(self.y2, other.y2)
    return max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)

  def intersects(self, other: "BBox") -> bool:
    return self.intersection_area(other) > 0.0
  
  @property
  def w(self) -> float: return max(0.0, self.x2 - self.x1)

  @property
  def h(self) -> float: return max(0.0, self.y2 - self.y1)
  
  @property
  def area(self) -> float: return self.w * self.h

@dataclass(frozen=True)
class TrackDet:
  """
  Tracker detection info
  """
  cls_name: str
  conf: float
  track_id: Optional[int]
  bbox: BBox