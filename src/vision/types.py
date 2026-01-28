from __future__ import annotations
from dataclasses import dataclass

from typing import Optional, Tuple

@dataclass(frozen=True)
class BBox:
  x1: float; y1: float; x2: float; y2: float
  
  def centroid(self) -> Tuple[float, float]:
    return ((self.x1 + self.x2)/2.0, (self.y1 + self.y2)/2.0)
  
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