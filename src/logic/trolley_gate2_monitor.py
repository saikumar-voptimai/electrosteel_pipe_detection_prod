from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

from geometry.roi import ROIManager
from utils.roi_names import RoiName
from vision.types import TrackDet


@dataclass(frozen=True)
class TrolleyGate2Intersection:
  timestamp: float
  trolley_track_id: int
  pipe_on_trolley: bool


@dataclass
class TrolleyGate2Monitor:
  rois: ROIManager
  stale_track_frames: int = 90
  trolley_class_name: str = "trolley"
  pipe_class_name: str = "pipe"
  _recorded_track_ids: set[int] = field(default_factory=set, init=False, repr=False)
  _last_seen_frame_by_track_id: dict[int, int] = field(default_factory=dict, init=False, repr=False)

  def update(self, frame_idx: int, timestamp: float, dets: Iterable[TrackDet]) -> list[TrolleyGate2Intersection]:
    self._forget_stale_tracks(frame_idx)

    pipes: list[TrackDet] = []
    trolleys: list[TrackDet] = []
    trolley_class_name = self.trolley_class_name.lower()
    pipe_class_name = self.pipe_class_name.lower()

    for det in dets:
      cls_name = det.cls_name.strip().lower()
      if cls_name == pipe_class_name:
        pipes.append(det)
      elif cls_name == trolley_class_name and det.track_id is not None:
        trolleys.append(det)
        self._last_seen_frame_by_track_id[int(det.track_id)] = frame_idx

    gate2_closed = self.rois.roi(RoiName.GATE2_CLOSED.value)
    events: list[TrolleyGate2Intersection] = []

    for trolley in trolleys:
      trolley_track_id = int(trolley.track_id)
      if trolley_track_id in self._recorded_track_ids:
        continue
      if not gate2_closed.intersects_bbox(trolley.bbox):
        continue

      pipe_on_trolley = any(pipe.bbox.intersects(trolley.bbox) for pipe in pipes)
      self._recorded_track_ids.add(trolley_track_id)
      events.append(
        TrolleyGate2Intersection(
          timestamp=timestamp,
          trolley_track_id=trolley_track_id,
          pipe_on_trolley=pipe_on_trolley,
        )
      )

    return events

  def _forget_stale_tracks(self, frame_idx: int) -> None:
    stale_track_ids = [
      track_id
      for track_id, last_seen_frame in self._last_seen_frame_by_track_id.items()
      if frame_idx - last_seen_frame > self.stale_track_frames
    ]
    for track_id in stale_track_ids:
      self._last_seen_frame_by_track_id.pop(track_id, None)
      self._recorded_track_ids.discard(track_id)
