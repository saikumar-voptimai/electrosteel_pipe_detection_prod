from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from geometry.roi import ROIManager
from logic.trolley_gate2_monitor import TrolleyGate2Monitor
from utils.roi_names import RoiName
from vision.types import BBox, TrackDet


def _monitor(stale_track_frames: int = 90) -> TrolleyGate2Monitor:
  return TrolleyGate2Monitor(
    rois=ROIManager({RoiName.GATE2_CLOSED.value: [(100, 100), (200, 100), (200, 200), (100, 200)]}),
    stale_track_frames=stale_track_frames,
  )


def _det(cls_name: str, track_id: int | None, bbox: BBox) -> TrackDet:
  return TrackDet(cls_name=cls_name, conf=0.9, track_id=track_id, bbox=bbox)


class TrolleyGate2MonitorTests(unittest.TestCase):
  def test_records_first_gate2_intersection_with_pipe_on_trolley_flag(self) -> None:
    monitor = _monitor()

    events = monitor.update(
      frame_idx=1,
      timestamp=123.0,
      dets=[
        _det("trolley", 7, BBox(150, 150, 250, 250)),
        _det("pipe", None, BBox(160, 160, 180, 180)),
      ],
    )

    self.assertEqual(len(events), 1)
    self.assertEqual(events[0].timestamp, 123.0)
    self.assertEqual(events[0].trolley_track_id, 7)
    self.assertTrue(events[0].pipe_on_trolley)

    duplicate_events = monitor.update(
      frame_idx=2,
      timestamp=124.0,
      dets=[
        _det("trolley", 7, BBox(150, 150, 250, 250)),
        _det("pipe", 31, BBox(160, 160, 180, 180)),
      ],
    )
    self.assertEqual(duplicate_events, [])

  def test_records_zero_when_no_pipe_intersects_trolley(self) -> None:
    events = _monitor().update(
      frame_idx=1,
      timestamp=200.0,
      dets=[
        _det("trolley", 8, BBox(150, 150, 250, 250)),
        _det("pipe", 32, BBox(0, 0, 50, 50)),
      ],
    )

    self.assertEqual(len(events), 1)
    self.assertFalse(events[0].pipe_on_trolley)

  def test_ignores_trolley_outside_gate2_closed_roi(self) -> None:
    events = _monitor().update(
      frame_idx=1,
      timestamp=300.0,
      dets=[_det("trolley", 9, BBox(0, 0, 50, 50))],
    )

    self.assertEqual(events, [])

  def test_allows_same_numeric_track_id_after_it_becomes_stale(self) -> None:
    monitor = _monitor(stale_track_frames=1)
    first = monitor.update(
      frame_idx=1,
      timestamp=1.0,
      dets=[_det("trolley", 10, BBox(150, 150, 250, 250))],
    )
    second = monitor.update(
      frame_idx=3,
      timestamp=3.0,
      dets=[_det("trolley", 10, BBox(150, 150, 250, 250))],
    )

    self.assertEqual(len(first), 1)
    self.assertEqual(len(second), 1)


if __name__ == "__main__":
  unittest.main()
