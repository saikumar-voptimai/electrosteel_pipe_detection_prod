from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from geometry.roi import PolygonROI
from vision.types import BBox


class ROIIntersectionTests(unittest.TestCase):
  def test_bbox_intersects_roi_for_overlap_containment_and_crossing_edges(self) -> None:
    roi = PolygonROI("roi_gate2_closed", [(10, 10), (20, 10), (20, 20), (10, 20)])

    self.assertTrue(roi.intersects_bbox(BBox(15, 15, 30, 30)))
    self.assertTrue(roi.intersects_bbox(BBox(0, 0, 30, 30)))
    self.assertTrue(roi.intersects_bbox(BBox(0, 15, 30, 16)))
    self.assertFalse(roi.intersects_bbox(BBox(0, 0, 9, 9)))

  def test_bbox_intersects_requires_positive_overlap_area(self) -> None:
    self.assertTrue(BBox(0, 0, 10, 10).intersects(BBox(5, 5, 15, 15)))
    self.assertFalse(BBox(0, 0, 10, 10).intersects(BBox(10, 0, 20, 10)))


if __name__ == "__main__":
  unittest.main()
