from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from db.repo import SqliteRepo


class TrolleyGate2RepoTests(unittest.TestCase):
  def test_insert_and_fetch_trolley_gate2_intersections(self) -> None:
    repo = SqliteRepo(":memory:")
    self.addCleanup(repo.close)

    repo.insert_trolley_gate2_intersection(timestamp=12.5, trolley_track_id=42, pipe_on_trolley=True)
    repo.insert_trolley_gate2_intersection(timestamp=13.5, trolley_track_id=43, pipe_on_trolley=False)
    repo.commit()

    rows = repo.fetch_trolley_gate2_intersections(limit=10)

    self.assertEqual(rows, [(2, 13.5, 43, 0), (1, 12.5, 42, 1)])


if __name__ == "__main__":
  unittest.main()
