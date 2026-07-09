from __future__ import annotations

import os
import sqlite3
import sys
import tempfile
import time
import unittest
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ui.caster_monitor import (
  active_caster_contexts,
  aggregate_metrics,
  caster_statuses,
  fetch_recent_pipes,
  get_all_casters,
  health_rows,
)


def _write_yaml(path: Path, payload: dict) -> None:
  path.parent.mkdir(parents=True, exist_ok=True)
  path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def _runtime(tmp_path: Path, caster_id: int) -> dict:
  return {
    "debug_mode": False,
    "video_source": "test1.mp4",
    "model_path": "models/model.engine",
    "tracker_yaml": str(tmp_path / f"caster_{caster_id}/bytetrack.yaml"),
    "imgsz": 640,
    "conf": 0.25,
    "iou": 0.5,
    "device": "cpu",
    "half": False,
    "max_fps": 30,
    "frame_skip": 0,
    "update_fps": 0,
    "db_path": "ignored.db",
    "latest_jpg_path": "ignored.jpg",
    "publish_fps": 5,
    "publish_imgsz": 0,
    "publish_overlay": True,
    "run_headless": True,
    "db_flush_interval_s": 1.0,
    "log_level": "INFO",
    "log_path": "log.txt",
    "origin_confirm_frames": 2,
    "loadcell_enter_confirm_frames": 1,
    "loadcell_exit_confirm_frames": 2,
    "stale_track_frames": 45,
    "rearm_empty_frames": 3,
    "min_pipe_gap_seconds": 5,
    "loadcell_covered_per": 50,
    "class_name_to_id": {"pipe": 0},
    "gate": {"source_default": "geometry"},
  }


def _rois() -> dict:
  square = [[0, 0], [10, 0], [10, 10], [0, 10]]
  return {
    "roi_loadcell": square,
    "roi_caster_origin": square,
    "roi_left_origin": square,
    "roi_right_origin": square,
    "roi_safety_critical": square,
    "roi_gate1_open": square,
    "roi_gate2_open": square,
    "roi_gate1_closed": square,
    "roi_gate2_closed": square,
  }


def _caster_dir(tmp_path: Path, caster_id: int) -> Path:
  caster_dir = tmp_path / f"caster_{caster_id}"
  _write_yaml(caster_dir / "config.yaml", {"caster_id": caster_id, "storage_dir": str(tmp_path / "var" / f"caster_{caster_id}")})
  _write_yaml(caster_dir / "runtime.yaml", _runtime(tmp_path, caster_id))
  _write_yaml(caster_dir / "rois.yaml", _rois())
  _write_yaml(caster_dir / "camera.yaml", {"camera": {"type": "basler", "basler": {"serial_number": str(caster_id)}}})
  _write_yaml(caster_dir / "plc.yaml", {"mode": "mock", "tags": {}, "pulse_ms": 300})
  _write_yaml(caster_dir / "weight.yaml", {"enabled": False, "machines": {}})
  _write_yaml(caster_dir / "bytetrack.yaml", {"tracker_type": "bytetrack"})
  return caster_dir


def _create_db(path: Path, detections: int) -> None:
  path.parent.mkdir(parents=True, exist_ok=True)
  now = time.time()
  conn = sqlite3.connect(path)
  conn.execute(
    """
    CREATE TABLE pipes (
      pipe_uid TEXT PRIMARY KEY,
      origin TEXT,
      t_origin REAL,
      t_loadcell_enter REAL,
      t_loadcell_exit REAL,
      weight REAL,
      weight_quality TEXT,
      weight_samples INTEGER,
      avg_conf_full REAL,
      avg_conf_till_gate REAL,
      frames_missing INTEGER,
      state TEXT,
      last_seen_ts REAL
    )
    """
  )
  for i in range(detections):
    conn.execute(
      "INSERT INTO pipes(pipe_uid, origin, t_origin, avg_conf_full, state, last_seen_ts) VALUES(?,?,?,?,?,?)",
      (f"p{i}", "caster", now - i, 0.8, "active", now - i),
    )
  conn.commit()
  conn.close()


class CasterMonitorTests(unittest.TestCase):
  def test_discovers_configured_casters_and_aggregates_metrics(self) -> None:
    with tempfile.TemporaryDirectory() as tmp:
      root = Path(tmp)
      _caster_dir(root, 1)
      _caster_dir(root, 2)

      contexts = get_all_casters(root)
      self.assertEqual([ctx.caster_key for ctx in contexts], ["caster_1", "caster_2"])

      _create_db(contexts[0].db_path, 2)
      _create_db(contexts[1].db_path, 3)
      metrics = aggregate_metrics(contexts)

      self.assertEqual(metrics.last_hour, 5)
      self.assertEqual(metrics.last_8h, 5)
      self.assertEqual(metrics.last_24h, 5)

      self.assertEqual(len(fetch_recent_pipes(contexts[0], limit=1)[0]), 14)

  def test_health_rows_are_dynamic(self) -> None:
    with tempfile.TemporaryDirectory() as tmp:
      contexts = get_all_casters(Path(tmp))
      self.assertEqual(health_rows(contexts), [])

  def test_active_casters_follow_latest_frame_age(self) -> None:
    with tempfile.TemporaryDirectory() as tmp:
      root = Path(tmp)
      _caster_dir(root, 1)
      _caster_dir(root, 2)
      contexts = get_all_casters(root)

      contexts[0].latest_frame_path.write_bytes(b"fresh")
      contexts[1].latest_frame_path.write_bytes(b"stale")
      stale_ts = time.time() - 600
      os.utime(contexts[1].latest_frame_path, (stale_ts, stale_ts))

      statuses = caster_statuses(contexts, active_after_s=30)
      self.assertTrue(statuses["caster_1"].is_active)
      self.assertFalse(statuses["caster_2"].is_active)
      self.assertEqual(
        [ctx.caster_key for ctx in active_caster_contexts(contexts, active_after_s=30)],
        ["caster_1"],
      )

      rows = health_rows(contexts, active_after_s=30, statuses=statuses)
      self.assertEqual([row["Active"] for row in rows], ["Yes", "No"])

if __name__ == "__main__":
  unittest.main()
