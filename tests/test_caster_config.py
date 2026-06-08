from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from utils.config import (
  load_caster_config,
  resolve_caster_database_path,
  resolve_caster_id,
  resolve_caster_storage_path,
  validate_caster_id,
)


def _write_yaml(path: Path, payload: dict) -> None:
  path.parent.mkdir(parents=True, exist_ok=True)
  path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def _minimal_runtime() -> dict:
  return {
    "debug_mode": False,
    "video_source": "base.mp4",
    "model_path": "models/model.engine",
    "tracker_yaml": "base/bytetrack.yaml",
    "imgsz": 640,
    "conf": 0.25,
    "iou": 0.5,
    "device": "cpu",
    "half": False,
    "max_fps": 30,
    "frame_skip": 0,
    "update_fps": 0,
    "db_path": "var/base/pipes.db",
    "latest_jpg_path": "var/base/latest.jpg",
    "publish_fps": 5,
    "publish_imgsz": 0,
    "publish_overlay": True,
    "run_headless": True,
    "db_flush_interval_s": 1.0,
    "log_level": "INFO",
    "log_path": "var/base/pipe_detect.log",
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


def _minimal_rois(legacy: bool = False) -> dict:
  origin_name = "roi_caster5_origin" if legacy else "roi_caster_origin"
  square = [[0, 0], [10, 0], [10, 10], [0, 10]]
  return {
    "roi_loadcell": square,
    origin_name: square,
    "roi_left_origin": square,
    "roi_right_origin": square,
    "roi_safety_critical": square,
    "roi_gate1_open": square,
    "roi_gate2_open": square,
    "roi_gate1_closed": square,
    "roi_gate2_closed": square,
  }


def _caster_fixture(tmp_path: Path, caster_id: int = 1, legacy_rois: bool = False) -> Path:
  caster_dir = tmp_path / f"caster_{caster_id}"
  runtime = caster_dir / "runtime.yaml"
  rois = caster_dir / "rois.yaml"
  camera = caster_dir / "camera.yaml"
  plc = caster_dir / "plc.yaml"
  weight = caster_dir / "weight.yaml"
  bytetrack = caster_dir / "bytetrack.yaml"

  runtime_payload = _minimal_runtime()
  runtime_payload["video_source"] = caster_id - 1
  _write_yaml(runtime, runtime_payload)
  _write_yaml(rois, _minimal_rois(legacy=legacy_rois))
  _write_yaml(camera, {"camera": {"id": caster_id - 1, "width": 960, "height": 640, "fps": 8}})
  _write_yaml(plc, {"mode": "mock", "tags": {f"caster_{caster_id}_new": f"caster_{caster_id}_new"}, "pulse_ms": 300})
  _write_yaml(weight, {"enabled": False, "machines": {}})
  _write_yaml(bytetrack, {"tracker_type": "bytetrack"})
  return caster_dir


class CasterConfigTests(unittest.TestCase):
  def test_valid_caster_ids(self) -> None:
    for caster_id in [1, 2, 8, 9, 25]:
      self.assertEqual(validate_caster_id(caster_id), caster_id)
      self.assertEqual(resolve_caster_id(f"caster_{caster_id}"), caster_id)
      self.assertEqual(resolve_caster_id(str(caster_id)), caster_id)

  def test_invalid_caster_ids_fail(self) -> None:
    for caster_id in [0, -1]:
      with self.assertRaises(ValueError):
        validate_caster_id(caster_id)
    with self.assertRaises(ValueError):
      resolve_caster_id("camera_1")

  def test_caster_config_path_resolution_and_overrides(self) -> None:
    with tempfile.TemporaryDirectory() as tmp:
      tmp_path = Path(tmp)
      caster_config = _caster_fixture(tmp_path, caster_id=2)
      cfg = load_caster_config("caster_2", str(caster_config))

      self.assertEqual(cfg.caster_id, 2)
      self.assertEqual(cfg.caster_key, "caster_2")
      self.assertEqual(cfg.caster_config_path, str(caster_config))
      self.assertTrue(cfg.rois_path.endswith("caster_2/rois.yaml"))
      self.assertTrue(cfg.camera_cfg_path.endswith("caster_2/camera.yaml"))
      self.assertEqual(cfg.runtime.video_source, 1)
      self.assertTrue(cfg.runtime.tracker_yaml.endswith("caster_2/bytetrack.yaml"))
      self.assertTrue(cfg.runtime.db_path.endswith("var/caster_2/caster_2_pipes.db"))
      self.assertTrue(cfg.runtime.latest_jpg_path.endswith("var/caster_2/latest.jpg"))
      self.assertTrue(cfg.runtime.log_path.endswith("var/caster_2/pipe_detect.log"))

  def test_legacy_caster5_origin_maps_to_general_name(self) -> None:
    with tempfile.TemporaryDirectory() as tmp:
      caster_config = _caster_fixture(Path(tmp), caster_id=5, legacy_rois=True)
      cfg = load_caster_config(5, str(caster_config))

      self.assertIn("roi_caster_origin", cfg.rois)
      self.assertEqual(cfg.rois["roi_caster_origin"], cfg.rois["roi_caster5_origin"])

  def test_dynamic_storage_and_database_paths(self) -> None:
    self.assertEqual(str(resolve_caster_storage_path("caster_12")), "var/caster_12")
    self.assertEqual(str(resolve_caster_database_path(12)), "var/caster_12/caster_12_pipes.db")

  def test_per_caster_paths_are_unique(self) -> None:
    with tempfile.TemporaryDirectory() as tmp:
      tmp_path = Path(tmp)
      cfg1 = load_caster_config(1, str(_caster_fixture(tmp_path / "c1", caster_id=1)))
      cfg2 = load_caster_config(2, str(_caster_fixture(tmp_path / "c2", caster_id=2)))

      self.assertNotEqual(cfg1.runtime.db_path, cfg2.runtime.db_path)
      self.assertNotEqual(cfg1.runtime.latest_jpg_path, cfg2.runtime.latest_jpg_path)
      self.assertNotEqual(cfg1.runtime.log_path, cfg2.runtime.log_path)

  def test_basler_bandwidth_settings_are_loaded(self) -> None:
    with tempfile.TemporaryDirectory() as tmp:
      caster_config = _caster_fixture(Path(tmp), caster_id=2)
      _write_yaml(
        caster_config / "camera.yaml",
        {
          "camera": {
            "type": "basler",
            "width": 1920,
            "height": 1280,
            "fps": 5,
            "basler": {
              "serial_number": "25343513",
              "ip_address": "192.168.1.124",
              "pixel_format": "Mono8",
              "width": 1920,
              "height": 1280,
              "offset_x": 0,
              "offset_y": 0,
              "acquisition_frame_rate": 5,
              "packet_size": 1500,
              "inter_packet_delay": 1000,
              "max_num_buffer": 30,
            },
          }
        },
      )

      cfg = load_caster_config(2, str(caster_config))

      self.assertIsNotNone(cfg.camera_cfg)
      self.assertEqual(cfg.camera_cfg.width, 1920)
      self.assertEqual(cfg.camera_cfg.height, 1280)
      self.assertEqual(cfg.camera_cfg.fps, 5)
      self.assertIsNotNone(cfg.camera_cfg.basler)
      self.assertEqual(cfg.camera_cfg.basler.width, 1920)
      self.assertEqual(cfg.camera_cfg.basler.height, 1280)
      self.assertEqual(cfg.camera_cfg.basler.acquisition_frame_rate, 5)
      self.assertEqual(cfg.camera_cfg.basler.packet_size, 1500)
      self.assertEqual(cfg.camera_cfg.basler.inter_packet_delay, 1000)
      self.assertEqual(cfg.camera_cfg.basler.max_num_buffer, 30)


if __name__ == "__main__":
  unittest.main()
