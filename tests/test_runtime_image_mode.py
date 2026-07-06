from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from utils.config import load_config
from utils.image_mode import normalize_analysis_image_mode
from utils.runtime import prepare_analysis_frame, resize_for_inference


def _write_yaml(path: Path, payload: dict) -> None:
  path.parent.mkdir(parents=True, exist_ok=True)
  path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def _minimal_runtime() -> dict:
  return {
    "debug_mode": False,
    "video_source": "test.mp4",
    "model_path": "models/model.engine",
    "tracker_yaml": "config/bytetrack.yaml",
    "imgsz": 640,
    "conf": 0.25,
    "iou": 0.5,
    "device": "cpu",
    "half": False,
    "max_fps": 30,
    "frame_skip": 0,
    "update_fps": 0,
    "db_path": "var/test/pipes.db",
    "latest_jpg_path": "var/test/latest.jpg",
    "publish_fps": 5,
    "publish_imgsz": 0,
    "publish_overlay": True,
    "run_headless": True,
    "db_flush_interval_s": 1.0,
    "log_level": "INFO",
    "log_path": "var/test/pipe_detect.log",
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


def _minimal_rois() -> dict:
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


def _write_config_files(root: Path, runtime_payload: dict) -> tuple[Path, Path, Path, Path, Path]:
  runtime = root / "runtime.yaml"
  rois = root / "rois.yaml"
  plc = root / "plc.yaml"
  camera = root / "camera.yaml"
  weight = root / "weight.yaml"

  _write_yaml(runtime, runtime_payload)
  _write_yaml(rois, _minimal_rois())
  _write_yaml(plc, {"mode": "mock", "tags": {"new": "new"}, "pulse_ms": 300})
  _write_yaml(camera, {"camera": {"id": 0, "width": 960, "height": 640, "fps": 8}})
  _write_yaml(weight, {"enabled": False, "machines": {}})

  return runtime, rois, plc, camera, weight


def _load_temp_config(runtime_payload: dict):
  tmp = tempfile.TemporaryDirectory()
  paths = _write_config_files(Path(tmp.name), runtime_payload)
  runtime, rois, plc, camera, weight = paths
  return tmp, load_config(str(runtime), str(rois), str(plc), str(camera), str(weight))


class RuntimeImageModeTests(unittest.TestCase):
  def test_normalizes_rgb_aliases(self) -> None:
    for value in [None, "rgb", "RGB", "rbg", "color", "colour", "bgr", " color "]:
      self.assertEqual(normalize_analysis_image_mode(value), "rgb")

  def test_normalizes_black_and_white_aliases(self) -> None:
    aliases = [
      "black_and_white",
      "black_white",
      "bw",
      "gray",
      "grey",
      "grayscale",
      "greyscale",
      "gray_scale",
      "grey_scale",
      "mono",
      "monochrome",
      "black-and-white",
    ]
    for value in aliases:
      self.assertEqual(normalize_analysis_image_mode(value), "black_and_white")

  def test_invalid_mode_raises_during_config_load(self) -> None:
    runtime = _minimal_runtime()
    runtime["analysis_image_mode"] = "infrared"
    with tempfile.TemporaryDirectory() as tmp:
      runtime_path, rois, plc, camera, weight = _write_config_files(Path(tmp), runtime)
      with self.assertRaisesRegex(ValueError, "Invalid analysis_image_mode"):
        load_config(str(runtime_path), str(rois), str(plc), str(camera), str(weight))

  def test_missing_mode_defaults_to_rgb_during_config_load(self) -> None:
    tmp, cfg = _load_temp_config(_minimal_runtime())
    self.addCleanup(tmp.cleanup)
    self.assertEqual(cfg.runtime.analysis_image_mode, "rgb")

  def test_rgb_mode_preserves_resized_shape_and_values(self) -> None:
    frame = np.arange(4 * 8 * 3, dtype=np.uint8).reshape((4, 8, 3))
    expected = resize_for_inference(frame, target_width=4)

    actual = prepare_analysis_frame(frame, target_width=4, mode="rgb")

    self.assertEqual(actual.shape, expected.shape)
    np.testing.assert_array_equal(actual, expected)

  def test_black_and_white_mode_returns_three_channel_frame(self) -> None:
    frame = np.array(
      [
        [[0, 0, 255], [0, 255, 0], [255, 0, 0], [10, 20, 30]],
        [[255, 255, 255], [40, 80, 120], [120, 80, 40], [0, 0, 0]],
      ],
      dtype=np.uint8,
    )

    actual = prepare_analysis_frame(frame, target_width=4, mode="grey_scale")

    self.assertEqual(actual.shape, frame.shape)
    self.assertEqual(actual.shape[2], 3)
    np.testing.assert_array_equal(actual[:, :, 0], actual[:, :, 1])
    np.testing.assert_array_equal(actual[:, :, 1], actual[:, :, 2])


if __name__ == "__main__":
  unittest.main()
