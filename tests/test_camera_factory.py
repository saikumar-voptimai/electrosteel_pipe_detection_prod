from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from camera.clients import BaslerCameraClient, OpenCVCameraClient, VAImagingCameraClient
from camera.factory import create_camera_client, resolve_camera_type
from utils.config import BaslerCameraCfg, CameraCfg


def _camera_cfg(camera_type: str) -> CameraCfg:
  return CameraCfg(
    type=camera_type,
    id=0,
    width=960,
    height=640,
    fps=8,
    auto_exposure=False,
    auto_gain=False,
    basler=BaslerCameraCfg(ip_address="192.168.1.124"),
  )


class CameraFactoryTests(unittest.TestCase):
  def test_factory_selects_va_imaging_client(self) -> None:
    client = create_camera_client("gige", _camera_cfg("va_imaging"), processing_image_type="B/W")
    self.assertIsInstance(client, VAImagingCameraClient)
    self.assertEqual(client.processing_image_type, "B/W")

  def test_factory_selects_basler_client(self) -> None:
    client = create_camera_client("basler", _camera_cfg("basler"), processing_image_type="B/W")
    self.assertIsInstance(client, BaslerCameraClient)
    self.assertEqual(client.basler_cfg.ip_address, "192.168.1.124")
    self.assertEqual(client.processing_image_type, "B/W")

  def test_factory_keeps_opencv_for_plain_sources(self) -> None:
    client = create_camera_client("test1.mp4", None)
    self.assertIsInstance(client, OpenCVCameraClient)
    self.assertEqual(resolve_camera_type(None, "test1.mp4"), "opencv")

  def test_video_source_overrides_physical_camera_config_for_testing(self) -> None:
    client = create_camera_client("test1.mp4", _camera_cfg("va_imaging"))
    self.assertIsInstance(client, OpenCVCameraClient)

  def test_invalid_camera_type_gives_clear_error(self) -> None:
    with self.assertRaisesRegex(ValueError, "Unsupported camera type"):
      create_camera_client("camera", _camera_cfg("unknown"))

  def test_basler_missing_dependency_error_is_lazy_and_clear(self) -> None:
    client = BaslerCameraClient(source="basler", camera_cfg=_camera_cfg("basler"))
    client._import_pylon = lambda: (_ for _ in ()).throw(  # type: ignore[method-assign]
      RuntimeError("Basler camera selected but pypylon is not installed.")
    )

    with self.assertRaisesRegex(RuntimeError, "pypylon is not installed"):
      client.open()

  def test_va_imaging_bw_uses_supported_rgb_conversion_then_grayscale(self) -> None:
    class FakeConvertedImage:
      def get_numpy_array(self):
        return np.array(
          [
            [[255, 0, 0], [0, 255, 0]],
            [[0, 0, 255], [255, 255, 255]],
          ],
          dtype=np.uint8,
        )

    class FakeRawImage:
      mode = None

      def convert(self, mode):
        self.mode = mode
        return FakeConvertedImage()

    raw = FakeRawImage()

    class FakeStream:
      def get_image(self):
        return raw

    class FakeCamera:
      data_stream = [FakeStream()]

    client = VAImagingCameraClient(
      source="gige",
      camera_cfg=_camera_cfg("va_imaging"),
      processing_image_type="B/W",
    )
    client._cam = FakeCamera()

    item = client.read()

    self.assertIsNotNone(item)
    frame, _ = item
    self.assertEqual(raw.mode, "RGB")
    self.assertEqual(frame.shape, (2, 2))
    self.assertEqual(frame.dtype, np.uint8)

  def test_va_imaging_unknown_access_status_is_allowed_to_try_open(self) -> None:
    class FakeAccessStatus:
      UNKNOWN = 0
      READWRITE = 1
      READONLY = 2
      NOACCESS = 3

    class FakeGx:
      GxAccessStatus = FakeAccessStatus

    client = VAImagingCameraClient(source="gige", camera_cfg=_camera_cfg("va_imaging"))
    client._gx = FakeGx()

    client._validate_control_access({"access_status": FakeAccessStatus.UNKNOWN, "sn": "FBJ24120608"})

  def test_va_imaging_readonly_access_status_still_fails_fast(self) -> None:
    class FakeAccessStatus:
      UNKNOWN = 0
      READWRITE = 1
      READONLY = 2
      NOACCESS = 3

    class FakeGx:
      GxAccessStatus = FakeAccessStatus

    client = VAImagingCameraClient(source="gige", camera_cfg=_camera_cfg("va_imaging"))
    client._gx = FakeGx()

    with self.assertRaisesRegex(RuntimeError, "READONLY"):
      client._validate_control_access({"access_status": FakeAccessStatus.READONLY, "sn": "FBJ24120608"})


if __name__ == "__main__":
  unittest.main()
