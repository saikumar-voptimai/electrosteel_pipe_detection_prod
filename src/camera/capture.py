from __future__ import annotations
import time
from dataclasses import dataclass, field
from typing import Tuple
import numpy as np
import logging
import gxipy as gx

from utils.config import CameraCfg

logger = logging.getLogger(__name__)


@dataclass
class Capture:
    source: str
    camera_cfg: CameraCfg | None = None
    reconnect_sleep_s: float = 1.0
    warmup_frames: int = 10

    _device_manager: gx.DeviceManager | None = field(default=None, init=False)
    _cam: gx.Device | None = field(default=None, init=False)
    _converter: gx.ImageFormatConvert | None = field(default=None, init=False)

    def _apply_camera_settings(self):
        """Configure Daheng camera parameters using SDK"""
        if self._cam is None or self.camera_cfg is None:
            return

        try:
            remote = self._cam.get_remote_device_feature_control()

            logger.info("Applying Daheng camera settings")

            if remote.is_writable("ExposureAuto"):
                remote.get_enum_feature("ExposureAuto").set("Off")

            if remote.is_writable("GainAuto"):
                remote.get_enum_feature("GainAuto").set("Off")

            if remote.is_writable("ExposureTime"):
                remote.get_float_feature("ExposureTime").set(
                    self.camera_cfg.exposure_us
                )

            if remote.is_writable("Gain"):
                remote.get_float_feature("Gain").set(
                    self.camera_cfg.gain_db
                )

            if remote.is_writable("AcquisitionFrameRateEnable"):
                remote.get_bool_feature("AcquisitionFrameRateEnable").set(True)

            if remote.is_writable("AcquisitionFrameRate"):
                remote.get_float_feature("AcquisitionFrameRate").set(
                    self.camera_cfg.fps
                )

            logger.info(
                "Camera settings applied | exposure=%s us | gain=%s dB | fps=%s",
                self.camera_cfg.exposure_us,
                self.camera_cfg.gain_db,
                self.camera_cfg.fps
            )

        except Exception as e:
            logger.warning("Camera configuration failed: %s", e)

    def open(self) -> None:
        """Open Daheng camera using gxipy SDK"""
        logger.info("Opening Daheng GigE camera via gxipy")

        self._device_manager = gx.DeviceManager()

        dev_num, dev_info_list = self._device_manager.update_all_device_list()

        if dev_num == 0:
            raise RuntimeError("No Daheng camera detected")

        logger.info("Detected cameras: %s", dev_info_list)

        sn = dev_info_list[0].get("sn")

        self._cam = self._device_manager.open_device_by_sn(sn)

        self._apply_camera_settings()

        self._cam.stream_on()

        self._converter = self._device_manager.create_image_format_convert()

        self._converter.set_dest_format(gx.GxPixelFormatEntry.RGB8)
        self._converter.set_valid_bits(gx.DxValidBit.BIT4_11)

        logger.info("Camera stream started")

        for _ in range(max(0, int(self.warmup_frames))):
            self._cam.data_stream[0].get_image()

    def read(self) -> Tuple[np.ndarray, float] | None:
        """Capture frame from camera"""

        if self._cam is None:
            self.open()

        try:
            raw = self._cam.data_stream[0].get_image()

            if raw is None:
                return None

            buffer_size = self._converter.get_buffer_size_for_conversion(raw)

            rgb_array = (gx.c_ubyte * buffer_size)()

            self._converter.convert(raw, rgb_array, buffer_size, False)

            img = np.frombuffer(
                rgb_array,
                dtype=np.uint8,
                count=buffer_size
            )

            img = img.reshape(
                raw.frame_data.height,
                raw.frame_data.width,
                3
            )
            # Convert RGB to BGR for OpenCV compatibility
            img = img[:, :, ::-1]

            return img, time.time()

        except Exception as e:
            logger.warning("Capture read failed: %s", e)

        logger.warning("Attempting reconnect")

        self.close()
        time.sleep(self.reconnect_sleep_s)

        try:
            self.open()
            raw = self._cam.data_stream[0].get_image()

            if raw is None:
                return None

            buffer_size = self._converter.get_buffer_size_for_conversion(raw)

            rgb_array = (gx.c_ubyte * buffer_size)()

            self._converter.convert(raw, rgb_array, buffer_size, False)

            img = np.frombuffer(
                rgb_array,
                dtype=np.uint8,
                count=buffer_size
            )

            img = img.reshape(
                raw.frame_data.height,
                raw.frame_data.width,
                3
            )
            # Convert RGB to BGR for OpenCV compatibility
            img = img[:, :, ::-1]
            return img, time.time()

        except Exception:
            logger.error("Capture read failed after reconnect")
            return None

    def close(self) -> None:
        """Close camera connection"""

        if self._cam is not None:
            logger.info("Closing Daheng camera")

            try:
                self._cam.stream_off()
                self._cam.close_device()
            except Exception:
                pass

            self._cam = None