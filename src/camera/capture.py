from __future__ import annotations
import time
from dataclasses import dataclass, field
from typing import Tuple
import numpy as np
import logging
import gxipy as gx

from utils.config import CameraCfg, CameraProfileCfg

logger = logging.getLogger(__name__)


@dataclass
class Capture:
    source: str
    camera_cfg: CameraCfg | None = None
    reconnect_sleep_s: float = 1.0
    warmup_frames: int = 10

    _device_manager: gx.DeviceManager | None = field(default=None, init=False)
    _cam: gx.Device | None = field(default=None, init=False)

    # Base camera configuration
    def _apply_base_camera_settings(self):
        if self._cam is None or self.camera_cfg is None:
            return
        cam = self._cam
        cfg = self.camera_cfg

        try:
            logger.info("Applying base camera settings")
            # Disable/enable auto exposure
            if hasattr(cam, "ExposureAuto"):
                cam.ExposureAuto.set(
                    gx.GxAutoEntry.CONTINUOUS if cfg.auto_exposure else gx.GxAutoEntry.OFF
                )
            # Disable/enable auto gain
            if hasattr(cam, "GainAuto"):
                cam.GainAuto.set(
                    gx.GxAutoEntry.CONTINUOUS if cfg.auto_gain else gx.GxAutoEntry.OFF
                )
            # Frame rate
            if hasattr(cam, "AcquisitionFrameRateMode"):
                cam.AcquisitionFrameRateMode.set(gx.GxSwitchEntry.ON)
            if hasattr(cam, "AcquisitionFrameRate"):
                cam.AcquisitionFrameRate.set(cfg.fps)
            logger.info(
                "Base camera settings applied | fps=%s",
                cfg.fps,
            )
        except Exception as e:
            logger.warning("Base camera configuration failed: %s", e)
    # Apply day/night profile
    def apply_profile(self, profile: CameraProfileCfg | None):

        if profile is None or self._cam is None:
            return
        cam = self._cam

        try:
            logger.info(
                "Applying camera profile | exposure=%s gain=%s gamma=%s",
                profile.exposure_us,
                profile.gain_db,
                profile.gamma,
            )
            # Exposure
            if hasattr(cam, "ExposureTime"):
                cam.ExposureTime.set(profile.exposure_us)
            # Gain
            if hasattr(cam, "Gain"):
                cam.Gain.set(profile.gain_db)
            # Gamma
            if hasattr(cam, "GammaEnable"):
                cam.GammaEnable.set(bool(profile.gamma_enable))
                if profile.gamma_enable:
                    if hasattr(cam, "GammaMode"):
                        cam.GammaMode.set(gx.GxGammaModeEntry.USER)
                    gamma_value = float(profile.gamma)
                    # clamp to Daheng supported range
                    gamma_value = max(0.1, min(10.0, gamma_value))
                    if hasattr(cam, "Gamma"):
                        cam.Gamma.set(gamma_value)
                    logger.info("Gamma enabled | value=%.3f", gamma_value)

        except Exception as e:
            logger.warning("Camera profile apply failed: %s", e)


    # Open camera
    def open(self) -> None:

        logger.info("Opening Daheng GigE camera via gxipy")
        self._device_manager = gx.DeviceManager()
        dev_num, dev_info_list = self._device_manager.update_all_device_list()

        if dev_num == 0:
            raise RuntimeError("No Daheng camera detected")
        logger.info("Detected cameras: %s", dev_info_list)
        sn = dev_info_list[0]["sn"]
        self._cam = self._device_manager.open_device_by_sn(sn)

        # Apply base settings
        self._apply_base_camera_settings()
        # Start camera streaming
        self._cam.stream_on()
        logger.info("Camera stream started")
        # Warmup frames
        for _ in range(max(0, int(self.warmup_frames))):
            img = self._cam.data_stream[0].get_image()
            if img is not None:
                img.convert("RGB")
    # Read frame
    def read(self) -> Tuple[np.ndarray, float] | None:
        if self._cam is None:
            self.open()

        try:
            raw = self._cam.data_stream[0].get_image()
            if raw is None:
                return None
            rgb = raw.convert("RGB")
            img = rgb.get_numpy_array()
            if img is None:
                return None
            # RGB → BGR for OpenCV
            img = img[:, :, ::-1]
            return img, time.time()

        except Exception as e:
            logger.warning("Capture read failed: %s", e)
            logger.warning("Attempting reconnect")
            self.close()
            time.sleep(self.reconnect_sleep_s)
            try:
                self.open()
                return self.read()
            except Exception:
                logger.error("Capture read failed after reconnect")
                return None

    # Close camera
    def close(self) -> None:
        if self._cam is not None:
            logger.info("Closing Daheng camera")
            try:
                self._cam.stream_off()
                self._cam.close_device()
            except Exception:
                pass
            self._cam = None
            