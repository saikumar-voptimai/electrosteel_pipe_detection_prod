from __future__ import annotations
import time
from dataclasses import dataclass, field
from typing import Tuple
import numpy as np
import logging
from utils.config import CameraCfg, CameraProfileCfg
logger = logging.getLogger(__name__)


@dataclass
class Capture:
    source: str
    camera_cfg: CameraCfg | None = None
    warmup_frames: int = 10
    max_retries: int = 5
    reconnect_sleep_s: float = 1.0

    # FIXED: proper dataclass fields
    _device_manager: object | None = field(default=None, init=False)
    _cam: object | None = field(default=None, init=False)
    _gx: object | None = field(default=None, init=False)
    _cap: object | None = field(default=None, init=False)

    # helper to detect if source is Daheng camera
    def _is_gige(self):
        return isinstance(self.source, str) and self.source.startswith("gige")

    # apply camera profile settings, only for Daheng cameras
    def apply_profile(self, profile: CameraProfileCfg | None):
    # Ignore if not Daheng camera
        if not self._is_gige():
            return

        if profile is None or self._cam is None or self._gx is None:
            return
        if self.camera_cfg and (self.camera_cfg.auto_exposure or self.camera_cfg.auto_gain):
            logger.info("Skipping profile (auto exposure/gain enabled)")
            return
        gx = self._gx
        cam = self._cam

        try:
            logger.info(
                "Applying camera profile | exposure=%s gain=%s gamma=%s",
                profile.exposure_us,
                profile.gain_db,
                profile.gamma,
            )

            if hasattr(cam, "ExposureTime"):
                cam.ExposureTime.set(profile.exposure_us)

            if hasattr(cam, "Gain"):
                cam.Gain.set(profile.gain_db)

            if hasattr(cam, "GammaEnable"):
                cam.GammaEnable.set(bool(profile.gamma_enable))

                if profile.gamma_enable:
                    if hasattr(cam, "GammaMode"):
                        cam.GammaMode.set(gx.GxGammaModeEntry.USER)

                    gamma_value = float(profile.gamma)
                    gamma_value = max(0.1, min(10.0, gamma_value))

                    if hasattr(cam, "Gamma"):
                        cam.Gamma.set(gamma_value)
        except Exception as e:
            logger.warning("Camera profile apply failed: %s", e)
    def _apply_base_camera_settings(self):
        if self._cam is None or self.camera_cfg is None or self._gx is None:
            return
        cam = self._cam
        gx = self._gx
        cfg = self.camera_cfg
        try:
            logger.info("Applying base camera settings")
            # Exposure auto
            if hasattr(cam, "ExposureAuto"):
                cam.ExposureAuto.set(
                    gx.GxAutoEntry.CONTINUOUS if cfg.auto_exposure else gx.GxAutoEntry.OFF
                )
            # Gain auto
            if hasattr(cam, "GainAuto"):
                cam.GainAuto.set(
                    gx.GxAutoEntry.CONTINUOUS if cfg.auto_gain else gx.GxAutoEntry.OFF
                )
            # FPS control
            if hasattr(cam, "AcquisitionFrameRateMode"):
                cam.AcquisitionFrameRateMode.set(gx.GxSwitchEntry.ON)
            if hasattr(cam, "AcquisitionFrameRate"):
                cam.AcquisitionFrameRate.set(cfg.fps)
            logger.info(
                "Base camera settings applied | auto_exp=%s auto_gain=%s fps=%s",
                cfg.auto_exposure,
                cfg.auto_gain,
                cfg.fps,
            )

        except Exception as e:
            logger.warning("Base camera configuration failed: %s", e)
    # camera open method, to be called on first read or manually if desired
    def open(self):
        if self._is_gige():
            #  IMPORT gxipy only when needed, to avoid import errors on non-Daheng setups
            import gxipy as gx
            self._gx = gx
            logger.info("Opening Daheng camera")
            self._device_manager = gx.DeviceManager()
            dev_num, dev_info_list = self._device_manager.update_all_device_list()

            if dev_num == 0:
                raise RuntimeError("No Daheng camera detected")

            sn = dev_info_list[0]["sn"]
            self._cam = self._device_manager.open_device_by_sn(sn)

            self._apply_base_camera_settings()
            self._cam.stream_on()

            # warmup
            for _ in range(max(0, int(self.warmup_frames))):
                img = self._cam.data_stream[0].get_image()
                if img is not None:
                    img.convert("RGB")

        else:
            import cv2

            logger.info("Opening video file: %s", self.source)

            self._cap = cv2.VideoCapture(self.source)

            if not self._cap.isOpened():
                raise RuntimeError(f"Cannot open video source: {self.source}")

    #  camera read method, returns (frame, timestamp) or None on failure
    def read(self):

        # ---------------- GIGE CAMERA ----------------
        if self._is_gige():
            if self._cam is None:
                self.open()
            try:
                raw = self._cam.data_stream[0].get_image()
                if raw is None:
                    raise RuntimeError("Empty frame")
                rgb = raw.convert("RGB")
                img = rgb.get_numpy_array()
                if img is None:
                    logger.warning("Invalid frame received, skipping")
                    return None
                img = np.ascontiguousarray(img[:, :, ::-1])
                return img, time.time()

            except Exception as e:
                logger.warning("Camera read failed: %s", e)
        # ---------------- VIDEO FILE ----------------
        else:
            if self._cap is None:
                self.open()
            ok, frame = self._cap.read()
            if ok and frame is not None:
                return frame, time.time()
            # loop video
            logger.info("Video ended → restarting")
            self._cap.set(0, 0)
            ok, frame = self._cap.read()
            if ok:
                return frame, time.time()
        # ---------------- RECONNECT ----------------
        logger.warning("Reconnect triggered...")
        self.close()
        cfg = self.camera_cfg.reconnect if self.camera_cfg and self.camera_cfg.reconnect else None
        max_retries = cfg.max_retries if cfg else self.max_retries
        sleep_s = cfg.sleep_s if cfg else self.reconnect_sleep_s
        for attempt in range(max_retries):
            try:
                time.sleep(sleep_s)
                logger.info(f"Reconnect attempt {attempt + 1}/{max_retries}")
                self.open()
                logger.info("Reconnect successful")
                return None  # next loop will read frame
            except Exception as e:
                logger.error(f"Reconnect attempt {attempt + 1} failed: {e}")
        logger.critical("Camera reconnect failed after all attempts")
        return None    
    # camera close method, to be called on shutdown
    def close(self):
        if self._cam is not None:
            try:
                self._cam.stream_off()
                self._cam.close_device()
            except Exception:
                pass
            self._cam = None
        if self._cap is not None:
            self._cap.release()
            self._cap = None