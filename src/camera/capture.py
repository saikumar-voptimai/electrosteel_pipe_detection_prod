from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field

from camera.clients import CameraClient
from camera.factory import create_camera_client, resolve_camera_type
from utils.config import CameraCfg, CameraProfileCfg

logger = logging.getLogger(__name__)


@dataclass
class Capture:
    source: int | str
    camera_cfg: CameraCfg | None = None
    warmup_frames: int = 10
    max_retries: int = 5
    reconnect_sleep_s: float = 1.0
    empty_read_reconnect_threshold: int = 5
    _client: CameraClient | None = field(default=None, init=False)
    _empty_reads: int = field(default=0, init=False)

    def _is_gige(self) -> bool:
        return resolve_camera_type(self.camera_cfg, self.source) == "va_imaging"

    def open(self) -> None:
        self._client = create_camera_client(
            source=self.source,
            camera_cfg=self.camera_cfg,
            warmup_frames=self.warmup_frames,
        )
        self._client.open()

    def read(self):
        if self._client is None:
            self.open()

        try:
            item = self._client.read()
            if item is not None:
                self._empty_reads = 0
                return item
        except Exception as exc:
            logger.warning("Camera read failed: %s", exc)

        self._empty_reads += 1
        if self._empty_reads < max(1, int(self.empty_read_reconnect_threshold)):
            logger.warning(
                "Camera returned no frame; waiting before reconnect | empty_reads=%s/%s",
                self._empty_reads,
                self.empty_read_reconnect_threshold,
            )
            return None

        logger.warning("Reconnect triggered...")
        self._empty_reads = 0
        self.close()
        cfg = self.camera_cfg.reconnect if self.camera_cfg and self.camera_cfg.reconnect else None
        max_retries = cfg.max_retries if cfg else self.max_retries
        sleep_s = cfg.sleep_s if cfg else self.reconnect_sleep_s
        for attempt in range(max_retries):
            try:
                time.sleep(sleep_s)
                logger.info("Reconnect attempt %s/%s", attempt + 1, max_retries)
                self.open()
                logger.info("Reconnect successful")
                return None
            except Exception as exc:
                logger.error("Reconnect attempt %s failed: %s", attempt + 1, exc)
        logger.critical("Camera reconnect failed after all attempts")
        return None

    def apply_profile(self, profile: CameraProfileCfg | None) -> None:
        if self._client is not None and hasattr(self._client, "apply_profile"):
            self._client.apply_profile(profile)

    def close(self) -> None:
        if self._client is not None:
            self._client.close()
            self._client = None

    def is_open(self) -> bool:
        return bool(self._client is not None and self._client.is_open())

    def get_metadata(self) -> dict:
        if self._client is None:
            return {"camera_type": resolve_camera_type(self.camera_cfg, self.source), "source": self.source}
        return self._client.get_metadata()
