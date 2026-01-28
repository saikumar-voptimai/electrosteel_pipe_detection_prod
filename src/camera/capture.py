from __future__ import annotations
import time
from dataclasses import dataclass
import cv2
import numpy as np
from typing import Tuple
import logging

logger = logging.getLogger(__name__)

@dataclass
class Capture:
  source: int | str
  reconnect_sleep_s: float = 1.0
  warmup_frames: int = 10

  _cap: cv2.VideoCapture | None = None

  def open(self) -> None:
    """
    Opens the video capture source.
    """
    logger.info("Opening video source: %s", self.source)
    self._cap = cv2.VideoCapture(self.source)
    if not self._cap.isOpened():
      raise RuntimeError(f"Cannot open video source: {self.source}")

    try:
      fps = self._cap.get(cv2.CAP_PROP_FPS)
      w = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
      h = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
      frames = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))
      logger.info("Capture opened | fps=%.2f | size=%dx%d | frame_count=%s", fps, w, h, frames if frames > 0 else "?")
    except Exception:
      logger.debug("Could not read capture properties", exc_info=True)
    # Warmup
    for _ in range(self.warmup_frames):
      self._cap.read()

  def read(self) -> Tuple[np.ndarray, float] | None:
    """
    Reads a frame from the capture source.
    Returns (frame, timestamp) or None if failed.
    """
    if self._cap is None:
      self.open()

    ok, frame = self._cap.read()
    if ok and frame is not None:
      return frame, time.time()
    
    # Try reconnect
    #TODO: Code duplication? Run retries in a loop?
    logger.warning("Capture read failed; reconnecting | source=%s", self.source)
    self.close()
    time.sleep(self.reconnect_sleep_s)
    self.open()
    ok, frame = self._cap.read()
    if ok and frame is not None:
      return frame, time.time()
    logger.error("Capture read failed after reconnect | source=%s", self.source)
    return None
  
  def close(self) -> None:
    """
    Closes the video capture if it is open.
    """
    if self._cap is not None:
      logger.info("Closing video capture")
      self._cap.release()
      self._cap = None