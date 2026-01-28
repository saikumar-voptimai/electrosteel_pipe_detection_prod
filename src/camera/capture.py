from __future__ import annotations
import time
from dataclasses import dataclass
from dataclasses import field
import cv2
import numpy as np
from typing import Tuple
import logging
import subprocess
from utils.config import CameraCfg

logger = logging.getLogger(__name__)

@dataclass
class Capture:
  source: int | str
  camera_cfg: CameraCfg | None = None
  reconnect_sleep_s: float = 1.0
  warmup_frames: int = 10

  _cap: cv2.VideoCapture | None = field(default=None, init=False)

  def open(self) -> None:
    """
    Opens the video capture source.
    """
    if isinstance(self.source, str) and self.source.startswith("gige"):
      logger.info("Opening GigE camera via GStreamer Aravis: %s", self.source)

      if self.camera_cfg is None:
        raise RuntimeError(
          "video_source is 'gige' but no camera_cfg was provided. "
          "Check config/camera.yaml and main.py --camera argument."
        )

      # Release existing camera uses if any with pkill
      # Note: these tools are typically available on Linux; make this best-effort.
      try:
        subprocess.run(["pkill", "-f", "arv-viewer"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        time.sleep(1.0)
        subprocess.run(["pkill", "-f", "arv-test"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        time.sleep(1.0)
        subprocess.run(["pkill", "-f", "gst-launch-1.0"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        time.sleep(1.0)
      except FileNotFoundError:
        logger.debug("Process cleanup tools not found; skipping pkill")

      # List cameras and see "Daheng" exists in the output. Also log.
      try:
        output = subprocess.run(
          ["arv-tool-0.10", "list"],
          capture_output=True,
          text=True,
        )
        if "daheng" in (output.stdout or "").lower():
          logger.info("Daheng camera detected:\n%s", output.stdout)
      except FileNotFoundError:
        logger.debug("arv-tool-0.10 not found; skipping camera list")
      
      pipeline = (
          f"aravissrc camera-name={self.camera_cfg.id} ! "
          f"exposure={self.camera_cfg.exposure_us} ! "
          f"exposure-auto={'true' if self.camera_cfg.auto_exposure else 'false'} ! "
          f"gain={self.camera_cfg.gain_db} ! "
          f"gain-auto={'true' if self.camera_cfg.auto_gain else 'false'} ! "
          f"packet-size=1500 ! "
          f"video/x-raw,width={self.camera_cfg.width},height={self.camera_cfg.height},framerate={self.camera_cfg.fps}/1 ! "
          "bayer2rgb ! "
          "videoconvert ! "
          "video/x-raw,format=BGR ! "
          "appsink drop=true max-buffers=1 sync=false"
      )

      self._cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
      if not self._cap.isOpened():
        raise RuntimeError(
          f"Cannot open GigE camera via GStreamer: {self.source}"
          "Check: gst-inspect-1.0 aravissrc, GST_PLUGIN_PATH, and camera connectivity.")

    else:
      self._cap = cv2.VideoCapture(self.source)
      logger.info("Opening video source: %s", self.source)
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

    # Warmup (single place to avoid skipping extra frames)
    for _ in range(max(0, int(self.warmup_frames))):
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