# utils/video_recorder.py
from __future__ import annotations

import time
import queue
import logging
import threading

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RecordingCfg:
    enabled: bool = False
    output_dir: str = "scripts/video"   
    segment_minutes: float = 10.0
    gap_seconds: float = 0.0

    container: str = "mp4"
    fourcc: str = "mp4v"
    fps: float = 10.0
    frame_size: Optional[Tuple[int, int]] = None  # (w,h); None = infer from first frame

    queue_size: int = 240
    drop_when_full: bool = True

    filename_prefix: str = "cam"
    timestamp_format: str = "%Y%m%d_%H%M%S"


class SegmentedVideoRecorder:
    def __init__(self, cfg: RecordingCfg, project_root: Optional[Path] = None):
        self.cfg = cfg
        self._project_root = project_root

        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

        self._q: "queue.Queue[Tuple[np.ndarray, float]]" = queue.Queue(
            maxsize=max(1, int(cfg.queue_size))
        )

        self._writer: Optional[cv2.VideoWriter] = None
        self._writer_path: Optional[Path] = None

        self._segment_end_ts = 0.0
        self._gap_until_ts = 0.0

        self._frame_size = cfg.frame_size
        self._fourcc = cv2.VideoWriter_fourcc(*cfg.fourcc)

    def start(self) -> None:
        if not self.cfg.enabled:
            logger.info("Video recording disabled")
            return
        if self._thread and self._thread.is_alive():
            return

        out_dir = self._resolve_output_dir()
        out_dir.mkdir(parents=True, exist_ok=True)

        self._stop.clear()
        self._thread = threading.Thread(
            target=self._run, name="segmented_video_recorder", daemon=True
        )
        self._thread.start()
        logger.info("SegmentedVideoRecorder started | out_dir=%s", out_dir)

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2.0)
            if self._thread.is_alive():
                logger.warning("Recorder thread did not stop within timeout")
        self._close_writer()

    def push(self, frame_bgr: np.ndarray, ts: float) -> None:
        if not self.cfg.enabled or self._stop.is_set():
            return
        if frame_bgr is None or not isinstance(frame_bgr, np.ndarray) or frame_bgr.size == 0:
            return

        try:
            if self.cfg.drop_when_full:
                self._q.put_nowait((frame_bgr, ts))
            else:
                self._q.put((frame_bgr, ts), timeout=0.01)
        except queue.Full:
            return  # drop

    # ---------------- internals ----------------

    def _resolve_output_dir(self) -> Path:
        p = Path(self.cfg.output_dir)
        if p.is_absolute():
            return p
        base = self._project_root if self._project_root is not None else Path.cwd()
        return (base / p).resolve()

    def _segment_duration_s(self) -> float:
        return max(1.0, float(self.cfg.segment_minutes) * 60.0)

    def _gap_s(self) -> float:
        return max(0.0, float(self.cfg.gap_seconds))

    def _ext(self) -> str:
        return "avi" if (self.cfg.container or "").strip().lower() == "avi" else "mp4"

    def _new_segment_path(self) -> Path:
        out_dir = self._resolve_output_dir()
        ts = datetime.now().strftime(self.cfg.timestamp_format)
        return out_dir / f"{self.cfg.filename_prefix}_{ts}.{self._ext()}"

    def _open_writer_if_needed(self, frame0: np.ndarray) -> None:
        if self._writer is not None:
            return

        if self._frame_size is None:
            h, w = frame0.shape[:2]
            self._frame_size = (int(w), int(h))

        w, h = self._frame_size
        path = self._new_segment_path()

        writer = cv2.VideoWriter(str(path), self._fourcc, float(self.cfg.fps), (w, h))
        if not writer.isOpened():
            raise RuntimeError(f"VideoWriter open failed: {path}")

        self._writer = writer
        self._writer_path = path
        now = time.time()
        self._segment_end_ts = now + self._segment_duration_s()

        logger.info(
            "Recording segment started | path=%s | size=%dx%d | fps=%.2f",
            path, w, h, float(self.cfg.fps)
        )

    def _close_writer(self) -> None:
        if self._writer is None:
            return
        try:
            self._writer.release()
        except Exception:
            logger.exception("Writer release failed")
        finally:
            logger.info("Recording segment closed | path=%s", self._writer_path)
            self._writer = None
            self._writer_path = None

    def _write_frame(self, frame: np.ndarray) -> None:
        if self._writer is None or self._frame_size is None:
            return
        w, h = self._frame_size
        if frame.shape[1] != w or frame.shape[0] != h:
            frame = cv2.resize(frame, (w, h), interpolation=cv2.INTER_LINEAR)
        self._writer.write(frame)

    def _rotate_segment(self, now: float) -> None:
        self._close_writer()
        self._gap_until_ts = now + self._gap_s()

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                frame, _ts = self._q.get(timeout=0.2)
            except queue.Empty:
                if self._writer is not None and time.time() >= self._segment_end_ts:
                    self._rotate_segment(time.time())
                continue

            now = time.time()

            if now < self._gap_until_ts:
                continue

            if self._writer is not None and now >= self._segment_end_ts:
                self._rotate_segment(now)
                if now < self._gap_until_ts:
                    continue

            if self._writer is None:
                try:
                    self._open_writer_if_needed(frame)
                except Exception:
                    logger.exception("Failed to open writer; retrying")
                    time.sleep(0.5)
                    continue

            try:
                self._write_frame(frame)
            except Exception:
                logger.exception("Write failed; closing segment")
                self._close_writer()
                time.sleep(0.2)
