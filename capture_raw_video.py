from __future__ import annotations

import argparse
import logging
import sys
import time
from datetime import datetime
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from camera.capture import Capture
from utils.camera_scheduler import CameraProfileScheduler
from utils.config import load_caster_config, resolve_caster_id
from utils.logging import setup_logging


logger = logging.getLogger("raw_video_capture")


def _caster_id_arg(raw: str) -> int:
    try:
        return resolve_caster_id(raw)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


def _video_source(raw: str) -> int | str:
    text = raw.strip()
    if text.isdecimal():
        return int(text)
    return raw


def _validate_mode_words(words: list[str]) -> None:
    if not words:
        return
    normalized = " ".join(words).strip().lower().replace("_", " ")
    if normalized != "raw video":
        raise SystemExit(f"Unknown positional argument(s): {' '.join(words)}. Expected optional words: raw video")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Record raw, unannotated video from a caster camera for ML training data."
    )
    parser.add_argument("--caster", type=_caster_id_arg, default=1, help="Caster id, e.g. 1 or caster_1.")
    parser.add_argument("--caster-config", default=None, help="Path to caster config directory or legacy caster YAML.")
    parser.add_argument(
        "mode_words",
        nargs="*",
        help="Optional compatibility words. You may pass: raw video",
    )
    parser.add_argument("--minutes", type=float, required=True, help="Recording duration in minutes.")
    parser.add_argument("--seconds", type=float, default=0.0, help="Extra recording duration in seconds.")
    parser.add_argument(
        "--video-source",
        type=_video_source,
        default=None,
        help="Override runtime video_source, e.g. 0, /dev/video0, gige, basler, or a file path.",
    )
    parser.add_argument("--output", default=None, help="Exact output video path.")
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory. Defaults to var/caster_<id>/raw_video.",
    )
    parser.add_argument("--ext", default="mp4", help="Output extension when --output is not set.")
    parser.add_argument("--codec", default="mp4v", help="OpenCV fourcc codec, e.g. mp4v, XVID, MJPG.")
    parser.add_argument("--fps", type=float, default=None, help="Saved video FPS. Defaults to camera config FPS.")
    parser.add_argument("--warmup-frames", type=int, default=10, help="Frames to warm up supported cameras.")
    parser.add_argument("--progress-seconds", type=float, default=10.0, help="Console progress interval in seconds.")
    parser.add_argument("--log-level", default="INFO", help="DEBUG, INFO, WARNING, ERROR, or CRITICAL.")
    args = parser.parse_args(argv)

    _validate_mode_words(args.mode_words)
    duration_s = (args.minutes * 60.0) + args.seconds
    if duration_s <= 0:
        raise SystemExit("--minutes/--seconds must produce a duration greater than zero.")
    if args.fps is not None and args.fps <= 0:
        raise SystemExit("--fps must be greater than zero.")
    if len(args.codec) != 4:
        raise SystemExit("--codec must be a 4-character OpenCV fourcc, e.g. mp4v or MJPG.")
    return args


def _default_output_path(caster_key: str, storage_path: str, ext: str, output_dir: str | None) -> Path:
    suffix = ext.lstrip(".") or "mp4"
    base_dir = Path(output_dir) if output_dir else Path(storage_path) / "raw_video"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return base_dir / f"{caster_key}_raw_{timestamp}.{suffix}"


def _open_writer(output_path: Path, codec: str, fps: float, frame_shape: tuple[int, ...]):
    import cv2

    height, width = frame_shape[:2]
    is_color = len(frame_shape) == 3 and frame_shape[2] > 1
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*codec)
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height), isColor=is_color)
    if not writer.isOpened():
        raise RuntimeError(
            f"Could not open video writer for {output_path}. "
            f"Try --codec MJPG --ext avi if this OpenCV build cannot write {output_path.suffix}."
        )
    return writer


def record_raw_video(args: argparse.Namespace) -> Path:
    import cv2

    runtime_overrides = {}
    if args.video_source is not None:
        runtime_overrides["video_source"] = args.video_source

    cfg = load_caster_config(
        args.caster,
        args.caster_config,
        runtime_overrides=runtime_overrides,
    )
    duration_s = (args.minutes * 60.0) + args.seconds
    fps = float(args.fps or (cfg.camera_cfg.fps if cfg.camera_cfg else cfg.runtime.max_fps) or 8)
    output_path = (
        Path(args.output)
        if args.output
        else _default_output_path(cfg.caster_key, cfg.caster_storage_path, args.ext, args.output_dir)
    )

    capture = Capture(
        source=cfg.runtime.video_source,
        camera_cfg=cfg.camera_cfg,
        warmup_frames=max(0, int(args.warmup_frames)),
    )
    scheduler: CameraProfileScheduler | None = None
    writer = None
    frame_count = 0
    empty_reads = 0
    start_monotonic = 0.0
    last_progress = 0.0

    logger.info(
        "Starting raw video capture | caster=%s | source=%s | duration_s=%.1f | fps=%.2f | output=%s",
        cfg.caster_id,
        cfg.runtime.video_source,
        duration_s,
        fps,
        output_path,
    )

    try:
        capture.open()
        if cfg.camera_cfg and cfg.camera_cfg.profiles and capture._is_gige():
            scheduler = CameraProfileScheduler(capture, cfg.camera_cfg.profiles)
            scheduler.start()

        start_monotonic = time.monotonic()
        last_progress = start_monotonic
        deadline = start_monotonic + duration_s

        while time.monotonic() < deadline:
            item = capture.read()
            if item is None:
                empty_reads += 1
                time.sleep(0.05)
                continue

            frame, _ts = item
            if writer is None:
                writer = _open_writer(output_path, args.codec, fps, frame.shape)
                logger.info("Video writer opened | shape=%s | codec=%s", frame.shape, args.codec)

            writer.write(frame)
            frame_count += 1

            now = time.monotonic()
            if args.progress_seconds > 0 and now - last_progress >= args.progress_seconds:
                elapsed = now - start_monotonic
                recorded_fps = frame_count / elapsed if elapsed > 0 else 0.0
                remaining = max(0.0, deadline - now)
                logger.info(
                    "Recording progress | elapsed=%.1fs | remaining=%.1fs | frames=%d | capture_fps=%.2f",
                    elapsed,
                    remaining,
                    frame_count,
                    recorded_fps,
                )
                last_progress = now
    except KeyboardInterrupt:
        logger.info("Interrupted by user; closing partial video.")
    finally:
        if writer is not None:
            writer.release()
        if scheduler is not None:
            scheduler.stop()
        capture.close()
        cv2.destroyAllWindows()

    elapsed = max(0.0, time.monotonic() - start_monotonic) if start_monotonic else 0.0
    if writer is None or frame_count == 0:
        raise RuntimeError("No frames were captured; no video file was written.")

    logger.info(
        "Finished raw video capture | output=%s | frames=%d | elapsed_s=%.1f | empty_reads=%d",
        output_path,
        frame_count,
        elapsed,
        empty_reads,
    )
    return output_path


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    setup_logging(level=args.log_level, log_path=None)
    output_path = record_raw_video(args)
    print(f"[OK] Raw video saved: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
