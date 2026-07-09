from __future__ import annotations

import importlib.util
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import yaml
from ultralytics import YOLO
from ultralytics.trackers.byte_tracker import BYTETracker

from utils.device import RuntimeDevice, configure_torch_backend, select_runtime_device
from vision.types import BBox, TrackDet

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Detection:
    cls_name: str
    cls_id: int
    conf: float
    bbox: BBox


class SharedYoloDetector:
    """
    Detection-only YOLO wrapper intended to be shared by multiple cameras.

    Tracking is intentionally kept outside this class so each camera can own a
    separate tracker state.
    """

    def __init__(
        self,
        model_path: str,
        *,
        device: int | str | None = "auto",
        half: bool = False,
    ) -> None:
        self.model_path = model_path
        self._model_suffix = Path(str(model_path)).suffix.lower()
        self._is_tensorrt = self._model_suffix == ".engine"
        self._runtime_device: RuntimeDevice = select_runtime_device(
            requested=device,
            model_path=model_path,
            half=half,
        )
        self._resolved_device = self._runtime_device.ultralytics_device
        configure_torch_backend(self._runtime_device)

        logger.info(
            "Shared inference loading model once | model=%s | model_type=%s | requested=%s | ultralytics_device=%s | torch_device=%s | half=%s | tensorrt_model=%s",
            self.model_path,
            self.model_type,
            self._runtime_device.requested,
            self._resolved_device,
            self._runtime_device.torch_device,
            self._runtime_device.half,
            self._is_tensorrt,
        )
        if self._is_tensorrt and importlib.util.find_spec("tensorrt") is None:
            logger.warning(
                "TensorRT model configured but Python TensorRT bindings are not importable in this environment."
            )

        t0 = time.perf_counter()
        self.model = YOLO(self.model_path)
        if self._model_suffix == ".pt" and self._runtime_device.is_cuda:
            self.model.to(self._runtime_device.torch_device)
        self.load_ms = (time.perf_counter() - t0) * 1000.0
        self.names = getattr(self.model, "names", {}) or {}
        logger.info(
            "Shared inference model ready | model=%s | model_type=%s | load_ms=%.1f | names=%d",
            self.model_path,
            self.model_type,
            self.load_ms,
            len(self.names),
        )

    @property
    def model_type(self) -> str:
        return self._model_suffix.lstrip(".") or "unknown"

    def _predict_kwargs(self, frame: np.ndarray, *, conf: float, iou: float, imgsz: int) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "source": frame,
            "conf": conf,
            "iou": iou,
            "imgsz": imgsz,
            "verbose": False,
        }
        if self._resolved_device is not None and not self._is_tensorrt:
            kwargs["device"] = self._resolved_device
        if self._runtime_device.half:
            kwargs["half"] = True
        return kwargs

    def infer(self, frame: np.ndarray, *, conf: float, iou: float, imgsz: int) -> list[Detection]:
        try:
            import torch
        except Exception:
            torch = None

        kwargs = self._predict_kwargs(frame, conf=conf, iou=iou, imgsz=imgsz)
        if torch is not None:
            with torch.inference_mode():
                results = self.model.predict(**kwargs)
        else:
            results = self.model.predict(**kwargs)

        if not results:
            return []
        boxes = getattr(results[0], "boxes", None)
        if boxes is None or len(boxes) == 0:
            return []

        xyxy = _to_numpy(boxes.xyxy)
        class_ids = _to_numpy(boxes.cls).astype(int)
        confs = _to_numpy(boxes.conf)

        out: list[Detection] = []
        for i in range(len(xyxy)):
            cls_id = int(class_ids[i])
            name = _class_name(self.names, cls_id)
            x1, y1, x2, y2 = map(float, xyxy[i])
            out.append(
                Detection(
                    cls_name=str(name),
                    cls_id=cls_id,
                    conf=float(confs[i]),
                    bbox=BBox(x1, y1, x2, y2),
                )
            )
        return out


class _TrackerDetections:
    def __init__(self, detections: list[Detection]) -> None:
        self.xyxy = np.asarray(
            [[d.bbox.x1, d.bbox.y1, d.bbox.x2, d.bbox.y2] for d in detections],
            dtype=np.float32,
        ).reshape((-1, 4))
        self.conf = np.asarray([d.conf for d in detections], dtype=np.float32)
        self.cls = np.asarray([d.cls_id for d in detections], dtype=np.float32)
        self.xywh = _xyxy_to_xywh(self.xyxy)

    def __len__(self) -> int:
        return int(len(self.conf))

    def __getitem__(self, idx) -> "_TrackerDetections":
        sliced = object.__new__(_TrackerDetections)
        sliced.xyxy = np.asarray(self.xyxy[idx], dtype=np.float32).reshape((-1, 4))
        sliced.conf = np.asarray(self.conf[idx], dtype=np.float32).reshape((-1,))
        sliced.cls = np.asarray(self.cls[idx], dtype=np.float32).reshape((-1,))
        sliced.xywh = _xyxy_to_xywh(sliced.xyxy)
        return sliced


class PerCameraByteTracker:
    """
    Thin adapter around Ultralytics BYTETracker with one instance per camera.
    """

    def __init__(self, tracker_yaml: str, class_names: dict[int, str] | dict[Any, Any]) -> None:
        self.tracker_yaml = tracker_yaml
        self.class_names = class_names
        args = SimpleNamespace(**_load_tracker_cfg(tracker_yaml))
        if getattr(args, "tracker_type", "bytetrack") != "bytetrack":
            raise ValueError(
                f"Shared multi-camera mode currently supports tracker_type='bytetrack', got {args.tracker_type!r}"
            )
        self._tracker = BYTETracker(args=args)

    def update(self, detections: list[Detection], frame: np.ndarray) -> list[TrackDet]:
        results = _TrackerDetections(detections)
        tracks = self._tracker.update(results, frame)
        if tracks is None or len(tracks) == 0:
            return []

        out: list[TrackDet] = []
        for row in np.asarray(tracks):
            x1, y1, x2, y2 = map(float, row[:4])
            track_id = int(row[4])
            conf = float(row[5])
            cls_id = int(row[6])
            cls_name = _class_name(self.class_names, cls_id)
            out.append(
                TrackDet(
                    cls_name=str(cls_name),
                    conf=conf,
                    track_id=track_id,
                    bbox=BBox(x1, y1, x2, y2),
                )
            )
        return out


def _load_tracker_cfg(path: str) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    cfg.setdefault("tracker_type", "bytetrack")
    cfg.setdefault("track_high_thresh", 0.25)
    cfg.setdefault("track_low_thresh", 0.1)
    cfg.setdefault("new_track_thresh", 0.25)
    cfg.setdefault("track_buffer", 30)
    cfg.setdefault("match_thresh", 0.8)
    cfg.setdefault("fuse_score", True)
    return cfg


def _to_numpy(value) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        return value.numpy()
    return np.asarray(value)


def _xyxy_to_xywh(xyxy: np.ndarray) -> np.ndarray:
    xyxy = np.asarray(xyxy, dtype=np.float32).reshape((-1, 4))
    xywh = np.empty_like(xyxy, dtype=np.float32)
    xywh[:, 0] = (xyxy[:, 0] + xyxy[:, 2]) * 0.5
    xywh[:, 1] = (xyxy[:, 1] + xyxy[:, 3]) * 0.5
    xywh[:, 2] = np.maximum(0.0, xyxy[:, 2] - xyxy[:, 0])
    xywh[:, 3] = np.maximum(0.0, xyxy[:, 3] - xyxy[:, 1])
    return xywh


def _class_name(names, cls_id: int) -> str:
    if isinstance(names, dict):
        return str(names.get(cls_id, str(cls_id)))
    if isinstance(names, (list, tuple)) and 0 <= cls_id < len(names):
        return str(names[cls_id])
    return str(cls_id)
