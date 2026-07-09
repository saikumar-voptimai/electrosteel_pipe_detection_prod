from __future__ import annotations

import logging
import os
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import cv2

from app import App
from camera.capture import Capture
from db.repo import SqliteRepo
from geometry.roi import ROIManager
from logic.events import GateClosedEvent, GateOpenedEvent
from logic.weight_service import WeightService
from plc.factory import create_plc
from utils.camera_scheduler import CameraProfileScheduler
from utils.config import AppCfg
from utils.roi_names import REQUIRED_ROIS, as_keys
from utils.runtime import prepare_analysis_frame, resize_for_inference
from utils.timing import RateLimiter
from vision.overlay import LatestFramePublisher, draw_overlay
from vision.shared_inference import Detection, PerCameraByteTracker, SharedYoloDetector
from vision.types import BBox, TrackDet

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class FramePacket:
    caster_id: int
    frame_idx: int
    frame_orig: object
    ts: float
    captured_perf_ts: float


class LatestFrameBuffer:
    """
    Size-1 latest frame slot. New frames replace older unprocessed frames.
    """

    def __init__(self, caster_id: int) -> None:
        self.caster_id = caster_id
        self._lock = threading.Lock()
        self._frame: FramePacket | None = None
        self.put_count = 0
        self.drop_count = 0

    def put(self, frame: FramePacket) -> None:
        with self._lock:
            if self._frame is not None:
                self.drop_count += 1
            self._frame = frame
            self.put_count += 1

    def take_latest(self) -> FramePacket | None:
        with self._lock:
            frame = self._frame
            self._frame = None
            return frame

    def has_frame(self) -> bool:
        with self._lock:
            return self._frame is not None


@dataclass
class CameraCaptureWorker:
    cfg: AppCfg
    buffer: LatestFrameBuffer
    stop_event: threading.Event
    thread: threading.Thread = field(init=False)
    capture: Capture | None = field(default=None, init=False)
    scheduler: CameraProfileScheduler | None = field(default=None, init=False)

    def __post_init__(self) -> None:
        self.thread = threading.Thread(
            target=self.run,
            name=f"CameraCaptureWorker-caster-{self.cfg.caster_id}",
            daemon=True,
        )

    def start(self) -> None:
        self.thread.start()

    def join(self, timeout: float | None = None) -> None:
        self.thread.join(timeout=timeout)

    def run(self) -> None:
        limiter = RateLimiter(self.cfg.runtime.max_fps)
        frame_idx = 0
        last_log = time.time()
        captured_since_log = 0
        try:
            logger.info(
                "Camera capture starting | caster=%s | source=%s | pid=%d",
                self.cfg.caster_id,
                self.cfg.runtime.video_source,
                os.getpid(),
            )
            self.capture = Capture(source=self.cfg.runtime.video_source, camera_cfg=self.cfg.camera_cfg)
            self.capture.open()
            if self.cfg.camera_cfg and self.cfg.camera_cfg.profiles and self.capture._is_gige():
                self.scheduler = CameraProfileScheduler(self.capture, self.cfg.camera_cfg.profiles)
                self.scheduler.start()

            while not self.stop_event.is_set():
                item = self.capture.read()
                if item is None:
                    time.sleep(0.01)
                    continue

                frame_orig, ts = item
                should_publish = not (
                    self.cfg.runtime.frame_skip > 0
                    and (frame_idx % (self.cfg.runtime.frame_skip + 1) != 0)
                )
                if should_publish:
                    self.buffer.put(
                        FramePacket(
                            caster_id=self.cfg.caster_id,
                            frame_idx=frame_idx,
                            frame_orig=frame_orig,
                            ts=ts,
                            captured_perf_ts=time.perf_counter(),
                        )
                    )
                    captured_since_log += 1

                frame_idx += 1
                limiter.sleep_if_needed()

                now = time.time()
                if now - last_log >= 5.0:
                    logger.info(
                        "Camera capture heartbeat | caster=%s | captured_5s=%d | latest_replacements=%d | frame_idx=%d",
                        self.cfg.caster_id,
                        captured_since_log,
                        self.buffer.drop_count,
                        frame_idx,
                    )
                    last_log = now
                    captured_since_log = 0
        except Exception:
            logger.exception("Camera capture worker failed | caster=%s", self.cfg.caster_id)
            self.stop_event.set()
        finally:
            try:
                if self.scheduler:
                    self.scheduler.stop()
            except Exception:
                logger.debug("Camera scheduler stop failed | caster=%s", self.cfg.caster_id, exc_info=True)
            try:
                if self.capture:
                    self.capture.close()
            except Exception:
                logger.debug("Camera close failed | caster=%s", self.cfg.caster_id, exc_info=True)
            logger.info("Camera capture stopped | caster=%s", self.cfg.caster_id)


@dataclass
class PerCameraPipelineState:
    cfg: AppCfg
    model_path: str
    repo: SqliteRepo | None = field(default=None, init=False)
    plc: object | None = field(default=None, init=False)
    weight_service: WeightService | None = field(default=None, init=False)
    rois: ROIManager | None = field(default=None, init=False)
    tracker: PerCameraByteTracker | None = field(default=None, init=False)
    pipe_fsm: object | None = field(default=None, init=False)
    gate_fsm: object | None = field(default=None, init=False)
    publisher: LatestFramePublisher | None = field(default=None, init=False)
    gate_source: str = field(default="", init=False)
    default_gate_source: str = field(default="", init=False)
    last_viz_ts: float = field(default=0.0, init=False)
    last_commit: float = field(default_factory=time.time, init=False)
    last_setting_poll: float = field(default_factory=time.time, init=False)
    last_fps_log: float = field(default_factory=time.time, init=False)
    fps_log_frames: int = field(default=0, init=False)
    processed_frame_idx: int = field(default=0, init=False)
    runfps: float = field(default=1.0, init=False)
    opened: bool = field(default=False, init=False)

    def open(self, class_names: dict) -> None:
        if self.opened:
            return

        for name in as_keys(REQUIRED_ROIS):
            if name not in self.cfg.rois:
                raise RuntimeError(
                    f"Missing required ROI: {name} in {self.cfg.rois_path}. Run --redraw to define ROIs."
                )

        os.makedirs(os.path.dirname(self.cfg.runtime.db_path), exist_ok=True)
        os.makedirs(os.path.dirname(self.cfg.runtime.latest_jpg_path), exist_ok=True)

        self.repo = SqliteRepo(self.cfg.runtime.db_path)
        self.plc = create_plc(self.cfg.plc)
        self.rois = ROIManager(self.cfg.rois)
        self.tracker = PerCameraByteTracker(self.cfg.runtime.tracker_yaml, class_names)

        if getattr(self.cfg, "weight", None) is not None and self.cfg.weight.enabled:
            self.weight_service = WeightService(self.cfg.weight, max_duration_s=30.0)
            logger.info("Weight capture enabled | caster=%s | machine_default=%s", self.cfg.caster_id, self.cfg.weight.machine_id_default)

        app_helper = App(self.cfg)
        pulse_tag = app_helper._resolve_caster_pulse_tag()
        from logic.pipe_fsm import PipeFlowFSM

        self.pipe_fsm = PipeFlowFSM(
            rois=self.rois,
            plc=self.plc,
            pulse_tag=pulse_tag,
            pulse_ms=self.cfg.plc.pulse_ms,
            origin_confirm_frames=self.cfg.runtime.origin_confirm_frames,
            loadcell_enter_confirm_frames=self.cfg.runtime.loadcell_enter_confirm_frames,
            loadcell_exit_confirm_frames=self.cfg.runtime.loadcell_exit_confirm_frames,
            stale_track_frames=self.cfg.runtime.stale_track_frames,
            rearm_empty_frames=self.cfg.runtime.rearm_empty_frames,
            min_pipe_gap_seconds=self.cfg.runtime.min_pipe_gap_seconds,
            loadcell_covered_per=self.cfg.runtime.loadcell_covered_per,
            remove_pipe_id_pipe_checkpoint_not_entered=self.cfg.runtime.remove_pipe_id_pipe_checkpoint_not_entered,
        )

        self.default_gate_source = self.cfg.runtime.gate.source_default
        self.repo.set_setting("gate_source", self.repo.get_setting("gate_source", self.default_gate_source))
        self.gate_source = self.repo.get_setting("gate_source", self.default_gate_source)
        self.gate_fsm = app_helper._build_gate_fsm(self.gate_source, self.rois, self.plc)
        self.publisher = LatestFramePublisher(
            out_path=self.cfg.runtime.latest_jpg_path,
            fps=self.cfg.runtime.publish_fps,
            history_cfg=self.cfg.runtime.history,
            class_name_to_id=self.cfg.runtime.class_name_to_id,
        )
        self.opened = True
        logger.info(
            "Per-camera pipeline ready | caster=%s | model=%s | tracker_yaml=%s | db=%s | latest_jpg=%s",
            self.cfg.caster_id,
            self.model_path,
            self.cfg.runtime.tracker_yaml,
            self.cfg.runtime.db_path,
            self.cfg.runtime.latest_jpg_path,
        )

    def process(
        self,
        packet: FramePacket,
        frame_scaled,
        detections: list[Detection],
        inference_ms: float,
    ) -> None:
        if not self.opened:
            raise RuntimeError(f"Pipeline for caster {self.cfg.caster_id} is not open")
        assert self.rois is not None
        assert self.repo is not None
        assert self.tracker is not None
        assert self.pipe_fsm is not None
        assert self.gate_fsm is not None
        assert self.publisher is not None

        loop_start = packet.captured_perf_ts
        frame_orig = packet.frame_orig
        ts = packet.ts
        orig_h, orig_w = frame_orig.shape[:2]
        scaled_h, scaled_w = frame_scaled.shape[:2]
        inv_scale_x = orig_w / scaled_w
        inv_scale_y = orig_h / scaled_h

        dets_scaled = self.tracker.update(detections, frame_scaled)
        dets_orig = [
            TrackDet(
                cls_name=d.cls_name,
                conf=d.conf,
                track_id=d.track_id,
                bbox=BBox(
                    d.bbox.x1 * inv_scale_x,
                    d.bbox.y1 * inv_scale_y,
                    d.bbox.x2 * inv_scale_x,
                    d.bbox.y2 * inv_scale_y,
                ),
            )
            for d in dets_scaled
            if d.track_id is not None
        ]

        gate_events, gate_metrics = self.gate_fsm.update(frame=frame_orig, dets=dets_orig)
        for event in gate_events:
            if isinstance(event, GateOpenedEvent):
                self.repo.gate_open(event.gate_name, event.t_open)
            elif isinstance(event, GateClosedEvent):
                self.repo.gate_close(event.gate_name, event.t_closed)

        updated_pipes, pipe_events = self.pipe_fsm.update(
            frame_idx=self.processed_frame_idx,
            ts=ts,
            dets=dets_orig,
        )
        self._handle_pipe_events(pipe_events)
        self._drain_weight_results()
        self._upsert_pipes(updated_pipes)
        self._publish_if_needed(frame_orig, dets_orig, ts, gate_metrics)
        self._commit_if_needed()
        self._poll_settings_if_needed()

        self.processed_frame_idx += 1
        loop_ms = (time.perf_counter() - loop_start) * 1000.0
        self.runfps = 1000.0 / loop_ms if loop_ms > 0 else 0.0
        if self.cfg.runtime.debug_mode:
            self.fps_log_frames += 1
            now = time.time()
            if now - self.last_fps_log >= 5.0:
                elapsed = now - self.last_fps_log
                avg_fps = self.fps_log_frames / elapsed if elapsed > 0 else 0.0
                logger.info(
                    "Shared camera result | caster=%s | pid=%d | model=%s | detections=%d | tracks=%d | fps_current=%.2f | fps_avg_5s=%.2f | inference_ms=%.1f | loop_ms=%.1f | frame_idx=%d | source_frame_idx=%d",
                    self.cfg.caster_id,
                    os.getpid(),
                    self.model_path,
                    len(detections),
                    len(dets_orig),
                    self.runfps,
                    avg_fps,
                    inference_ms,
                    loop_ms,
                    self.processed_frame_idx,
                    packet.frame_idx,
                )
                self.last_fps_log = now
                self.fps_log_frames = 0

    def _handle_pipe_events(self, pipe_events: list) -> None:
        assert self.repo is not None
        for event in pipe_events:
            logger.info("Pipe event | caster=%s | event=%s", self.cfg.caster_id, event)
            if event.__class__.__name__ == "PipeEnteredLoadcellEvent":
                if event.pipe_uid:
                    self.repo.insert_event("pipe_enter_loadcell", event.pipe_uid, f"tid={event.tracker_id}")
                else:
                    self.repo.insert_unknown_loadcell_event(
                        "pipe_enter_loadcell",
                        event.tracker_id,
                        "missing_pipe_uid",
                        ts=event.t_enter,
                    )
                if self.plc is not None and "pipe_on_loadcell" in self.cfg.plc.tags:
                    self.plc.pulse(self.cfg.plc.tags["pipe_on_loadcell"], self.cfg.plc.pulse_ms)

                if event.pipe_uid and self.weight_service is not None:
                    ok = self.weight_service.start(
                        pipe_uid=event.pipe_uid,
                        machine_id=self.cfg.weight.machine_id_default,
                    )
                    if ok:
                        self.repo.insert_event(
                            "weight_capture_start",
                            event.pipe_uid,
                            f"machine_id={self.cfg.weight.machine_id_default}",
                        )

            if event.__class__.__name__ == "PipeExitedLoadcellEvent":
                if event.pipe_uid:
                    self.repo.insert_event("pipe_exit_loadcell", event.pipe_uid, f"tid={event.tracker_id}")
                else:
                    self.repo.insert_unknown_loadcell_event(
                        "pipe_exit_loadcell",
                        event.tracker_id,
                        "missing_pipe_uid",
                        ts=event.t_exit,
                    )
                if self.weight_service is not None:
                    self.weight_service.stop()

            if event.__class__.__name__ == "PipeRemovedBeforeCheckpointEvent":
                self.repo.delete_pipe(event.pipe_uid)
                self.repo.insert_event("pipe_deleted", event.pipe_uid, event.reason)

    def _drain_weight_results(self) -> None:
        if self.weight_service is None or self.repo is None:
            return
        for fin in self.weight_service.drain_results():
            w = fin.result.weight
            quality = fin.result.quality
            samples = fin.result.samples
            self.repo.upsert_pipe(
                {
                    "pipe_uid": fin.pipe_uid,
                    "weight": w,
                    "weight_quality": quality,
                    "weight_samples": samples,
                }
            )
            self.repo.insert_event(
                "weight_captured",
                fin.pipe_uid,
                f"weight={w} quality={quality} samples={samples} reason={fin.reason}",
            )

    def _upsert_pipes(self, updated_pipes: list) -> None:
        assert self.repo is not None
        for p in updated_pipes:
            avg_full = (p.conf_sum_full / p.conf_count_full) if p.conf_count_full > 0 else 0.0
            avg_till_gate = (p.conf_sum_till_gate / p.conf_count_till_gate) if p.conf_count_till_gate > 0 else 0.0
            self.repo.upsert_pipe(
                {
                    "pipe_uid": p.pipe_uid,
                    "tracker_id": p.tracker_id,
                    "origin": p.origin,
                    "pipe_checkpoint": 1 if p.pipe_checkpoint else 0,
                    "state": p.state,
                    "t_origin": p.t_origin,
                    "t_loadcell_enter": p.t_loadcell_enter,
                    "t_loadcell_exit": p.t_loadcell_exit,
                    "avg_conf_full": avg_full,
                    "conf_count_full": p.conf_count_full,
                    "avg_conf_till_gate": avg_till_gate,
                    "conf_count_till_gate": p.conf_count_till_gate,
                    "frames_missing": p.frames_missing,
                    "last_seen_ts": p.last_seen_ts,
                    "reached_gate_zone": 1 if int(p.reached_gate_zone) else 0,
                }
            )

    def _publish_if_needed(self, frame_orig, dets_orig: list[TrackDet], ts: float, gate_metrics: dict) -> None:
        if int(self.cfg.runtime.publish_fps) <= 0:
            return
        now = time.time()
        if now - self.last_viz_ts < 1.0 / float(self.cfg.runtime.publish_fps):
            return
        self.last_viz_ts = now

        orig_h, orig_w = frame_orig.shape[:2]
        vis_base = frame_orig
        pub_size = self.cfg.runtime.publish_imgsz
        if isinstance(pub_size, int) and pub_size > 0:
            vis_base = resize_for_inference(frame_orig, target_width=pub_size)
        elif isinstance(pub_size, tuple) and len(pub_size) == 2:
            target_w, target_h = pub_size
            vis_base = cv2.resize(frame_orig, (target_w, target_h))

        vis_h, vis_w = vis_base.shape[:2]
        vis_scale_x = vis_w / float(orig_w)
        vis_scale_y = vis_h / float(orig_h)
        dets_vis = [
            TrackDet(
                cls_name=d.cls_name,
                conf=d.conf,
                track_id=d.track_id,
                bbox=BBox(
                    d.bbox.x1 * vis_scale_x,
                    d.bbox.y1 * vis_scale_y,
                    d.bbox.x2 * vis_scale_x,
                    d.bbox.y2 * vis_scale_y,
                ),
            )
            for d in dets_orig
        ]

        if bool(self.cfg.runtime.publish_overlay):
            vis = draw_overlay(
                vis_base,
                self.rois,
                dets_vis,
                ts,
                scale_x=vis_scale_x,
                scale_y=vis_scale_y,
                gate_metrics=gate_metrics,
                debug=self.cfg.runtime.debug_mode,
                runfps=self.runfps,
            )
            self.publisher.publish(vis)
        else:
            self.publisher.publish(vis_base, dets=dets_vis, ts=ts, gate_metrics=gate_metrics)

    def _commit_if_needed(self) -> None:
        assert self.repo is not None
        if time.time() - self.last_commit >= self.cfg.runtime.db_flush_interval_s:
            self.repo.commit()
            self.last_commit = time.time()

    def _poll_settings_if_needed(self) -> None:
        assert self.repo is not None
        if time.time() - self.last_setting_poll < 200.0:
            return
        new_source = self.repo.get_setting("gate_source", self.default_gate_source)
        if new_source != self.gate_source:
            logger.info(
                "Gate source changed | caster=%s | old=%s | new=%s",
                self.cfg.caster_id,
                self.gate_source,
                new_source,
            )
            self.gate_source = new_source
            app_helper = App(self.cfg)
            self.gate_fsm = app_helper._build_gate_fsm(self.gate_source, self.rois, self.plc)
            self.repo.insert_event("setting_changed", None, f"gate_source={self.gate_source}")
            self.repo.commit()
        self.last_setting_poll = time.time()

    def close(self) -> None:
        try:
            if self.repo is not None:
                self.repo.commit()
                self.repo.close()
        except Exception:
            logger.debug("Repo close failed | caster=%s", self.cfg.caster_id, exc_info=True)
        try:
            if self.weight_service is not None:
                self.weight_service.stop()
        except Exception:
            logger.debug("Weight service stop failed | caster=%s", self.cfg.caster_id, exc_info=True)
        try:
            if self.plc is not None:
                self.plc.close()
        except Exception:
            logger.debug("PLC close failed | caster=%s", self.cfg.caster_id, exc_info=True)
        logger.info("Per-camera pipeline closed | caster=%s", self.cfg.caster_id)


@dataclass
class SharedInferenceWorker:
    model_path: str
    detector: SharedYoloDetector
    states: list[PerCameraPipelineState]
    buffers: dict[int, LatestFrameBuffer]
    stop_event: threading.Event
    gpu_lock: threading.Lock
    thread: threading.Thread = field(init=False)

    def __post_init__(self) -> None:
        self.thread = threading.Thread(
            target=self.run,
            name=f"SharedInferenceWorker-{Path(self.model_path).name}",
            daemon=True,
        )

    def start(self) -> None:
        self.thread.start()

    def join(self, timeout: float | None = None) -> None:
        self.thread.join(timeout=timeout)

    def run(self) -> None:
        state_by_caster = {s.cfg.caster_id: s for s in self.states}
        caster_ids = [s.cfg.caster_id for s in self.states]
        cursor = 0
        logger.info(
            "Shared inference worker starting | model=%s | model_instances=1 | casters=%s",
            self.model_path,
            ",".join(str(x) for x in caster_ids),
        )

        try:
            while not self.stop_event.is_set():
                if not caster_ids:
                    return

                processed = False
                for _ in range(len(caster_ids)):
                    caster_id = caster_ids[cursor % len(caster_ids)]
                    cursor += 1
                    packet = self.buffers[caster_id].take_latest()
                    if packet is None:
                        continue

                    processed = True
                    state = state_by_caster[caster_id]
                    frame_scaled = prepare_analysis_frame(
                        packet.frame_orig,
                        target_width=state.cfg.runtime.imgsz,
                        mode=state.cfg.runtime.analysis_image_mode,
                    )
                    infer_start = time.perf_counter()
                    with self.gpu_lock:
                        detections = self.detector.infer(
                            frame_scaled,
                            conf=state.cfg.runtime.conf,
                            iou=state.cfg.runtime.iou,
                            imgsz=state.cfg.runtime.imgsz,
                        )
                    inference_ms = (time.perf_counter() - infer_start) * 1000.0
                    logger.debug(
                        "Shared inference result | caster=%s | model=%s | detections=%d | inference_ms=%.1f | source_frame_idx=%d",
                        caster_id,
                        self.model_path,
                        len(detections),
                        inference_ms,
                        packet.frame_idx,
                    )
                    state.process(packet, frame_scaled, detections, inference_ms)
                    break

                if not processed:
                    time.sleep(0.005)
        except Exception:
            logger.exception("Shared inference worker failed | model=%s", self.model_path)
            self.stop_event.set()
        finally:
            logger.info("Shared inference worker stopped | model=%s", self.model_path)


class SharedMultiCameraRuntime:
    def __init__(self, cfgs: Iterable[AppCfg]) -> None:
        self.cfgs = list(cfgs)
        self.stop_event = threading.Event()
        self.gpu_lock = threading.Lock()
        self.buffers = {cfg.caster_id: LatestFrameBuffer(cfg.caster_id) for cfg in self.cfgs}
        self.capture_workers = [
            CameraCaptureWorker(cfg=cfg, buffer=self.buffers[cfg.caster_id], stop_event=self.stop_event)
            for cfg in self.cfgs
        ]
        self.pipeline_states = [
            PerCameraPipelineState(cfg=cfg, model_path=cfg.runtime.model_path) for cfg in self.cfgs
        ]
        self.inference_workers: list[SharedInferenceWorker] = []
        self.detectors: dict[str, SharedYoloDetector] = {}
        self._stop_lock = threading.Lock()
        self._stopped = False

    def describe(self) -> dict[str, list[int]]:
        groups: dict[str, list[int]] = {}
        for cfg in self.cfgs:
            groups.setdefault(_model_group_key(cfg.runtime.model_path), []).append(cfg.caster_id)
        return groups

    def start(self) -> None:
        groups = self.describe()
        logger.info(
            "Shared multi-camera runtime starting | pid=%d | casters=%s | unique_models=%d",
            os.getpid(),
            ",".join(str(cfg.caster_id) for cfg in self.cfgs),
            len(groups),
        )

        state_by_caster = {state.cfg.caster_id: state for state in self.pipeline_states}
        cfg_by_caster = {cfg.caster_id: cfg for cfg in self.cfgs}
        for model_key, caster_ids in groups.items():
            first_cfg = cfg_by_caster[caster_ids[0]]
            self._warn_if_model_group_runtime_differs(model_key, caster_ids, cfg_by_caster)
            detector = SharedYoloDetector(
                first_cfg.runtime.model_path,
                device=first_cfg.runtime.device,
                half=first_cfg.runtime.half,
            )
            self.detectors[model_key] = detector
            states = [state_by_caster[caster_id] for caster_id in caster_ids]
            for state in states:
                state.open(detector.names)
            self.inference_workers.append(
                SharedInferenceWorker(
                    model_path=first_cfg.runtime.model_path,
                    detector=detector,
                    states=states,
                    buffers=self.buffers,
                    stop_event=self.stop_event,
                    gpu_lock=self.gpu_lock,
                )
            )

        for worker in self.inference_workers:
            worker.start()
        for worker in self.capture_workers:
            worker.start()

    def run_forever(self) -> None:
        self.start()
        try:
            while not self.stop_event.is_set():
                time.sleep(0.5)
        except KeyboardInterrupt:
            logger.info("Shared multi-camera runtime received KeyboardInterrupt")
        finally:
            self.stop()

    def stop(self) -> None:
        with self._stop_lock:
            if self._stopped:
                return
            self._stopped = True
            self.stop_event.set()
            for worker in self.capture_workers:
                worker.join(timeout=5.0)
            for worker in self.inference_workers:
                worker.join(timeout=5.0)
            for state in self.pipeline_states:
                state.close()
            cv2.destroyAllWindows()
            logger.info("Shared multi-camera runtime stopped")

    def _warn_if_model_group_runtime_differs(
        self,
        model_key: str,
        caster_ids: list[int],
        cfg_by_caster: dict[int, AppCfg],
    ) -> None:
        first = cfg_by_caster[caster_ids[0]].runtime
        for caster_id in caster_ids[1:]:
            runtime = cfg_by_caster[caster_id].runtime
            if runtime.device != first.device or runtime.half != first.half:
                logger.warning(
                    "Shared model group uses first caster device/half settings | model=%s | first_caster=%s | caster=%s | first_device=%s | caster_device=%s | first_half=%s | caster_half=%s",
                    model_key,
                    caster_ids[0],
                    caster_id,
                    first.device,
                    runtime.device,
                    first.half,
                    runtime.half,
                )


def _model_group_key(model_path: str) -> str:
    return str(Path(model_path).expanduser())
