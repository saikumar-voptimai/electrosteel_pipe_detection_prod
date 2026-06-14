# App lifecycle, graceful shutdown, watchdog

from __future__ import annotations

import os
import time
import logging
import cv2
import numpy as np
from dataclasses import dataclass

from plc.client import PLCClient
from vision.types import BBox, TrackDet
from utils.config import AppCfg
from utils.timing import RateLimiter
from utils.roi_names import REQUIRED_ROIS, as_keys

from camera.capture import Capture
from geometry.roi import ROIManager

from vision.tracker import YoloByteTrack
from vision.overlay import LatestFramePublisher, draw_overlay

from db.repo import SqliteRepo
from plc.factory import create_plc

from logic.pipe_fsm import PipeFlowFSM
from logic.gate_fsm import GateFSM
from logic.gate_sources import GeometryGateSource, PLCGateSource, VisionGateSource
from logic.events import GateClosedEvent, GateOpenedEvent
from logic.weight_service import WeightService
from utils.logging import setup_logging
from utils.runtime import resize_for_inference
from ui.formatting import fmt_ts
from utils.camera_scheduler import CameraProfileScheduler

logger = logging.getLogger("pipe_detect")



@dataclass
class App:
  cfg: AppCfg

  def run(self) -> None:
    """
    """
    setup_logging(level=self.cfg.runtime.log_level, log_path=self.cfg.runtime.log_path)
    logger.info(
      "Starting app | source=%s | model=%s | db=%s | latest_jpg=%s | max_fps=%s | frame_skip=%s | publish_fps=%s | publish_imgsz=%s | headless=%s | pid=%d",
      self.cfg.runtime.video_source,
      self.cfg.runtime.model_path,
      self.cfg.runtime.db_path,
      self.cfg.runtime.latest_jpg_path,
      self.cfg.runtime.max_fps,
      self.cfg.runtime.frame_skip,
      self.cfg.runtime.publish_fps,
      self.cfg.runtime.publish_imgsz,
      self.cfg.runtime.run_headless,
      os.getpid(),
    )

    # Validate ROIs
    for name in as_keys(REQUIRED_ROIS):
        if name not in self.cfg.rois:
            raise RuntimeError(
                f"Missing required ROI: {name} in config/rois.yaml. Run --redraw to define ROIs."
            )
    os.makedirs(os.path.dirname(self.cfg.runtime.db_path), exist_ok=True)
    os.makedirs(os.path.dirname(self.cfg.runtime.latest_jpg_path), exist_ok=True)

    repo = SqliteRepo(self.cfg.runtime.db_path)
    plc = create_plc(self.cfg.plc)

    weight_service: WeightService | None = None
    if getattr(self.cfg, "weight", None) is not None and self.cfg.weight.enabled:
      weight_service = WeightService(self.cfg.weight, max_duration_s=30.0)
      logger.info("Weight capture enabled | machine_default=%s", self.cfg.weight.machine_id_default)

    rois = ROIManager(self.cfg.rois)
    capture = Capture(source=self.cfg.runtime.video_source, camera_cfg=self.cfg.camera_cfg)
    capture.open()
    # Start camera profile scheduler. 
    scheduler = None

    if self.cfg.camera_cfg and self.cfg.camera_cfg.profiles and capture._is_gige():
        scheduler = CameraProfileScheduler(
            capture,
            self.cfg.camera_cfg.profiles
        )
        scheduler.start()

    tracker = YoloByteTrack(
        model_path=self.cfg.runtime.model_path,
        tracker_yaml=self.cfg.runtime.tracker_yaml,
        conf=self.cfg.runtime.conf,
        iou=self.cfg.runtime.iou,
        imgsz=self.cfg.runtime.imgsz,
    )

    # Pipe Flow FSM
    pipe_fsm = PipeFlowFSM(
      rois=rois,
      plc=plc,
      pulse_tag=self.cfg.plc.tags["caster_5_new"],
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

    # Gate source switching via DB setting
    default_gate_source = self.cfg.runtime.gate.source_default
    repo.set_setting("gate_source", repo.get_setting("gate_source", default_gate_source))
    gate_source = repo.get_setting("gate_source", default_gate_source)

    gate_fsm = self._build_gate_fsm(gate_source, rois, plc)

    publisher = LatestFramePublisher(
      out_path=self.cfg.runtime.latest_jpg_path,
      fps=self.cfg.runtime.publish_fps,
      history_cfg=self.cfg.runtime.history,
      class_name_to_id=self.cfg.runtime.class_name_to_id,
    )

    limiter = RateLimiter(self.cfg.runtime.max_fps)

    # Independent throttles for non-inference logic and visualization.
    update_fps = int(getattr(self.cfg.runtime, "update_fps", 0) or 0)
    last_update_ts = 0.0
    last_viz_ts = 0.0

    last_commit = time.time()
    last_setting_poll = time.time() 

    frame_idx = 0

    window_name = "Pipe Detection = Live"
    if not self.cfg.runtime.run_headless:
      # Allow resizing the window on larger displays.
      cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    runfps = 1.0
    try:
      while True:
        iter_time = time.time()
        now = iter_time
        item = capture.read()
        if item is None:
          logger.warning("No frame captured, retrying...")
          continue
        st1 = time.time()
        frame_orig, ts = item
        # We get the original frame (and size) as recorded by the source camera
        orig_h, orig_w = frame_orig.shape[:2]       # Ex: VA - Imaging --> (w2620, h1216)

        frame_scaled = resize_for_inference(frame_orig, target_width=self.cfg.runtime.imgsz)
        # imgsz - w960. scaled frame size: (w960, h445)
        # NOTE: This is the frame passed to the ml model for inference.
        # TODO: FPS of the source is 18.0 fps. We can force it along with the inference and rendering fps.
        scaled_h, scaled_w = frame_scaled.shape[:2]

        # Coordinate mapping:
        # - ROIs are stored in ORIGINAL frame pixel coordinates (from ROI redraw wizard)
        # - Inference runs on frame_scaled
        # - tracker.infer() returns bboxes in SCALED coordinates
        # Therefore, to map detections back to original coords, multiply by (orig/scaled).
        # For drawing ROIs on the scaled frame, multiply ROI points by (scaled/orig).
        inv_scale_x = orig_w / scaled_w     # Ex: 2620 / 960 = 2.729
        inv_scale_y = orig_h / scaled_h     # Ex: 1216 / 445 = 2.732

        logger.debug("Frame captured | idx=%d | ts=%.3f | shape=%s", frame_idx, ts, getattr(frame_scaled, "shape", None))
        st2 = time.time()
        # Skip frames if configured
        if self.cfg.runtime.frame_skip > 0 and (frame_idx % (self.cfg.runtime.frame_skip + 1) != 0):
          frame_idx += 1
          continue

        dets = tracker.infer(frame_scaled) # Inference running on scaled frame (w960, h445)
        st3 = time.time()
        # It doesnt matter to yolo what the other dimension. Since it can detect the presence of objects
        # and reports in absolutre pixel coordinates of the scaled frame.
        dets_orig = []
        for d in dets:
          if d.track_id is None:
              continue
          # Original dets will be larger than inference coords. Hence multiplied by scale factors > 1.
          x1 = d.bbox.x1 * inv_scale_x # Now we will map the dets to original frame coords.
          y1 = d.bbox.y1 * inv_scale_y
          x2 = d.bbox.x2 * inv_scale_x
          y2 = d.bbox.y2 * inv_scale_y

          dets_orig.append(
              TrackDet(
                  cls_name=d.cls_name,
                  conf=d.conf,
                  track_id=d.track_id,
                  bbox=BBox(x1, y1, x2, y2),
              )
          )
        logger.debug("Inference results | idx=%d | dets=%d", frame_idx, len(dets))

        gate_metrics = {}
        updated_pipes = []
        pipe_events = []

        st = time.time()
        # Update gate FSM
        gate_events, gate_metrics = gate_fsm.update(frame=frame_orig, dets=dets_orig)
        for event in gate_events:
            if isinstance(event, GateOpenedEvent):
                repo.gate_open(event.gate_name, event.t_open)

            elif isinstance(event, GateClosedEvent):
                repo.gate_close(event.gate_name, event.t_closed)


        # Update pipe FSM (full logic)
        updated_pipes, pipe_events = pipe_fsm.update(frame_idx=frame_idx, ts=ts, dets=dets_orig)
        logger.debug("Pipe FSM updated | idx=%d | updated=%d | events=%d", frame_idx, len(updated_pipes), len(pipe_events))

        # Handle pipe events + optional extra PLC tag for debugging
        for event in pipe_events:
          logger.info(f"Pipe event: {event}")
          if event.__class__.__name__ == "PipeEnteredLoadcellEvent":
            repo.insert_event("pipe_enter_loadcell", event.pipe_uid, f"tid={event.tracker_id}")
            if "pipe_on_loadcell" in self.cfg.plc.tags:
              plc.pulse(self.cfg.plc.tags["pipe_on_loadcell"], self.cfg.plc.pulse_ms)

            if weight_service is not None:
              ok = weight_service.start(pipe_uid=event.pipe_uid, machine_id=self.cfg.weight.machine_id_default)
              if ok:
                repo.insert_event("weight_capture_start", event.pipe_uid, f"machine_id={self.cfg.weight.machine_id_default}")

          if event.__class__.__name__ == "PipeExitedLoadcellEvent":
            repo.insert_event("pipe_exit_loadcell", event.pipe_uid, f"tid={event.tracker_id}")

            if weight_service is not None:
              weight_service.stop()
          if event.__class__.__name__ == "PipeRemovedBeforeCheckpointEvent":
            repo.delete_pipe(event.pipe_uid)
            repo.insert_event("pipe_deleted", event.pipe_uid, event.reason)
        freq = 1 / (time.time() - st) if (time.time() - st) > 0 else 0.0
        logger.debug("Non-inference logic update complete | freq=%.2f Hz", freq)

        # Persist any finalized weights (done in background thread)
        if weight_service is not None:
          for fin in weight_service.drain_results():
            w = fin.result.weight
            quality = fin.result.quality
            samples = fin.result.samples
            repo.upsert_pipe({
              "pipe_uid": fin.pipe_uid,
              "weight": w,
              "weight_quality": quality,
              "weight_samples": samples,
            })
            repo.insert_event(
              "weight_captured",
              fin.pipe_uid,
              f"weight={w} quality={quality} samples={samples} reason={fin.reason}",
            )
        
        # Upsert updated pipes
        for p in updated_pipes:
          avg_full = (p.conf_sum_full / p.conf_count_full) if p.conf_count_full > 0 else 0.0
          avg_till_gate = (p.conf_sum_till_gate / p.conf_count_till_gate) if p.conf_count_till_gate > 0 else 0.0
          repo.upsert_pipe({
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
          })
        st4 = time.time()
        # Visualization and publishing are throttled by publish_fps.
        do_viz = (int(self.cfg.runtime.publish_fps) > 0) and ((now - last_viz_ts) >= (1.0 / float(self.cfg.runtime.publish_fps)))
        if do_viz:
          st = time.time()
          last_viz_ts = now

          # Draw and publish latest frame (visualization sizing is separate from inference sizing)
          vis_base = frame_orig # w2620, h1216
          pub_size = self.cfg.runtime.publish_imgsz

          if isinstance(pub_size, int) and pub_size > 0:
              vis_base = resize_for_inference(
                  frame_orig,
                  target_width=pub_size
              )

          elif isinstance(pub_size, tuple) and len(pub_size) == 2:
              target_w, target_h = pub_size
              vis_base = cv2.resize(frame_orig, (target_w, target_h))
          vis_h, vis_w = vis_base.shape[:2]
          vis_scale_x = vis_w / float(orig_w) # e.g. 1920 / 2620 = 0.732
          vis_scale_y = vis_h / float(orig_h) # e.g. 888 / 1216 = 0.730

          # Original dets will be smaller in vis coords. Hence multiplied by scale factors < 1.
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

          vis = draw_overlay(
            vis_base,
            rois,
            dets_vis,
            ts,
            scale_x=vis_scale_x,
            scale_y=vis_scale_y,
            gate_metrics=gate_metrics,
            debug=self.cfg.runtime.debug_mode,
            runfps=runfps,
          )

          if not self.cfg.runtime.run_headless:
            cv2.imshow(window_name, vis)
            key = cv2.waitKey(1) & 0xFF
            if key == 27:   # ESC key
              logger.info("Quit signal received, shutting down...")
              break
          
          fps = 1 / (time.time() - st) if (time.time() - st) > 0 else 0.0
          logger.debug("Visualization complete | freq=%.2f Hz", fps)
          publish_overlay = bool(self.cfg.runtime.publish_overlay)
          if publish_overlay:
              publisher.publish(vis)        # overlay image
          else:
              publisher.publish(vis_base, dets=dets_vis, ts=ts, gate_metrics=gate_metrics)   # raw image + txt metadata
        st5 = time.time()
        # Commit DB periodically
        if time.time() - last_commit >= self.cfg.runtime.db_flush_interval_s:
          logger.debug("DB commit | interval_s=%.3f", self.cfg.runtime.db_flush_interval_s)
          repo.commit()
          last_commit = time.time()

        # Poll settings for gate source change
        if time.time() - last_setting_poll >= 200.0:
          new_source = repo.get_setting("gate_source", default_gate_source)
          if new_source != gate_source:
            logger.info(f"Gate source changed from {gate_source} to {new_source}, updating FSM.")
            gate_source = new_source
            gate_fsm = self._build_gate_fsm(gate_source, rois, plc)
            repo.insert_event("setting_changed", None, f"gate_source={gate_source}")
            repo.commit()
          last_setting_poll = time.time()
        
        limiter.sleep_if_needed()
        frame_idx += 1
        iter_duration = time.time() - iter_time
        runfps = 1.0 / iter_duration if iter_duration > 0 else 0.0
        logger.debug("Frame processed | idx=%d | iter_duration=%.3f s | runfps=%.2f", frame_idx, iter_duration, runfps)

        st6 = time.time()
        logger.debug("time for full loop: %.3f s", st6 - st1)
        logger.debug("-----------------------------------------------------")
        logger.debug("Fraction times | read+scale=%.3f | inference=%.3f | logic=%.3f | render=%.3f | overhead=%.3f", st2 - st1, st3 - st2, st4 - st3, st5 - st4, st6 - st5)
    except KeyboardInterrupt:
      logger.info("Shutting down application...")
    finally:
      try:
        repo.commit()
        repo.close()
      except Exception:
        pass
      try:
        if scheduler:
          scheduler.stop()
      except Exception:
        pass
      try:
        if weight_service is not None:
          weight_service.stop()
      except Exception:
        pass
      try:
        plc.close()
      except Exception:
        pass
      try:
        capture.close()
      except Exception:
        pass
      cv2.destroyAllWindows()
      
  # Internal methods
  def _build_gate_fsm(self, gate_source: str, rois: ROIManager, plc: PLCClient) -> GateFSM:
    """
    Build GateFSM with appropriate source
    """
    gate_source = (gate_source or "geometry").lower()
    if gate_source == "plc":
      # NOTE: PLC tags need to changed as per config
      open_tags = {"gate1": self.cfg.plc.tags["gate1_open"], "gate2": self.cfg.plc.tags["gate2_open"]}
      source = PLCGateSource(plc=plc, open_tags=open_tags)
    
    elif gate_source == "vision":
      source = VisionGateSource(
        rois=rois,
        min_conf=self.cfg.runtime.gate.min_conf,
      )
    
    else:  # geometry
      source = GeometryGateSource(
        rois=rois,
        min_gate_conf=self.cfg.runtime.gate.min_conf,
        max_area_ratio_vs_closed=self.cfg.runtime.gate.max_area_ratio_vs_closed,
        max_w_over_h=self.cfg.runtime.gate.max_w_over_h,
        human_iou_occlusion=self.cfg.runtime.gate.human_iou_occlusion,
      )
    
    return GateFSM(
      source=source,
      plc=plc,
      pulse_ms=self.cfg.plc.pulse_ms,
      stable_frames=self.cfg.runtime.gate.stable_frames,
      gate_tags={
        "gate1": self.cfg.plc.tags.get("gate1_open", ""),
        "gate2": self.cfg.plc.tags.get("gate2_open", ""),
      },
      plc_signal_on_open=False,
    )
