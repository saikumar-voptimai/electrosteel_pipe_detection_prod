# App lifecycle, graceful shutdown, watchdog

from __future__ import annotations

import os
import time
import logging
import cv2

from dataclasses import dataclass

from plc.client import PLCClient
from vision.types import BBox, TrackDet
from utils.config import AppCfg
from utils.timing import RateLimiter

from camera.capture import Capture
from geometry.roi import ROIManager

from vision.tracker import YoloByteTrack
from vision.overlay import LatestFramePublisher, draw_overlay

from db.repo import SqliteRepo
from plc.factory import create_plc

from logic.pipe_fsm import PipeFlowFSM
from logic.gate_fsm import GateFSM
from logic.gate_sources import GateStatusSource, GeometryGateSource, PLCGateSource, VisionGateSource
from utils.logging import setup_logging
from utils.runtime import resize_for_inference

from pathlib import Path
from utils.video_recorder import SegmentedVideoRecorder

logger = logging.getLogger("pipe_detect")

#TODO: Use enums or constants for ROI names
REQUIRED_ROIS = [
    "roi_loadcell",
    "roi_caster5_origin",
    "roi_left_origin",
    "roi_right_origin",
    "roi_safety_critical",
    "roi_gate1_open", "roi_gate2_open",
    "roi_gate1_closed", "roi_gate2_closed",
]


@dataclass
class App:
  cfg: AppCfg

  def run(self) -> None:
    """
    """
    setup_logging(level=self.cfg.runtime.log_level, log_path=self.cfg.runtime.log_path)
    logger.info(
      "Starting app | source=%s | model=%s | db=%s | latest_jpg=%s | max_fps=%s | frame_skip=%s | publish_fps=%s | publish_imgsz=%s | headless=%s",
      self.cfg.runtime.video_source,
      self.cfg.runtime.model_path,
      self.cfg.runtime.db_path,
      self.cfg.runtime.latest_jpg_path,
      self.cfg.runtime.max_fps,
      self.cfg.runtime.frame_skip,
      self.cfg.runtime.publish_fps,
      self.cfg.runtime.publish_imgsz,
      self.cfg.runtime.run_headless,
    )

    # Validate ROIs
    for name in REQUIRED_ROIS:
      if name not in self.cfg.rois:
        raise RuntimeError(f"Missing required ROI: {name} in config/rois.yaml. Run --redraw to define ROIs.")
    
    os.makedirs(os.path.dirname(self.cfg.runtime.db_path), exist_ok=True)
    os.makedirs(os.path.dirname(self.cfg.runtime.latest_jpg_path), exist_ok=True)

    repo = SqliteRepo(self.cfg.runtime.db_path)
    plc = create_plc(self.cfg.plc)

    rois = ROIManager(self.cfg.rois)
    capture = Capture(source=self.cfg.runtime.video_source, camera_cfg=self.cfg.camera_cfg)
    capture.open()
    # ---- Video recorder ----
    project_root = Path.cwd()
    recorder = None
    rec_cfg = self.cfg.runtime.recording # RecordingCfg | None (from config loader)
    if rec_cfg:
      recorder = SegmentedVideoRecorder(rec_cfg, project_root=project_root)
      recorder.start()
    else:
      logger.info("Video recording disabled")

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
    )

    # Gate source switching via DB setting
    default_gate_source = self.cfg.runtime.gate.source_default
    repo.set_setting("gate_source", repo.get_setting("gate_source", default_gate_source))
    gate_source = repo.get_setting("gate_source", default_gate_source)

    gate_fsm = self._build_gate_fsm(gate_source, rois, plc)

    publisher = LatestFramePublisher(
      out_path=self.cfg.runtime.latest_jpg_path,
      fps=self.cfg.runtime.publish_fps,
    )

    limiter = RateLimiter(self.cfg.runtime.max_fps)

    last_commit = time.time()
    last_setting_poll = time.time() 

    frame_idx = 0

    window_name = "Pipe Detection = Live"
    if not self.cfg.runtime.run_headless:
      # Allow resizing the window on larger displays.
      cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    try:
      while True:
        item = capture.read()
        if item is None:
          logger.warning("No frame captured, retrying...")
          continue
        frame_orig, ts = item

        orig_h, orig_w = frame_orig.shape[:2]

        frame_scaled = resize_for_inference(frame_orig, target_width=self.cfg.runtime.imgsz)
        scaled_h, scaled_w = frame_scaled.shape[:2]

        # Coordinate mapping:
        # - ROIs are stored in ORIGINAL frame pixel coordinates (from ROI redraw wizard)
        # - Inference runs on frame_scaled
        # - tracker.infer() returns bboxes in SCALED coordinates
        # Therefore, to map detections back to original coords, multiply by (orig/scaled).
        # For drawing ROIs on the scaled frame, multiply ROI points by (scaled/orig).
        scale_x = scaled_w / orig_w
        scale_y = scaled_h / orig_h
        inv_scale_x = orig_w / scaled_w
        inv_scale_y = orig_h / scaled_h

        logger.debug("Frame captured | idx=%d | ts=%.3f | shape=%s", frame_idx, ts, getattr(frame_scaled, "shape", None))

        # Skip frames if configured
        if self.cfg.runtime.frame_skip > 0 and (frame_idx % (self.cfg.runtime.frame_skip + 1) != 0):
          frame_idx += 1
          continue

        dets = tracker.infer(frame_scaled)
        dets_orig = []
        for d in dets:
          if d.track_id is None:
              continue

          x1 = d.bbox.x1 * inv_scale_x
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
        # ---- Record overlay at ORIGINAL resolution (recommended) ----
        vis_record = draw_overlay(frame_orig.copy(), rois, dets_orig, ts, scale_x=1.0, scale_y=1.0)
        if recorder:
          recorder.push(vis_record, ts)


        logger.debug("Inference results | idx=%d | dets=%d", frame_idx, len(dets))

        # Update gate FSM
        gate_events = gate_fsm.update(frame=frame_orig, dets=dets_orig)
        for event in gate_events:
          logger.info(f"Gate opened: {event.gate_name} at {event.t_open}")
          repo.insert_event("gate_open", None, f"{event.gate_name}@{event.t_open:.3f}")
        
        # Update pipe FSM
        updated_pipes, pipe_events = pipe_fsm.update(frame_idx=frame_idx, ts=ts, dets=dets_orig)
        logger.debug("Pipe FSM updated | idx=%d | updated=%d | events=%d", frame_idx, len(updated_pipes), len(pipe_events))

        # Handle pipe events + optional extra PLC tag for debugging
        for event in pipe_events:
          logger.info(f"Pipe event: {event}")
          if event.__class__.__name__ == "PipeEnteredLoadcellEvent":
            repo.insert_event("pipe_enter_loadcell", event.pipe_uid, f"tid={event.tracker_id}")
            if "pipe_on_loadcell" in self.cfg.plc.tags:
              plc.pulse(self.cfg.plc.tags["pipe_on_loadcell"], self.cfg.plc.pulse_ms)
          
          if event.__class__.__name__ == "PipeExitedLoadcellEvent":
            repo.insert_event("pipe_exit_loadcell", event.pipe_uid, f"tid={event.tracker_id}")
        
        # Upsert updated pipes
        for p in updated_pipes:
          avg_full = (p.conf_sum_full / p.conf_count_full) if p.conf_count_full > 0 else 0.0
          avg_till_gate = (p.conf_sum_till_gate / p.conf_count_till_gate) if p.conf_count_till_gate > 0 else 0.0
          repo.upsert_pipe({
            "pipe_uid": p.pipe_uid,
            "tracker_id": p.tracker_id,
            "origin": p.origin,
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
        
        # Draw and publish latest frame (visualization sizing is separate from inference sizing)
        vis_base = frame_orig
        if int(self.cfg.runtime.publish_imgsz) > 0:
          vis_base = resize_for_inference(frame_orig, target_width=int(self.cfg.runtime.publish_imgsz))

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

        vis = draw_overlay(vis_base.copy(), rois, dets_vis, ts, scale_x=vis_scale_x, scale_y=vis_scale_y)
        if not self.cfg.runtime.run_headless:
          cv2.imshow(window_name, vis)
          key = cv2.waitKey(1) & 0xFF
          if key == 27:   # ESC key
            logger.info("Quit signal received, shutting down...")
            break
        publisher.publish(vis)
        
        # Commit DB periodically
        if time.time() - last_commit >= self.cfg.runtime.db_flush_interval_s:
          logger.debug("DB commit | interval_s=%.3f", self.cfg.runtime.db_flush_interval_s)
          repo.commit()
          last_commit = time.time()

        # Poll settings for gate source change
        if time.time() - last_setting_poll >= 2.0:
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
    except KeyboardInterrupt:
      logger.info("Shutting down application...")
    finally:
      try:
        repo.commit()
        repo.close()
      except Exception:
        pass
      try:
        plc.close()
      except Exception:
        pass
      try:
        capture.close()
        capture._cap.release()
      except Exception:
        pass
      if recorder:
        try:
          recorder.stop()
        except Exception:
          logger.exception("Recorder stop failed")
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
      plc_signal_on_open=True,
    )