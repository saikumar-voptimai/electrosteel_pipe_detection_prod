# App lifecycle, graceful shutdown, watchdog

from __future__ import annotations

import os
import time
import logging
import cv2

from dataclasses import dataclass

from plc.client import PLCClient
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
      "Starting app | source=%s | model=%s | db=%s | latest_jpg=%s | max_fps=%s | frame_skip=%s | publish_fps=%s",
      self.cfg.runtime.video_source,
      self.cfg.runtime.model_path,
      self.cfg.runtime.db_path,
      self.cfg.runtime.latest_jpg_path,
      self.cfg.runtime.max_fps,
      self.cfg.runtime.frame_skip,
      self.cfg.runtime.publish_fps,
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
    capture = Capture(source=self.cfg.runtime.video_source)
    capture.open()

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

    try:
      while True:
        item = capture.read()
        if item is None:
          logger.warning("No frame captured, retrying...")
          continue
        frame, ts = item

        logger.debug("Frame captured | idx=%d | ts=%.3f | shape=%s", frame_idx, ts, getattr(frame, "shape", None))

        # Skip frames if configured
        if self.cfg.runtime.frame_skip > 0 and (frame_idx % (self.cfg.runtime.frame_skip + 1) != 0):
          frame_idx += 1
          continue

        dets = tracker.infer(frame)
        logger.debug("Inference results | idx=%d | dets=%d", frame_idx, len(dets))

        # Update gate FSM
        gate_events = gate_fsm.update(frame=frame, dets=dets)
        for event in gate_events:
          logger.info(f"Gate opened: {event.gate_name} at {event.t_open}")
          repo.insert_event("gate_open", None, f"{event.gate_name}@{event.t_open:.3f}")
        
        # Update pipe FSM
        updated_pipes, pipe_events = pipe_fsm.update(frame_idx=frame_idx, ts=ts, dets=dets)
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
        
        # Draw and publish latest frame
        vis = draw_overlay(frame.copy(), rois, dets)

        cv2.imshow("Pipe Detection = Live", vis)
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
      plc_signal_on_open=True,
    )