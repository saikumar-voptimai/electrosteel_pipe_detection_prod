# eligibility + loadcell trigger state machine
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Tuple
import time
import logging

from geometry.roi import ROIManager
from logic.datatypes import PipeStats
from logic.events import PipeEnteredLoadcellEvent, PipeExitedLoadcellEvent
from plc.client import PLCClient
from vision.types import BBox, TrackDet
from utils.roi_names import RoiName


logger = logging.getLogger(__name__)
  

@dataclass
class PipeFlowFSM:
  """
  Pipe flow finite state machine for eligibility and loadcell triggering.
  Rearm means the loadcell can trigger again after being empty for some frames.
  Stale tracks are removed after some frames of no detection.
  """
  rois: ROIManager
  plc: PLCClient
  pulse_tag: str
  pulse_ms: int
  
  #TODO: Tune these
  origin_confirm_frames: int = 2
  loadcell_enter_confirm_frames: int = 1
  loadcell_exit_confirm_frames: int = 2
  stale_track_frames: int = 45
  rearm_empty_frames: int = 10

  # Internal state
  pipes: Dict[int, PipeStats] = None
  seq: int = 0
  loadcell_armed: bool = True
  loadcell_empty_streak: int = 0

  def __post_init__(self) -> None:
    if self.pipes is None:
      self.pipes = {}
  
  def _new_pipe_uid(self) -> str:
    """
    Generate a new unique pipe UID.
    """
    self.seq += 1
    # Stable unique id, per run/day
    return f"caster5_{int(time.time())}_{self.seq:06d}"

  def update(self, frame_idx: int, ts: float, dets: List[TrackDet]) -> List[PipeStats]:
    """
    Returns list of updated PipeStats (for DB flush)
    """  
    updated: List[PipeStats] = []
    events: List[object] = []

    logger.debug("PipeFSM update | frame_idx=%d | ts=%.3f | dets=%d | tracks=%d", frame_idx, ts, len(dets), len(self.pipes))

    # Determine if loadcell ROI is empty (any pipe, not only eligible)
    any_pipe_in_loadcell = False
    for d in dets:
      if d.cls_name != "pipe" or d.track_id is None:
        continue
      cx, cy = d.bbox.centroid()
      #TODO: Pure centroid check may be insufficient, consider bbox overlap and iou
      if self.rois.contains(RoiName.LOADCELL.value, cx, cy):
        any_pipe_in_loadcell = True
        break

    if any_pipe_in_loadcell:
      self.loadcell_empty_streak = 0
    else:
      self.loadcell_empty_streak += 1
      if not self.loadcell_armed and self.loadcell_empty_streak >= self.rearm_empty_frames:
        self.loadcell_armed = True
        logger.info("Loadcell re-armed | empty_streak=%d", self.loadcell_empty_streak)
    

    # Process Pipe detections
    for d in dets:
      if d.cls_name != "pipe" or d.track_id is None:
        continue

      tid = int(d.track_id)
      cx, cy = d.bbox.centroid()

      p = self.pipes.get(tid)
      if p is None:
        # Create a record with a stable uid immediately (even before origin is confirmed)
        p = PipeStats(
          pipe_uid=self._new_pipe_uid(),
          tracker_id=tid
        )
        p.last_seen_frame = frame_idx
        p.last_seen_ts = ts
        logger.debug("New pipe track | tid=%d | uid=%s", tid, p.pipe_uid)

      # Update seen/missing counters
      if p.frames_seen > 0:
        gap = (frame_idx - p.last_seen_frame) - 1
        if gap > 0:
          p.frames_missing += gap
      
      p.frames_seen += 1
      p.last_seen_frame = frame_idx
      p.last_seen_ts = ts
      p.tracker_id = tid

      # Origin assignment to the pipe
      if p.origin is None:
        if self.rois.contains(RoiName.CASTER5_ORIGIN.value, cx, cy):
          p.origin_hits += 1
          if p.origin_hits >= self.origin_confirm_frames:
            p.origin = "caster"
            if p.t_origin is None:
              p.t_origin = ts
              logger.info(f"Pipe {p.pipe_uid} origin confirmed as caster at {ts:.3f}")
        else:
          # If it appears in exclusion ROIS first, mark as other
          if self.rois.contains(RoiName.LEFT_ORIGIN.value, cx, cy) or self.rois.contains(RoiName.RIGHT_ORIGIN.value, cx, cy):
            p.origin = "other"
            logger.info("Pipe origin set to other | uid=%s | tid=%d", p.pipe_uid, tid)
      
      # Confidence tracking
      p.conf_sum_full += d.conf
      p.conf_count_full += 1

      # "Till gate" = until first time it enters gate zone
      # TODO: Use also the gate open/close status to determine pipe approaching gate
      if not p.reached_gate_zone:
        p.conf_sum_till_gate += d.conf
        p.conf_count_till_gate += 1
        if self.rois.contains(RoiName.GATE1_OPEN.value, cx, cy):
          p.reached_gate_zone = True
      
      # Loadcell Enter/Exit logic (only for eligible caster pipes)
      eligible = (p.origin == "caster")
      if eligible and p.t_loadcell_enter is None:
        if self.rois.contains(RoiName.LOADCELL.value, cx, cy):
          # TODO: Also add condition that that pipe is to the right of gates and to the left of roi_loadcell
          p.loadcell_hits += 1
          if self.loadcell_armed and p.loadcell_hits >= self.loadcell_enter_confirm_frames: # Persisted enter
            p.t_loadcell_enter = ts
            p.state = "on_loadcell"
            if not p.counted:
              # Trigger PLC pulse
              self.plc.pulse(self.pulse_tag, self.pulse_ms)    # Trigger PLC
              p.counted = True
              self.loadcell_armed = False                      # Disarm until loadcell is empty for some frames
              events.append(PipeEnteredLoadcellEvent(
                pipe_uid=p.pipe_uid,
                tracker_id=tid,
                t_enter=ts
              ))
              logger.info("Pipe entered loadcell | uid=%s | tid=%d | ts=%.3f", p.pipe_uid, tid, ts)
        
        else:
          p.loadcell_hits = 0
      
      # Loadcell Exit: once on_loadcell, watch for leaving ROI
      if eligible and p.t_loadcell_enter is not None and p.t_loadcell_exit is None:
        #TODO: As soon as a pipe exits, we stop detecting it. So even when bbox is lost, we consider it exited.
        if not self.rois.contains(RoiName.LOADCELL.value, cx, cy):
          p.loadcell_exit_misses += 1
          if p.loadcell_exit_misses >= self.loadcell_exit_confirm_frames:
            p.t_loadcell_exit = ts
            p.state = "parked"
            events.append(PipeExitedLoadcellEvent(
              pipe_uid=p.pipe_uid,
              tracker_id=tid,
              t_exit=ts
            ))
            logger.info("Pipe exited loadcell | uid=%s | tid=%d | ts=%.3f", p.pipe_uid, tid, ts)
        # Pipe used to be tracked, entered loadcell, but now lost inside loadcell ROI
        else:
          p.loadcell_exit_misses = 0
      
      self.pipes[tid] = p
      updated.append(p)

    # Clean up stale tracks
    stale_ids = [tid for tid, p in self.pipes.items() if frame_idx - p.last_seen_frame > self.stale_track_frames]
    for tid in stale_ids:
      p = self.pipes[tid]
      logger.debug("Stale track cleanup | tid=%d | uid=%s | last_seen_frame=%d", tid, p.pipe_uid, p.last_seen_frame)
      # If it was on loadcell but disaappeared, consider it exited
      if p.t_loadcell_enter is not None and p.t_loadcell_exit is None:
        p.t_loadcell_exit = ts
        p.state = "parked"
        events.append(PipeExitedLoadcellEvent(
          pipe_uid=p.pipe_uid,
          tracker_id=tid,
          t_exit=ts
        ))
        updated.append(p)
        logger.info("Pipe considered exited (stale) | uid=%s | tid=%d | ts=%.3f", p.pipe_uid, tid, ts)
      del self.pipes[tid]
    
    return updated, events

