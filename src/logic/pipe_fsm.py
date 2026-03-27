from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List
import time
import logging

from utils.roi_names import RoiName
from geometry.roi import ROIManager
from logic.datatypes import PipeStats
from logic.events import PipeEnteredLoadcellEvent, PipeExitedLoadcellEvent
from plc.client import PLCClient
from vision.types import TrackDet
from ui.formatting import fmt_ts

logger = logging.getLogger(__name__)


@dataclass
class PipeFlowFSM:

    rois: ROIManager
    plc: PLCClient
    pulse_tag: str
    pulse_ms: int

    origin_confirm_frames: int = 2
    loadcell_enter_confirm_frames: int = 1
    loadcell_exit_confirm_frames: int = 2
    stale_track_frames: int = 45
    rearm_empty_frames: int = 10
    min_pipe_gap_seconds: int = 15
    loadcell_covered_per: int = 90
    pipe_lost_frames: int = 2
    pipe_reconnect_window_sec: int = 3

    pipes: Dict[int, PipeStats] = None
    seq: int = 0
    loadcell_armed: bool = True
    loadcell_empty_streak: int = 0

    #  store last confirmed caster pipe
    last_caster_pipe: PipeStats | None = None

    def __post_init__(self):
        if self.pipes is None:
            self.pipes = {}
        self.last_caster_pipe = None
        self.recent_lost_pipes = []

    # loadcell coverage area (e.g., 90%)
    def loadcell_covered_percentage(self, bbox):
        r = self.rois.rois[RoiName.LOADCELL.value]
        xs, ys = zip(*r)
        ix = max(0, min(max(xs), bbox.x2) - max(min(xs), bbox.x1))
        iy = max(0, min(max(ys), bbox.y2) - max(min(ys), bbox.y1))

        result = (ix * iy) / ((max(xs) - min(xs)) * (max(ys) -
                                                     min(ys))) >= self.loadcell_covered_per / 100.0
        # print(result)
        return (ix * iy) / ((max(xs) - min(xs)) * (max(ys) - min(ys))) >= self.loadcell_covered_per / 100.0

    # UID generation logic with reuse based on time gap
    def _new_pipe_uid(self) -> str:
        self.seq += 1
        return f"caster_{int(time.time())}_{self.seq:06d}"

    def update(self, frame_idx: int, ts: float, dets: List[TrackDet]):

        updated: List[PipeStats] = []
        events: List[object] = []

        # Check loadcell empty for rearm
        any_pipe_in_loadcell = False
        for d in dets:
            if d.cls_name != "pipe" or d.track_id is None:
                continue

            cx, cy = d.bbox.centroid()

            # skips pipe comes from left or righ origin
            if self.rois.contains(RoiName.LEFT_ORIGIN.value, cx, cy) or \
                    self.rois.contains(RoiName.RIGHT_ORIGIN.value, cx, cy):
                continue

            if self.loadcell_covered_percentage(d.bbox):
                any_pipe_in_loadcell = True
                break
        if any_pipe_in_loadcell:
            self.loadcell_empty_streak = 0
        else:
            self.loadcell_empty_streak += 1
            if not self.loadcell_armed and \
               self.loadcell_empty_streak >= self.rearm_empty_frames:
                self.loadcell_armed = True
                logger.info("Loadcell re-armed")

        # Process detections
        #  EARLY LOST PIPE DETECTION
        for tid, p in list(self.pipes.items()):
            if frame_idx - p.last_seen_frame > self.pipe_lost_frames:
                if p not in self.recent_lost_pipes:
                    logger.info("Pipe marked as lost | uid=%s", p.pipe_uid)
                    self.recent_lost_pipes.append(p)

        for d in dets:

            if d.cls_name != "pipe" or d.track_id is None:
                continue

            tid = int(d.track_id)
            cx, cy = d.bbox.centroid()
            if self.rois.contains(RoiName.RIGHT_ORIGIN.value, cx, cy) or self.rois.contains(RoiName.LEFT_ORIGIN.value, cx, cy):
                continue

            p = self.pipes.get(tid)
            # logger.info(p)
            #  MATCH WITH RECENT LOST PIPE UID
            if p is None:
                for old in getattr(self, "recent_lost_pipes", []):
                    # logger.info(self.recent_lost_pipes)
                    if ts - old.last_seen_ts <= self.pipe_reconnect_window_sec:
                        # TODO : add distance information 
                        p = old
                        p.tracker_id = tid  # assign new track id
                        self.recent_lost_pipes.remove(old)
                        logger.info("Re-linked lost pipe | uid=%s", p.pipe_uid)
                        break

            # Create temporary track (no UID yet)
            if p is None:
                p = PipeStats(pipe_uid=None, tracker_id=tid)
                p.last_seen_frame = frame_idx
                p.last_seen_ts = ts

            # Update tracking counters
            if p.frames_seen > 0:
                gap = (frame_idx - p.last_seen_frame) - 1
                if gap > 0:
                    p.frames_missing += gap

            p.frames_seen += 1
            p.last_seen_frame = frame_idx
            p.last_seen_ts = ts
            p.tracker_id = tid

            # ORIGIN CONFIRMATION
            if p.origin is None:

                if self.rois.contains(RoiName.CASTER_ORIGIN.value, cx, cy):

                    p.origin_hits += 1

                    if p.origin_hits >= self.origin_confirm_frames:

                        p.origin = "caster"

                        #  Merge logic based on t_origin gap
                        reuse_uid = None

                        if (
                            self.last_caster_pipe is not None
                            and self.last_caster_pipe.t_origin is not None
                        ):
                            gap_sec = ts - self.last_caster_pipe.last_seen_ts

                            if 0 <= gap_sec <= self.min_pipe_gap_seconds:
                                reuse_uid = self.last_caster_pipe.pipe_uid
                                logger.info(
                                    "Reusing previous pipe UID | uid=%s | gap=%.2fs",
                                    reuse_uid,
                                    gap_sec
                                )

                        if reuse_uid:
                            # reuse previous pipe fully
                            p.pipe_uid = self.last_caster_pipe.pipe_uid
                            p.t_origin = self.last_caster_pipe.t_origin   # keep old time
                        else:
                            p.pipe_uid = self._new_pipe_uid()
                            p.t_origin = ts   # only set new time for truly new pipe

                        self.last_caster_pipe = p

                        logger.info(
                            "Pipe origin confirmed | uid=%s | ts=%s",
                            p.pipe_uid,
                            fmt_ts(ts)
                        )

                elif (
                    self.rois.contains(RoiName.LEFT_ORIGIN.value, cx, cy)
                    or self.rois.contains(RoiName.RIGHT_ORIGIN.value, cx, cy)
                ):
                    p.origin_hits += 1
                    if p.origin_hits >= self.origin_confirm_frames:
                        p.origin = "other"

            # LOADCELL ENTER

            eligible = (p.origin == "caster")

            if eligible and p.t_loadcell_enter is None:

                if self.loadcell_covered_percentage(d.bbox):

                    p.loadcell_hits += 1

                    if self.loadcell_armed and \
                       p.loadcell_hits >= self.loadcell_enter_confirm_frames:

                        p.t_loadcell_enter = ts
                        p.state = "on_loadcell"

                        if not p.counted:
                            self.plc.pulse(self.pulse_tag, self.pulse_ms)
                            p.counted = True
                            self.loadcell_armed = False

                            events.append(
                                PipeEnteredLoadcellEvent(
                                    pipe_uid=p.pipe_uid,
                                    tracker_id=tid,
                                    t_enter=ts
                                )
                            )

                            logger.info(
                                "Pipe entered loadcell | uid=%s",
                                p.pipe_uid
                            )
                else:
                    p.loadcell_hits = 0

            # LOADCELL EXIT
            if eligible and p.t_loadcell_enter and not p.t_loadcell_exit:

                if not self.loadcell_covered_percentage(d.bbox):

                    p.loadcell_exit_misses += 1

                    if p.loadcell_exit_misses >= self.loadcell_exit_confirm_frames:

                        p.t_loadcell_exit = ts
                        p.state = "parked"

                        events.append(
                            PipeExitedLoadcellEvent(
                                pipe_uid=p.pipe_uid,
                                tracker_id=tid,
                                t_exit=ts
                            )
                        )

                        logger.info(
                            "Pipe exited loadcell | uid=%s",
                            p.pipe_uid
                        )
                else:
                    p.loadcell_exit_misses = 0

            self.pipes[tid] = p
            if p.pipe_uid is not None:
                updated.append(p)

        # Cleanup stale tracks
        stale_ids = [
            tid for tid, p in self.pipes.items()
            if frame_idx - p.last_seen_frame > self.stale_track_frames
        ]

        for tid in stale_ids:
            p = self.pipes[tid]

            logger.debug(
                "Stale track cleanup | tid=%d | uid=%s | last_seen_frame=%d",
                tid,
                p.pipe_uid,
                p.last_seen_frame
            )
            # If pipe entered loadcell but never exited, consider it exited
            if p.t_loadcell_enter is not None and p.t_loadcell_exit is None:

                p.t_loadcell_exit = ts
                p.state = "parked"

                events.append(
                    PipeExitedLoadcellEvent(
                        pipe_uid=p.pipe_uid,
                        tracker_id=tid,
                        t_exit=ts
                    )
                )

                updated.append(p)

                logger.info(
                    "Pipe considered exited (stale) | uid=%s | tid=%d | ts=%s",
                    p.pipe_uid,
                    tid,
                    fmt_ts(ts)
                )
            # store before deleting
            if not hasattr(self, "recent_lost_pipes"):
                self.recent_lost_pipes = []

            self.recent_lost_pipes.append(p)

            del self.pipes[tid]

        return updated, events
