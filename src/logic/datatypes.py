# PipeTrack, GateTrack dataclasses
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional

@dataclass
class PipeStats:
  # Stable ID for this pipe (not tracker ID)
  pipe_uid: str
  # Tracker ID
  tracker_id: int

  origin: Optional[str] = None               # Origin of the pipe - "caster" | "other"
  state: str = "moving"                      # "moving" | "on_loadcell" | "parked"

  t_origin: Optional[float] = None           # Epoch seconds when first seen in origin ROI
  t_loadcell_enter: Optional[float] = None   # Epoch seconds when placed on loadcell
  t_loadcell_exit: Optional[float] = None    # Epoch seconds when removed from loadcell

  frames_seen: int = 0                       # To check persistence
  frames_missing: int = 0                    # To check staleness

  # Running confidence aggregates
  conf_sum_full: float = 0.0
  conf_count_full: int = 0

  conf_sum_till_gate: float = 0.0
  conf_count_till_gate: int = 0
  reached_gate_zone: bool = False

  last_seen_frame: int = 0
  last_seen_ts: float = 0.0

  counted: bool = False                      # Whether this pipe has been counted already
  origin_hits: int = 0                       # Number of origin zone hits (has to persist these many frames atleast)
  loadcell_hits: int = 0                     # Number of loadcell zone hits
  loadcell_exit_misses: int = 0              # Frames outside loadcell after being on it


@dataclass
class GateStatus:
  name: str                                  # "gate1" | "gate2"
  position: str = "unknown"                  # "open" | "closed" | "unknown"
  stable: int = 0                            # Number of consecutive frames with same status
  last_ts: float = 0.0                       # Last time this gate status was updated

  # Backward compatibile alias
  @property
  def status(self) -> str:
    return self.position
  
  @status.setter
  def status(self, value: str) -> None:
    self.position = value