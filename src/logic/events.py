from __future__ import annotations
from dataclasses import dataclass

@dataclass(frozen=True)
class PipeEnteredLoadcellEvent:
  pipe_uid: str
  tracker_id: int
  t_enter: float  # Epoch seconds

@dataclass(frozen=True)
class PipeExitedLoadcellEvent:
  pipe_uid: str
  tracker_id: int
  t_exit: float  # Epoch seconds

@dataclass(frozen=True)
class PipeMergedEvent:
  kept_uid: str       # The pipe_uid that survives
  removed_uid: str    # The provisional pipe_uid that was discarded
  origin: str
  gap_seconds: float  # Time gap between stale and re-detection

@dataclass(frozen=True)
class GateOpenedEvent:
  gate_name: str  # "gate1" | "gate2"
  t_open: float  # Epoch seconds