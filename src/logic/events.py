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
class GateOpenedEvent:
  gate_name: str  # "gate1" | "gate2"
  t_open: float  # Epoch seconds