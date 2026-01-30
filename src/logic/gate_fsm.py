# gate open/close debounced transitions
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List
import time
import numpy as np
import logging

from logic.datatypes import GateStatus
from logic.events import GateOpenedEvent
from logic.gate_sources import GateStatusSource
from vision.types import TrackDet
from plc.client import PLCClient

logger = logging.getLogger(__name__)


@dataclass
class GateFSM:
  source: GateStatusSource
  plc: PLCClient
  pulse_ms: int
  stable_frames: int
  gate_tags: Dict[str, str]  # e.g. {"gate1": "gate1_open", "gate2": "gate2_pulse"}
  plc_signal_on_open: bool = True

  gates: Dict[str, GateStatus] = None

  def __post_init__(self) -> None:
    if self.gates is None:
      self.gates = {"gate1": GateStatus(name="gate1"), "gate2": GateStatus(name="gate2")}
  
  def update(self, frame: np.ndarray | None = None, dets: List[TrackDet] | None = None) -> List[str]:
    """
    Returns list of gate open events emitted.
    """
    events: List[GateOpenedEvent] = []
    now = time.time()

    for gate_name, gs in self.gates.items():
      pos, metrics = self.source.get_position(gate_name, frame=frame, dets=dets)

      logger.debug("Gate pos read | gate=%s | pos=%s | stable=%d", gate_name, pos, gs.stable)

      # If unknown, do not flip state
      if pos == "unknown":
        gs.last_ts = now
        self.gates[gate_name] = gs
        continue

      if pos == gs.position:
        gs.stable += 1
      else:
        logger.info("Gate state change | gate=%s | %s -> %s", gate_name, gs.position, pos)
        gs.position = pos
        gs.stable = 1

      gs.last_ts = now

      # Emit on stable transition to open
      if gs.position == "open" and gs.stable == self.stable_frames:
        tag = self.gate_tags.get(gate_name)
        if self.plc_signal_on_open and tag:
          self.plc.pulse(tag, self.pulse_ms)
        events.append(GateOpenedEvent(gate_name=gate_name, t_open=now))
        logger.info("Gate opened (debounced) | gate=%s | ts=%.3f", gate_name, now)
        
      self.gates[gate_name] = gs
    return events, metrics