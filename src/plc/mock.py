from __future__ import annotations
import time
import logging
from dataclasses import dataclass, field
from plc.client import PLCClient
from typing import Dict

logger = logging.getLogger(__name__)

@dataclass
class MockPLCClient(PLCClient):
  state: Dict[str, bool] = field(default_factory=dict)

  def pulse(self, tag: str, ms: int) -> None:
    """
    Mock pulse implementation: logs the pulse action.
    """
    logger.info("[MOCK PLC] pulse %s %dms", tag, ms)
    self.state[tag] = True
    time.sleep(ms/1000.0)
    self.state[tag] = False

  def read_bool(self, tag: str) -> bool:
    """
    Mock read_bool implementation: returns the stored state or False if not set.
    """
    return bool(self.state.get(tag, False))
  
  def close(self) -> None:
    logger.info("[MOCK PLC] close connection")