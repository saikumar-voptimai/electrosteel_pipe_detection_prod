from __future__ import annotations
import time
from dataclasses import dataclass
from typing import Dict

from pymodbus.client import ModbusTcpClient
from plc.client import PLCClient

@dataclass
class ModbusPLCClient(PLCClient):
  host: str
  port: int
  coils: Dict[str, int]  # Mapping from tag names to coil addresses

  def __post_init__(self) -> None:
    self.client = ModbusTcpClient(self.host, port=self.port)
    if not self.client.connect():
      raise RuntimeError(f"Failed Modbus connect to {self.host}:{self.port}")
    
  def pulse(self, tag: str, ms: int) -> None:
    """
    Pulse a coil for specified milliseconds.
    """
    if tag not in self.coils:
      return
    addr = int(self.coils[tag])
    self.client.write_coil(addr, True)
    time.sleep(ms / 1000.0)
    self.client.write_coil(addr, False)
  
  def read_bool(self, tag: str) -> bool:
    """
    Read a boolean coil value.
    """
    if tag not in self.coils:
      return False
    addr = int(self.coils[tag])
    rr = self.client.read_coils(addr, count=1)
    return bool(rr.bits[0]) if rr and getattr(rr, "bits", None) else False
  
  def close(self) -> None:
    self.client.close()