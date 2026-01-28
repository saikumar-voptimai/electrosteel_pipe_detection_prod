# interface: pulse(tag), set(tag,val), heartbeat()
from __future__ import annotations
from abc import ABC, abstractmethod

class PLCClient(ABC):
  """
  Abstract PLC client interface
  """
  @abstractmethod
  def pulse(self, tag: str, ms: int) -> None:
    """
    Set a pulse tag high for `ms` milliseconds
    """
    ...
  
  @abstractmethod
  def read_bool(self, tag: str) -> bool:
    """
    Read a boolean tag
    """
    ...
  
  @abstractmethod
  def close(self) -> None:
    """
    Close the PLC connection
    """
    ...