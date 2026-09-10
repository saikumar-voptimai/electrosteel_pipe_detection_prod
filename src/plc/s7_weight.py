from __future__ import annotations

import time
from dataclasses import dataclass


@dataclass
class S7WeightClient:
  """Minimal Siemens S7 DB read client for weight capture."""

  ip: str
  rack: int
  slot: int
  db_num: int

  def __post_init__(self) -> None:
    try:
      import snap7  # type: ignore
    except Exception as e:
      raise RuntimeError(
        "python-snap7 is required for S7 weight reading. Install with: pip install python-snap7"
      ) from e
    self._snap7 = snap7
    self.client = snap7.client.Client()

  def __enter__(self) -> "S7WeightClient":
    self.client.connect(self.ip, self.rack, self.slot)
    return self

  def __exit__(self, exc_type, exc, tb) -> None:
    self.close()

  def close(self) -> None:
    try:
      self.client.disconnect()
    except Exception:
      pass

  def pulse_bit(self, byte_index: int, bit_index: int, pulse_ms: int) -> None:
    """Pulse a bit in the PLC DB for `pulse_ms` milliseconds."""
    from snap7.util import set_bool  # type: ignore

    db_data = self.client.db_read(self.db_num, int(byte_index), 1)
    set_bool(db_data, 0, int(bit_index), True)
    self.client.db_write(self.db_num, int(byte_index), db_data)
    # time.sleep(float(pulse_ms) / 1000.0)
    set_bool(db_data, 0, int(bit_index), False)
    self.client.db_write(self.db_num, int(byte_index), db_data)

  def read_real(self, offset: int) -> float:
    """Read a REAL (float32) from the PLC DB at byte offset `offset`."""
    from snap7.util import get_real  # type: ignore

    db_data = self.client.db_read(self.db_num, int(offset), 4)
    return float(get_real(db_data, 0))