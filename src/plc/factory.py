from __future__ import annotations
from utils.config import PlcCfg
from plc.client import PLCClient
from plc.mock import MockPLCClient

def create_plc(cfg: PlcCfg) -> PLCClient:
  """
  Factory function to create a PLCClient or MockPLCClient based on config.
  """
  mode = cfg.mode.lower()

  if mode == "mock":
    return MockPLCClient()
  
  if mode == "modbus":
    mb = cfg.modbus or {}
    host = mb.get("host", "127.0.0.1")
    port = int(mb.get("port", 502))
    coils = dict((mb.get("coils", {}) or {}))
    from plc.modbus import ModbusPLCClient
    return ModbusPLCClient(host=host, port=port, coils=coils)

  raise ValueError(f"Unsupported PLC mode: {cfg.mode}")
