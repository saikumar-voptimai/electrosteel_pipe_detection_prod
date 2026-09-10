from __future__ import annotations

import threading
import time
import queue
from dataclasses import dataclass
from typing import List, Tuple, Optional

from plc.s7_weight import S7WeightClient
from logic.weight_processing import compute_weight, WeightResult
from utils.config import WeightCfg


@dataclass(frozen=True)
class WeightFinalized:
  pipe_uid: str
  machine_id: int
  result: WeightResult
  started_at: float
  finished_at: float
  reason: str  # "exit" | "timeout" | "shutdown"


def _read_weight_loop(
  cfg: WeightCfg,
  machine_id: int,
  stop_evt: threading.Event,
  max_duration_s: float,
) -> Tuple[List[Tuple[float, float]], str]:
  m = cfg.machines.get(machine_id)
  if m is None:
    return [], "no_machine"

  raw: List[Tuple[float, float]] = []
  started = time.time()

  with S7WeightClient(ip=m.ip, rack=cfg.rack, slot=cfg.slot, db_num=cfg.db_number) as plc:
    if m.trigger_byte is not None and m.trigger_bit is not None:
      plc.pulse_bit(m.trigger_byte, m.trigger_bit, cfg.pulse_ms)

    while True:
      elapsed = time.time() - started
      if stop_evt.is_set():
        return raw, "stop"
      if elapsed >= max_duration_s:
        return raw, "timeout"
      # Keep reading even if cfg.read_duration_s is larger.
      if elapsed >= float(cfg.read_duration_s):
        return raw, "duration"

      v = plc.read_real(m.weight_real_offset)
      raw.append((elapsed, float(v)))
      time.sleep(float(cfg.read_interval_s))


class WeightService:
  """
  Background weight capture per pipe. 

  - `active()` checks if a capture is active
  - `start()` begins a reader thread (if not already running)
  - `stop()` requests stop; result is returned via `drain_results()`
  - `drain_results()` returns all finalized weight results
  - `_worker()` is the internal thread function that reads weight and computes result
  """

  def __init__(self, cfg: WeightCfg, *, max_duration_s: float = 30.0) -> None:
    self.cfg = cfg
    self.max_duration_s = float(max_duration_s)
    self._thread: Optional[threading.Thread] = None
    self._stop_evt = threading.Event()
    self._active_pipe_uid: Optional[str] = None
    self._active_machine_id: Optional[int] = None
    self._results: "queue.Queue[WeightFinalized]" = queue.Queue()
    self._lock = threading.Lock()

  def active(self) -> bool:
    """
    Returns True if a weight capture is currently active.
    """
    t = self._thread
    return t is not None and t.is_alive()

  def start(self, pipe_uid: str, machine_id: int) -> bool:
    """
    Start weight capture for given pipe_uid and machine_id.
    Returns True if started, False if already active or disabled.
    """
    if not self.cfg.enabled:
      return False

    with self._lock:
      if self.active():
        # Only one capture at a time (loadcell is effectively single-occupancy).
        return False

      self._stop_evt.clear()                        # Reset stop event
      self._active_pipe_uid = pipe_uid              # Mark active - Pipe  
      self._active_machine_id = int(machine_id)     # Mark active - Caster
      self._started_at = time.time()                # Mark start time 
      
      # Daemon thread for weight capture from plc using snap7. Saves results to self._results.
      self._thread = threading.Thread(target=self._worker, 
                                      args=(pipe_uid, machine_id), 
                                      name="WeightService", 
                                      daemon=True)
      self._thread.start()
      return True

  def _worker(self, pipe_uid: str, machine_id: int) -> None:
    """
    Worker thread to read weight and compute result.
    Puts WeightFinalized into self._results when done.
    """
    reason = "shutdown"       # Default reason
    finished_at = time.time() # Default finish time
    try:
      raw, stop_reason = _read_weight_loop(
        cfg=self.cfg,
        machine_id=int(machine_id),
        stop_evt=self._stop_evt,
        max_duration_s=self.max_duration_s,
      )
      reason = "timeout" if stop_reason == "timeout" else "exit"
      result = compute_weight(
        raw=raw,
        interval_s=float(self.cfg.read_interval_s),
        stable_window_s=float(self.cfg.stable_window_s),
        min_non_zero=float(self.cfg.min_nonzero),
        max_std_rel=float(self.cfg.max_std_rel),
        max_std_abs=float(self.cfg.max_std_abs),
      )
      finished_at = time.time()
      self._results.put(
        WeightFinalized(
          pipe_uid=pipe_uid,
          machine_id=int(machine_id),
          result=result,
          started_at=self._started_at,
          finished_at=finished_at,
          reason=reason,
        )
      )
    except Exception as e:
      finished_at = time.time()
      self._results.put(
        WeightFinalized(
          pipe_uid=pipe_uid,
          machine_id=int(machine_id),
          result=WeightResult(weight=None, quality=f"error:{type(e).__name__}", samples=0, raw=[]),
          started_at=self._started_at,
          finished_at=finished_at,
          reason="exit",
        )
      )
    finally:
      with self._lock:
        self._active_pipe_uid = None
        self._active_machine_id = None

  def stop(self) -> None:
    """
    Request stop of weight capture.
    """
    self._stop_evt.set()

  def drain_results(self) -> List[WeightFinalized]:
    """
    Drain all available finalized weight results.
    """
    out: List[WeightFinalized] = []
    while True:
      try:
        out.append(self._results.get_nowait())
      except queue.Empty:
        break
    return out