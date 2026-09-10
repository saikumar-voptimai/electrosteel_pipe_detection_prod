from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple
import math
import statistics

@dataclass(frozen=True)
class WeightResult:
  weight: float | None
  quality: str                    # "stable" | "fallback_max" | "no_signal"
  samples: int
  raw: List[Tuple[float, float]]  # (time, value) samples


def compute_weight(
    raw: List[Tuple[float, float]],
    interval_s: float,
    stable_window_s: float = 3.0,
    min_non_zero: float = 0.1,
    max_std_rel: float = 0.01,
    max_std_abs: float = 2.0,
) -> WeightResult:
  """
  Compute weight from raw samples over the given interval.
  Returns WeightResult with weight, quality, samples used, and raw samples.
  """
  if not raw:
    return WeightResult(weight=None, quality="no_signal", samples=0, raw=raw)

  vals = [v for _, v in raw if v >= min_non_zero]

  if not vals:
    return WeightResult(weight=None, quality="no_signal", samples=0, raw=raw)
  
  w = max(2, math.ceil(stable_window_s / interval_s))

  best = None # (idx, mean, std)

  # Find most stable window
  for i in range(0, len(vals) - w + 1):
    window = vals[i:i+w]
    m = statistics.mean(window)
    std = statistics.pstdev(window) if len(window) > 1 else 0.0

    if m <= min_non_zero:
      continue
    if std <= max(max_std_abs, max_std_rel * m):
      best = (i, m, std)
  
  if best:
    _, m, _ = best
    return WeightResult(weight=float(m), quality="stable", samples=len(vals), raw=raw)
  
  # fallback: return max value
  return WeightResult(weight=float(max(vals)), quality="fallback_max", samples=len(vals), raw=raw)