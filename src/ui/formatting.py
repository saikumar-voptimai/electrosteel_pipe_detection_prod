from __future__ import annotations

from datetime import datetime


def fmt_ts(x) -> str:
  if x is None:
    return ""
  try:
    return datetime.fromtimestamp(float(x)).strftime("%Y-%m-%d %H:%M:%S")
  except Exception:
    return ""
