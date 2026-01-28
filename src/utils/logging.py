from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional


_LEVELS: dict[str, int] = {
  "CRITICAL": logging.CRITICAL,
  "ERROR": logging.ERROR,
  "WARNING": logging.WARNING,
  "INFO": logging.INFO,
  "DEBUG": logging.DEBUG,
}


def setup_logging(level: str = "INFO", log_path: Optional[str] = None) -> None:
  """Configure process-wide logging.

  - Console logging always enabled
  - Optional file logging (rotating is intentionally not added)
  """
  lvl = _LEVELS.get((level or "INFO").upper(), logging.INFO)

  handlers: list[logging.Handler] = []

  console = logging.StreamHandler()
  console.setLevel(lvl)
  handlers.append(console)

  if log_path:
    p = Path(log_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    file_handler = logging.FileHandler(str(p), encoding="utf-8")
    file_handler.setLevel(lvl)
    handlers.append(file_handler)

  logging.basicConfig(
    level=lvl,
    handlers=handlers,
    format="%(asctime)s.%(msecs)03d | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
  )

  # Reduce noisy third-party loggers unless user explicitly wants DEBUG everywhere
  if lvl > logging.DEBUG:
    logging.getLogger("ultralytics").setLevel(logging.WARNING)
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
