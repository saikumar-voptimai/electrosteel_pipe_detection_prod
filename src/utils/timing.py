# rate control, profiling
from __future__ import annotations
import time
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)

@dataclass
class RateLimiter:
    max_fps: int
    _last: float = 0.0

    def sleep_if_needed(self) -> None:
        """Sleep to maintain max_fps rate."""
        if self.max_fps <= 0:
            return
        now = time.time()
        period = 1.0 / float(self.max_fps)
        dt = now - self._last
        if dt < period:
            logger.debug("RateLimiter: sleeping for %f seconds", period - dt)
            time.sleep(period - dt)
        self._last = time.time()
