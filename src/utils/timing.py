from __future__ import annotations

import time
import logging
from dataclasses import dataclass
from datetime import datetime, time as dtime
from typing import List, Dict
import pytz


logger = logging.getLogger(__name__)

# Rate Limiting
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
            sleep_time = period - dt
            logger.debug("RateLimiter: sleeping for %f seconds", sleep_time)
            time.sleep(sleep_time)

        self._last = time.time()

# Time & Shift Utilities
class TimeUtils:
    """
    Centralized time utilities:
    - Timezone handling
    - Shift resolution
    - Time parsing
    """

    def __init__(self, timezone: str = "Asia/Kolkata") -> None:
        self.tz = pytz.timezone(timezone)

    # ---------------- Public API ---------------- #

    def now(self) -> datetime:
        """Return timezone-aware current datetime."""
        return datetime.now(self.tz)

    def from_timestamp(self, ts: float) -> datetime:
        """Convert timestamp to timezone-aware datetime."""
        return datetime.fromtimestamp(ts, self.tz)

    def resolve_shift(self, ts: datetime, shifts: List[Dict]) -> str:
        """
        Resolve shift name based on provided datetime and shift config.
        """
        if not shifts:
            return "shift_unknown"

        t = ts.timetz().replace(tzinfo=None)

        for s in shifts:
            name = str(s.get("name", "shift"))
            start = self._parse_hhmm(s.get("start", "00:00:00"))
            end = self._parse_hhmm(s.get("end", "23:59:59"))

            if start < end:
                if start <= t < end:
                    return name
            else:
                # Overnight shift (e.g. 22:00 → 06:00)
                if t >= start or t < end:
                    return name

        return str(shifts[0].get("name", "shift"))

    def format_datetime(self, ts: float, fmt: str) -> str:
        """Format timestamp using configured timezone."""
        dt = self.from_timestamp(ts)
        return dt.strftime(fmt)

    # ---------------- Internal ---------------- #

    @staticmethod
    def _parse_hhmm(v: str) -> dtime:
        """
        Parse 'HH:MM' or 'HH:MM:SS' string into datetime.time.
        """
        parts = v.split(":")
        hour = int(parts[0])
        minute = int(parts[1])
        second = int(parts[2]) if len(parts) > 2 else 0
        return dtime(hour, minute, second)