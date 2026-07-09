from __future__ import annotations

import os
from dataclasses import dataclass

try:
    import psutil
except Exception:  # pragma: no cover - optional runtime dependency
    psutil = None


@dataclass(frozen=True)
class ProcessMemory:
    rss_mb: float
    vms_mb: float


def get_process_memory_mb(pid: int | None = None) -> ProcessMemory | None:
    if psutil is None:
        return None

    try:
        process = psutil.Process(os.getpid() if pid is None else pid)
        info = process.memory_info()
    except Exception:
        return None

    mb = 1024.0 * 1024.0
    return ProcessMemory(rss_mb=info.rss / mb, vms_mb=info.vms / mb)


def fmt_mb(value: float | None) -> str:
    return "unavailable" if value is None else f"{value:.1f}"
