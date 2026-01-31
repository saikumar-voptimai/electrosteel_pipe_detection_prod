from __future__ import annotations

from enum import Enum
from typing import Iterable, List, Sequence


class RoiName(str, Enum):
    LOADCELL = "roi_loadcell"
    CASTER5_ORIGIN = "roi_caster5_origin"
    LEFT_ORIGIN = "roi_left_origin"
    RIGHT_ORIGIN = "roi_right_origin"
    SAFETY_CRITICAL = "roi_safety_critical"

    GATE1_OPEN = "roi_gate1_open"
    GATE2_OPEN = "roi_gate2_open"
    GATE1_CLOSED = "roi_gate1_closed"
    GATE2_CLOSED = "roi_gate2_closed"


REQUIRED_ROIS: Sequence[RoiName] = (
    RoiName.LOADCELL,
    RoiName.CASTER5_ORIGIN,
    RoiName.LEFT_ORIGIN,
    RoiName.RIGHT_ORIGIN,
    RoiName.SAFETY_CRITICAL,
    RoiName.GATE1_OPEN,
    RoiName.GATE2_OPEN,
    RoiName.GATE1_CLOSED,
    RoiName.GATE2_CLOSED,
)


def as_keys(rois: Iterable[RoiName]) -> List[str]:
    """Convert RoiName enums to their string values."""
    return [r.value for r in rois]

def gate_open_roi(gate_name: str) -> str:
    """Get the ROI name for the open state of a gate."""
    return f"roi_{gate_name}_open"

def gate_closed_roi(gate_name: str) -> str:
    """Get the ROI name for the closed state of a gate."""
    return f"roi_{gate_name}_closed"
