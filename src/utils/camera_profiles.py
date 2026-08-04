from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, time as datetime_time
import os
from pathlib import Path
import re
import stat
import tempfile
from typing import Any, Mapping

import yaml

from utils.config import CameraProfileCfg


TIME_RE = re.compile(r"^(?:[01]\d|2[0-3]):[0-5]\d$")


@dataclass(frozen=True)
class CameraControlState:
    auto_exposure: bool
    auto_gain: bool
    profiles: dict[str, CameraProfileCfg]


def default_camera_profiles() -> dict[str, CameraProfileCfg]:
    """Return a fresh two-profile, full-day default schedule."""
    return {
        "day": CameraProfileCfg(
            start="08:00",
            end="18:00",
            exposure_us=80000,
            gain_db=10,
            gamma_enable=False,
            gamma=0.8,
        ),
        "night": CameraProfileCfg(
            start="18:00",
            end="08:00",
            exposure_us=100000,
            gain_db=12,
            gamma_enable=False,
            gamma=1.4,
        ),
    }


def parse_time_minutes(value: str) -> int:
    text = str(value).strip()
    if not TIME_RE.fullmatch(text):
        raise ValueError(f"Invalid time {value!r}; expected HH:MM in 24-hour format.")
    hour, minute = (int(part) for part in text.split(":"))
    return hour * 60 + minute


def _minute_label(value: int) -> str:
    value %= 24 * 60
    return f"{value // 60:02d}:{value % 60:02d}"


def validate_camera_profiles(
    profiles: Mapping[str, CameraProfileCfg],
    *,
    require_full_day: bool = True,
) -> dict[str, CameraProfileCfg]:
    """Validate, normalize, and order an arbitrary camera profile schedule."""
    if len(profiles) < 2:
        raise ValueError("At least two camera profiles are required.")

    normalized: dict[str, CameraProfileCfg] = {}
    starts: dict[str, int] = {}
    segments: list[tuple[int, int, str]] = []

    for raw_name, raw_profile in profiles.items():
        name = str(raw_name).strip()
        if not name:
            raise ValueError("Every camera profile must have a name.")
        if name in normalized:
            raise ValueError(f"Duplicate camera profile name: {name!r}.")

        start_text = str(raw_profile.start).strip()
        end_text = str(raw_profile.end).strip()
        start = parse_time_minutes(start_text)
        end = parse_time_minutes(end_text)
        if start == end:
            raise ValueError(f"Profile {name!r} must not have the same start and end time.")

        exposure_us = int(raw_profile.exposure_us)
        gain_db = int(raw_profile.gain_db)
        gamma = float(raw_profile.gamma)
        if exposure_us <= 0:
            raise ValueError(f"Profile {name!r} exposure must be greater than zero.")
        if gain_db < 0:
            raise ValueError(f"Profile {name!r} gain must be zero or greater.")
        if not 0.1 <= gamma <= 10.0:
            raise ValueError(f"Profile {name!r} gamma must be between 0.1 and 10.0.")

        normalized[name] = CameraProfileCfg(
            start=start_text,
            end=end_text,
            exposure_us=exposure_us,
            gain_db=gain_db,
            gamma_enable=bool(raw_profile.gamma_enable),
            gamma=gamma,
        )
        starts[name] = start
        if start < end:
            segments.append((start, end, name))
        else:
            segments.append((start, 24 * 60, name))
            segments.append((0, end, name))

    if require_full_day:
        cursor = 0
        previous_name: str | None = None
        for start, end, name in sorted(segments):
            if start > cursor:
                raise ValueError(
                    f"Camera schedule has a gap from {_minute_label(cursor)} to {_minute_label(start)}."
                )
            if start < cursor:
                raise ValueError(
                    f"Camera profiles {previous_name!r} and {name!r} overlap near {_minute_label(start)}."
                )
            cursor = end
            previous_name = name
        if cursor < 24 * 60:
            raise ValueError(f"Camera schedule has a gap from {_minute_label(cursor)} to 00:00.")

    return dict(sorted(normalized.items(), key=lambda item: (starts[item[0]], item[0])))


def active_camera_profile(
    profiles: Mapping[str, CameraProfileCfg],
    now: datetime | datetime_time | None = None,
) -> tuple[str | None, CameraProfileCfg | None]:
    current = now or datetime.now()
    current_time = current.time() if isinstance(current, datetime) else current
    current_minutes = current_time.hour * 60 + current_time.minute

    for name, profile in profiles.items():
        start = parse_time_minutes(profile.start)
        end = parse_time_minutes(profile.end)
        if start < end:
            active = start <= current_minutes < end
        else:
            active = current_minutes >= start or current_minutes < end
        if active:
            return name, profile
    return None, None


def _camera_settings(raw: dict[str, Any]) -> dict[str, Any]:
    camera = raw.setdefault("camera", {})
    if not isinstance(camera, dict):
        raise ValueError("camera.yaml must contain a mapping under 'camera'.")
    camera_type = str(camera.get("type", "va_imaging")).strip().lower()
    if camera_type == "va_imaging" and isinstance(camera.get("va_imaging"), dict):
        return camera["va_imaging"]
    return camera


def load_camera_control_state(
    camera_cfg_path: str | Path,
    *,
    defaults_if_empty: bool = True,
) -> CameraControlState:
    path = Path(camera_cfg_path)
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"Invalid camera config in {path}.")
    settings = _camera_settings(raw)
    profiles_raw = settings.get("profiles", {}) or {}
    profiles = {
        str(name): CameraProfileCfg(
            start=str(profile["start"]),
            end=str(profile["end"]),
            exposure_us=int(profile["exposure_us"]),
            gain_db=int(profile["gain_db"]),
            gamma_enable=bool(profile.get("gamma_enable", False)),
            gamma=float(profile.get("gamma", 1.0)),
        )
        for name, profile in profiles_raw.items()
    }
    if not profiles and defaults_if_empty:
        profiles = default_camera_profiles()
    if profiles:
        profiles = validate_camera_profiles(profiles)
    return CameraControlState(
        auto_exposure=bool(settings.get("auto_exposure", False)),
        auto_gain=bool(settings.get("auto_gain", False)),
        profiles=profiles,
    )


def save_camera_control_state(
    camera_cfg_path: str | Path,
    state: CameraControlState,
) -> CameraControlState:
    """Atomically update only scheduler-controlled fields in camera.yaml."""
    path = Path(camera_cfg_path)
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"Invalid camera config in {path}.")
    profiles = validate_camera_profiles(state.profiles)
    settings = _camera_settings(raw)
    settings["profiles"] = {
        name: {
            "start": profile.start,
            "end": profile.end,
            "exposure_us": profile.exposure_us,
            "gain_db": profile.gain_db,
            "gamma_enable": profile.gamma_enable,
            "gamma": profile.gamma,
        }
        for name, profile in profiles.items()
    }
    settings["auto_exposure"] = bool(state.auto_exposure)
    settings["auto_gain"] = bool(state.auto_gain)

    path.parent.mkdir(parents=True, exist_ok=True)
    existing_mode = stat.S_IMODE(path.stat().st_mode) if path.exists() else 0o644
    temp_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as temp_file:
            yaml.safe_dump(raw, temp_file, sort_keys=False, allow_unicode=True)
            temp_file.flush()
            os.fsync(temp_file.fileno())
            temp_name = temp_file.name
        os.chmod(temp_name, existing_mode)
        os.replace(temp_name, path)
    finally:
        if temp_name and os.path.exists(temp_name):
            os.unlink(temp_name)

    return CameraControlState(
        auto_exposure=bool(state.auto_exposure),
        auto_gain=bool(state.auto_gain),
        profiles=profiles,
    )
