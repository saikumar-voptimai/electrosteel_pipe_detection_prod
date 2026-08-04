from __future__ import annotations

from datetime import datetime, timedelta
import logging
from pathlib import Path
import threading

from utils.camera_profiles import (
    CameraControlState,
    active_camera_profile,
    load_camera_control_state,
    parse_time_minutes,
)


logger = logging.getLogger(__name__)


class CameraProfileScheduler:
    """Apply camera profiles from a lightweight, independent scheduler thread."""

    def __init__(
        self,
        capture,
        profiles,
        *,
        config_path: str | Path | None = None,
        reload_interval_s: float = 2.0,
    ):
        self.capture = capture
        self.profiles = dict(profiles or {})
        camera_cfg = getattr(capture, "camera_cfg", None)
        self._state = CameraControlState(
            auto_exposure=bool(getattr(camera_cfg, "auto_exposure", False)),
            auto_gain=bool(getattr(camera_cfg, "auto_gain", False)),
            profiles=self.profiles,
        )
        self.config_path = Path(config_path) if config_path else None
        self.reload_interval_s = max(0.25, float(reload_interval_s))
        self.thread: threading.Thread | None = None
        self.running = False
        self._stop_event = threading.Event()
        self._config_mtime_ns: int | None = None
        self._last_signature: tuple | None = None
        self._last_load_error: str | None = None

    def start(self) -> None:
        if self.running or not self.profiles:
            return
        self.running = True
        self._stop_event.clear()
        self.thread = threading.Thread(
            target=self._run,
            name="camera-profile-scheduler",
            daemon=True,
        )
        self.thread.start()

    def stop(self) -> None:
        self.running = False
        self._stop_event.set()
        if self.thread and self.thread.is_alive() and self.thread is not threading.current_thread():
            self.thread.join(timeout=self.reload_interval_s + 1.0)

    def _reload_state_if_changed(self) -> CameraControlState:
        if self.config_path is None:
            return self._state
        try:
            mtime_ns = self.config_path.stat().st_mtime_ns
            if self._config_mtime_ns == mtime_ns:
                return self._state
            state = load_camera_control_state(self.config_path, defaults_if_empty=True)
            self._state = state
            self.profiles = state.profiles
            self._config_mtime_ns = mtime_ns
            self._last_load_error = None
            logger.info("Camera schedule reloaded | path=%s | profiles=%d", self.config_path, len(self.profiles))
        except Exception as exc:
            error = str(exc)
            if error != self._last_load_error:
                logger.error(
                    "Camera schedule reload rejected; keeping last valid schedule | path=%s | error=%s",
                    self.config_path,
                    error,
                )
                self._last_load_error = error
        return self._state

    def _run(self) -> None:
        while self.running:
            state = self._reload_state_if_changed()
            now = datetime.now()
            name, profile = self._get_active_profile(now)
            signature = (state.auto_exposure, state.auto_gain, name, profile)
            if profile is not None and signature != self._last_signature:
                self.capture.apply_camera_controls(
                    auto_exposure=state.auto_exposure,
                    auto_gain=state.auto_gain,
                )
                self.capture.apply_profile(profile)
                self._last_signature = signature
                next_switch = self._get_next_switch(now)
                logger.info(
                    "Camera profile applied | profile=%s | auto_exposure=%s | auto_gain=%s | next_switch=%s",
                    name,
                    state.auto_exposure,
                    state.auto_gain,
                    next_switch,
                )

            next_switch = self._get_next_switch(now)
            until_switch_s = max(0.05, (next_switch - now).total_seconds())
            self._stop_event.wait(min(self.reload_interval_s, until_switch_s))

    def _get_active_profile(self, now: datetime):
        return active_camera_profile(self.profiles, now)

    def _get_next_switch(self, now: datetime) -> datetime:
        times: list[datetime] = []
        for profile in self.profiles.values():
            start_minutes = parse_time_minutes(profile.start)
            dt = now.replace(
                hour=start_minutes // 60,
                minute=start_minutes % 60,
                second=0,
                microsecond=0,
            )
            if dt <= now:
                dt += timedelta(days=1)
            times.append(dt)
        return min(times) if times else now + timedelta(seconds=self.reload_interval_s)
