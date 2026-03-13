from datetime import datetime, timedelta
import threading
import time
import logging

logger = logging.getLogger(__name__)


class CameraProfileScheduler:
    def __init__(self, capture, profiles):
        self.capture = capture
        self.profiles = profiles
        self.thread = None
        self.running = False
    # Each profile is expected to have 'start' and 'end' in "HH:MM" 24-hour format.
    def start(self):
        if not self.profiles:
            return
        self.running = True
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()
    # Call this to stop the scheduler thread gracefully.
    def stop(self):
        self.running = False
    #` This is the main loop that checks the current time against profile schedules and applies settings.
    def _run(self):
        while self.running:
            now = datetime.now()
            name, profile = self._get_active_profile(now)
            logger.info("Camera profile → %s", name)
            self.capture.apply_profile(profile)
            next_switch = self._get_next_switch(now)
            sleep_s = (next_switch - now).total_seconds()
            logger.info(
                "Next camera switch at %s (sleep %.0f sec)",
                next_switch,
                sleep_s,
            )
            while sleep_s > 0 and self.running:
                time.sleep(min(60, sleep_s))
                sleep_s -= 60
    # This function determines which profile should be active based on the current time.
    def _get_active_profile(self, now):
        for name, p in self.profiles.items():
            start = datetime.strptime(p.start, "%H:%M").time()
            end = datetime.strptime(p.end, "%H:%M").time()
            if start < end:
                if start <= now.time() < end:
                    return name, p
            else:
                if now.time() >= start or now.time() < end:
                    return name, p
        return None, None
    # This function calculates when the next profile switch should occur.
    def _get_next_switch(self, now):
        times = []
        for p in self.profiles.values():
            t = datetime.strptime(p.start, "%H:%M").time()
            dt = now.replace(hour=t.hour, minute=t.minute, second=0, microsecond=0)
            if dt <= now:
                dt += timedelta(days=1)
            times.append(dt)

        return min(times)