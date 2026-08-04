from __future__ import annotations

from datetime import datetime
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from utils.camera_profiles import (
    CameraControlState,
    active_camera_profile,
    default_camera_profiles,
    load_camera_control_state,
    save_camera_control_state,
    validate_camera_profiles,
)
from utils.camera_scheduler import CameraProfileScheduler
from utils.config import CameraProfileCfg


def _profile(start: str, end: str, exposure: int = 80000) -> CameraProfileCfg:
    return CameraProfileCfg(
        start=start,
        end=end,
        exposure_us=exposure,
        gain_db=10,
        gamma_enable=False,
        gamma=1.0,
    )


def _four_profiles() -> dict[str, CameraProfileCfg]:
    return {
        "profile_1": _profile("06:00", "10:00", 60000),
        "profile_2": _profile("10:00", "16:00", 70000),
        "profile_3": _profile("16:00", "20:00", 80000),
        "profile_4": _profile("20:00", "06:00", 90000),
    }


class CameraProfileTests(unittest.TestCase):
    def test_default_schedule_has_day_and_night(self) -> None:
        profiles = validate_camera_profiles(default_camera_profiles())
        self.assertEqual(set(profiles), {"day", "night"})

    def test_custom_four_profile_schedule_and_overnight_selection(self) -> None:
        profiles = validate_camera_profiles(_four_profiles())
        name, profile = active_camera_profile(profiles, datetime(2026, 8, 3, 5, 30))
        self.assertEqual(name, "profile_4")
        self.assertEqual(profile.exposure_us, 90000)
        self.assertEqual(active_camera_profile(profiles, datetime(2026, 8, 3, 10, 0))[0], "profile_2")

    def test_schedule_rejects_gaps_and_overlaps(self) -> None:
        with self.assertRaisesRegex(ValueError, "gap"):
            validate_camera_profiles(
                {
                    "day": _profile("08:00", "17:00"),
                    "night": _profile("18:00", "08:00"),
                }
            )
        with self.assertRaisesRegex(ValueError, "overlap"):
            validate_camera_profiles(
                {
                    "day": _profile("08:00", "19:00"),
                    "night": _profile("18:00", "08:00"),
                }
            )

    def test_nested_va_config_is_updated_without_losing_other_settings(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "camera.yaml"
            path.write_text(
                yaml.safe_dump(
                    {
                        "camera": {
                            "type": "va_imaging",
                            "va_imaging": {
                                "id": "camera-1",
                                "ip": "192.168.1.123",
                                "profiles": {},
                            },
                        }
                    },
                    sort_keys=False,
                ),
                encoding="utf-8",
            )
            save_camera_control_state(
                path,
                CameraControlState(
                    auto_exposure=False,
                    auto_gain=True,
                    profiles=_four_profiles(),
                ),
            )

            raw = yaml.safe_load(path.read_text(encoding="utf-8"))
            settings = raw["camera"]["va_imaging"]
            self.assertEqual(settings["ip"], "192.168.1.123")
            self.assertTrue(settings["auto_gain"])
            self.assertEqual(len(settings["profiles"]), 4)
            loaded = load_camera_control_state(path)
            self.assertEqual(set(loaded.profiles), set(_four_profiles()))

    def test_empty_config_uses_two_defaults_in_dashboard_state(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "camera.yaml"
            path.write_text("camera:\n  type: va_imaging\n", encoding="utf-8")
            state = load_camera_control_state(path)
            self.assertEqual(set(state.profiles), {"day", "night"})

    def test_scheduler_reloads_valid_file_schedule(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "camera.yaml"
            path.write_text(
                yaml.safe_dump(
                    {
                        "camera": {
                            "type": "va_imaging",
                            "profiles": {
                                name: {
                                    "start": profile.start,
                                    "end": profile.end,
                                    "exposure_us": profile.exposure_us,
                                    "gain_db": profile.gain_db,
                                    "gamma_enable": profile.gamma_enable,
                                    "gamma": profile.gamma,
                                }
                                for name, profile in _four_profiles().items()
                            },
                            "auto_exposure": True,
                            "auto_gain": False,
                        }
                    },
                    sort_keys=False,
                ),
                encoding="utf-8",
            )
            capture = SimpleNamespace(
                camera_cfg=SimpleNamespace(auto_exposure=False, auto_gain=False)
            )
            scheduler = CameraProfileScheduler(
                capture,
                default_camera_profiles(),
                config_path=path,
            )
            state = scheduler._reload_state_if_changed()

            self.assertTrue(state.auto_exposure)
            self.assertEqual(set(scheduler.profiles), set(_four_profiles()))


if __name__ == "__main__":
    unittest.main()
