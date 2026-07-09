from __future__ import annotations

import argparse
import signal
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from utils.config import load_caster_config, resolve_caster_id
from utils.logging import setup_logging


def _parse_casters(raw: str) -> list[int]:
    casters: list[int] = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        try:
            casters.append(resolve_caster_id(part))
        except ValueError as exc:
            raise argparse.ArgumentTypeError(str(exc)) from exc
    if not casters:
        raise argparse.ArgumentTypeError("At least one caster id is required")
    return casters


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run multiple casters in one process with shared model inference."
    )
    parser.add_argument("--casters", type=_parse_casters, required=True, help="Comma-separated caster ids, e.g. 4,5")
    parser.add_argument("--dry-run", action="store_true", help="Load configs and show model groups without opening cameras/models.")
    parser.add_argument("--log-level", default=None, help="Override process log level for the shared runner.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    cfgs = [load_caster_config(caster_id) for caster_id in args.casters]
    log_level = args.log_level or (cfgs[0].runtime.log_level if cfgs else "INFO")
    setup_logging(level=log_level, log_path=None)

    groups: dict[str, list[int]] = {}
    for cfg in cfgs:
        groups.setdefault(str(Path(cfg.runtime.model_path).expanduser()), []).append(cfg.caster_id)
    print("[shared-runner] multi_camera_mode=true shared_inference=true", flush=True)
    print(f"[shared-runner] casters={','.join(str(c.caster_id) for c in cfgs)}", flush=True)
    for model_path, caster_ids in groups.items():
        print(
            f"[shared-runner] model_group model={model_path} casters={','.join(str(x) for x in caster_ids)} model_instances=1",
            flush=True,
        )

    if args.dry_run:
        return 0

    from multi_camera.shared_runtime import SharedMultiCameraRuntime

    runtime = SharedMultiCameraRuntime(cfgs)

    def request_stop(*_args) -> None:
        runtime.stop()

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)
    runtime.run_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
