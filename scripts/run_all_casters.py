from __future__ import annotations

import argparse
import signal
import subprocess
import sys
import time
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
  sys.path.insert(0, str(SRC_DIR))

from utils.config import resolve_caster_id


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
  parser = argparse.ArgumentParser(description="Run multiple caster app processes.")
  parser.add_argument("--casters", type=_parse_casters, required=True, help="Comma-separated caster ids, e.g. 1,2,3,4")
  return parser.parse_args()


def main() -> int:
  args = parse_args()
  children: list[subprocess.Popen] = []

  def terminate_children(*_args) -> None:
    for child in children:
      if child.poll() is None:
        child.terminate()

  signal.signal(signal.SIGINT, terminate_children)
  signal.signal(signal.SIGTERM, terminate_children)

  try:
    for caster_id in args.casters:
      cmd = [sys.executable, "src/app.py", "--caster", str(caster_id)]
      print(f"[launcher] starting caster {caster_id}: {' '.join(cmd)}", flush=True)
      children.append(subprocess.Popen(cmd, cwd=REPO_ROOT))

    exit_code = 0
    while children:
      for child in list(children):
        code = child.poll()
        if code is not None:
          children.remove(child)
          if code != 0:
            exit_code = code
            terminate_children()
      if children:
        time.sleep(1.0)
    return exit_code
  finally:
    terminate_children()
    for child in children:
      try:
        child.wait(timeout=5)
      except subprocess.TimeoutExpired:
        child.kill()


if __name__ == "__main__":
  raise SystemExit(main())
