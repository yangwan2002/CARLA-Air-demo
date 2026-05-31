"""Upload safety-related collect sources to server dirs."""
from __future__ import annotations

import os
import sys
from pathlib import Path

import paramiko

SCRIPT_DIR = Path(__file__).resolve().parent
COLLECT_ROOT = SCRIPT_DIR.parent

UPLOADS = [
    (COLLECT_ROOT / "src" / "scene_dressing.py", [
        "/root/autodl-tmp/CARLA-Air-demo/src/scene_dressing.py",
        "/root/autodl-tmp/CARLA-Air-demo-fix/src/scene_dressing.py",
    ]),
    (COLLECT_ROOT / "src" / "trajectory.py", [
        "/root/autodl-tmp/CARLA-Air-demo/src/trajectory.py",
        "/root/autodl-tmp/CARLA-Air-demo-fix/src/trajectory.py",
    ]),
    (COLLECT_ROOT / "scripts" / "main_collect.py", [
        "/root/autodl-tmp/CARLA-Air-demo/scripts/main_collect.py",
        "/root/autodl-tmp/CARLA-Air-demo-fix/scripts/main_collect.py",
    ]),
]


def load_env(path: Path) -> None:
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip())


def main() -> int:
    load_env(COLLECT_ROOT / ".env")
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(
        os.environ["SSH_HOST"],
        port=int(os.environ["SSH_PORT"]),
        username=os.environ["SSH_USER"],
        password=os.environ["SSH_PASSWORD"],
        timeout=60,
    )
    sftp = client.open_sftp()
    for local, remotes in UPLOADS:
        for remote in remotes:
            sftp.put(str(local), remote)
            size = sftp.stat(remote).st_size
            print(f"OK {remote} ({size} bytes)")
    sftp.close()
    client.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
