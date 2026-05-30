"""Upload paper_eval_l2_sem_rich.yaml to all server config dirs."""
from __future__ import annotations

import os
import sys
from pathlib import Path

import paramiko

SCRIPT_DIR = Path(__file__).resolve().parent
COLLECT_ROOT = SCRIPT_DIR.parent
LOCAL = COLLECT_ROOT / "configs" / "paper_eval_l2_sem_rich.yaml"

REMOTE_PATHS = [
    "/root/autodl-tmp/CARLA-Air-demo/configs/paper_eval_l2_sem_rich.yaml",
    "/root/autodl-tmp/CARLA-Air-demo-fix/configs/paper_eval_l2_sem_rich.yaml",
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
    for remote in REMOTE_PATHS:
        remote_dir = str(Path(remote).parent)
        try:
            sftp.stat(remote_dir)
        except OSError:
            parts = remote_dir.strip("/").split("/")
            cur = ""
            for p in parts:
                cur += "/" + p
                try:
                    sftp.stat(cur)
                except OSError:
                    sftp.mkdir(cur)
        sftp.put(str(LOCAL), remote)
        size = sftp.stat(remote).st_size
        print(f"OK {remote} ({size} bytes)")
    sftp.close()
    client.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
