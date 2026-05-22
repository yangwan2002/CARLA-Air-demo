"""Sync local src/, configs/, and collection scripts to AutoDL server."""
from __future__ import annotations

import os
import sys

import paramiko

HOST = os.environ["SSH_HOST"]
PORT = int(os.environ["SSH_PORT"])
USER = os.environ["SSH_USER"]
PASSWORD = os.environ["SSH_PASSWORD"]
REMOTE_ROOT = os.environ.get("REMOTE_ROOT", "/root/autodl-tmp/CARLA-Air-demo")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def ensure_dir(sftp: paramiko.SFTPClient, path: str) -> None:
    parts = path.replace("\\", "/").strip("/").split("/")
    cur = ""
    for p in parts:
        cur += "/" + p
        try:
            sftp.stat(cur)
        except OSError:
            sftp.mkdir(cur)


def upload_tree(sftp: paramiko.SFTPClient, local_dir: str, remote_dir: str) -> int:
    ensure_dir(sftp, remote_dir)
    n = 0
    for name in sorted(os.listdir(local_dir)):
        if name == "__pycache__" or name.endswith(".pyc"):
            continue
        lp = os.path.join(local_dir, name)
        rp = remote_dir.rstrip("/") + "/" + name
        if os.path.isdir(lp):
            n += upload_tree(sftp, lp, rp)
        else:
            sftp.put(lp, rp)
            print(f"  PUT {name}")
            n += 1
    return n


def main() -> int:
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(HOST, port=PORT, username=USER, password=PASSWORD, timeout=30)
    sftp = client.open_sftp()

    total = 0
    print("=== src/ ===")
    total += upload_tree(sftp, os.path.join(ROOT, "src"), f"{REMOTE_ROOT}/src")

    print("=== configs/ ===")
    total += upload_tree(sftp, os.path.join(ROOT, "configs"), f"{REMOTE_ROOT}/configs")

    print("=== scripts (main_collect.py + helpers) ===")
    scripts_local = os.path.join(ROOT, "scripts")
    scripts_remote = f"{REMOTE_ROOT}/scripts"
    ensure_dir(sftp, scripts_remote)
    for name in os.listdir(scripts_local):
        if not name.endswith(".py"):
            continue
        lp = os.path.join(scripts_local, name)
        if os.path.isfile(lp):
            sftp.put(lp, f"{scripts_remote}/{name}")
            print(f"  PUT {name}")
            total += 1

    sftp.close()
    client.close()
    print(f"\nDone. {total} files uploaded to {REMOTE_ROOT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
