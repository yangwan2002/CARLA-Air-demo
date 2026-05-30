"""Download evenly-spaced early/mid/late RGB frames from remote sequence."""
from __future__ import annotations

import os
import sys

import paramiko

HOST = os.environ["SSH_HOST"]
PORT = int(os.environ["SSH_PORT"])
USER = os.environ["SSH_USER"]
PASSWORD = os.environ["SSH_PASSWORD"]
REMOTE_SEQ = "/root/autodl-tmp/CARLA-Air-demo/AirGroundRelay-Sim/sequences/paper_eval_l2_sem_rich"
LOCAL_OUT = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    "samples",
    "paper_eval_l2_sem_rich_preview_frames",
)
N_FRAMES = 10
SUBDIRS = ("ugv/front_rgb", "uav/front_rgb")


def pick_indices(n_files: int, n_pick: int) -> list[int]:
    if n_files <= 0:
        return []
    if n_files <= n_pick:
        return list(range(n_files))
    # Evenly spaced: covers start, middle, end.
    return [round(i * (n_files - 1) / (n_pick - 1)) for i in range(n_pick)]


def list_pngs(sftp: paramiko.SFTPClient, remote_dir: str) -> list[str]:
    names = sorted(f for f in sftp.listdir(remote_dir) if f.lower().endswith(".png"))
    return names


def main() -> int:
    os.makedirs(LOCAL_OUT, exist_ok=True)
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(HOST, port=PORT, username=USER, password=PASSWORD, timeout=30)
    sftp = client.open_sftp()

    for sub in SUBDIRS:
        remote_dir = f"{REMOTE_SEQ}/{sub}"
        try:
            names = list_pngs(sftp, remote_dir)
        except FileNotFoundError:
            print(f"MISSING: {remote_dir}")
            continue

        idxs = pick_indices(len(names), N_FRAMES)
        local_sub = os.path.join(LOCAL_OUT, sub.replace("/", os.sep))
        os.makedirs(local_sub, exist_ok=True)
        print(f"{sub}: {len(names)} files -> downloading {len(idxs)} frames")

        for i in idxs:
            name = names[i]
            remote_path = f"{remote_dir}/{name}"
            local_path = os.path.join(local_sub, name)
            sftp.get(remote_path, local_path)
            print(f"  [{i:04d}/{len(names)-1:04d}] {name}")

    sftp.close()
    client.close()
    print(f"Done. Saved to {LOCAL_OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
