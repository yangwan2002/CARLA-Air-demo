"""Download evenly-spaced preview frames from a timestamped remote sequence."""
from __future__ import annotations

import os
import sys
from pathlib import Path

import paramiko

SCRIPT_DIR = Path(__file__).resolve().parent
COLLECT_ROOT = SCRIPT_DIR.parent


def load_env(path: Path) -> None:
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip())


def pick_indices(n_files: int, n_pick: int) -> list[int]:
    if n_files <= 0:
        return []
    if n_files <= n_pick:
        return list(range(n_files))
    return [round(i * (n_files - 1) / (n_pick - 1)) for i in range(n_pick)]


def main() -> int:
    load_env(COLLECT_ROOT / ".env")

    host = os.environ["SSH_HOST"]
    port = int(os.environ["SSH_PORT"])
    user = os.environ["SSH_USER"]
    password = os.environ["SSH_PASSWORD"]

    remote_seq = os.environ.get(
        "REMOTE_SEQ",
        "/root/autodl-tmp/CARLA-Air-demo/AirGroundRelay-Sim/sequences/"
        "paper_eval_l2_sem_rich_20260527_225159",
    )
    seq_name = Path(remote_seq).name
    local_out = COLLECT_ROOT / "samples" / f"{seq_name}_preview"
    n_frames = int(os.environ.get("N_FRAMES", "8"))
    subdirs = (
        "ugv/front_rgb",
        "uav/front_rgb",
        "uav/down_rgb",
        "relay/relay_01/rgb",
        "relay/relay_02/rgb",
        "relay/relay_03/rgb",
        "relay/relay_04/rgb",
        "relay/relay_05/rgb",
        "relay/relay_06/rgb",
        "relay/relay_07/rgb",
    )

    local_out.mkdir(parents=True, exist_ok=True)
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(host, port=port, username=user, password=password, timeout=60)
    sftp = client.open_sftp()

    total = 0
    for sub in subdirs:
        remote_dir = f"{remote_seq}/{sub}"
        try:
            names = sorted(
                f for f in sftp.listdir(remote_dir) if f.lower().endswith(".png")
            )
        except FileNotFoundError:
            print(f"MISSING: {remote_dir}")
            continue

        idxs = pick_indices(len(names), n_frames)
        local_sub = local_out / sub
        local_sub.mkdir(parents=True, exist_ok=True)
        print(f"{sub}: {len(names)} files -> downloading {len(idxs)} frames")

        for i in idxs:
            name = names[i]
            sftp.get(f"{remote_dir}/{name}", str(local_sub / name))
            total += 1
            print(f"  [{i:04d}/{len(names) - 1:04d}] {name}")

    sftp.close()
    client.close()
    print(f"Done. {total} images saved to {local_out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
