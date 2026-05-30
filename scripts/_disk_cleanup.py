"""Delete incomplete/duplicate CARLA sequence runs on remote server.

Keeps:
  - paper_eval_l2_sem_rich_20260528_155623  (525 frames, full 105s run)
  - paper_eval_l2_slam_rgbd                 (SLAM dataset, unless --include-slam)

Dry-run by default; pass --apply to actually delete.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import paramiko

SCRIPT_DIR = Path(__file__).resolve().parent
COLLECT_ROOT = SCRIPT_DIR.parent
SEQ_ROOT = "/root/autodl-tmp/CARLA-Air-demo/AirGroundRelay-Sim/sequences"

# Complete run to preserve.
KEEP_SEQUENCES = frozenset(
    {
        "paper_eval_l2_sem_rich_20260528_155623",
    }
)

# Explicit delete list: incomplete tests, duplicates, empty dirs, redundant archives.
DELETE_SEQUENCES = [
    "paper_eval_l2_sem_rich",  # untimestamped duplicate (~16G)
    "paper_eval_l2_sem_rich_fix1",
    "paper_eval_l2_sem_rich_e4153e2_check",
    "paper_eval_l2_sem_rich_20260527_191726",
    "paper_eval_l2_sem_rich_20260527_172625",
    "paper_eval_l2_sem_rich_20260527_192009",
    "paper_eval_l2_sem_rich_20260527_195828",
    "paper_eval_l2_sem_rich_20260527_203136",
    "paper_eval_l2_sem_rich_20260527_215613",
    "paper_eval_l2_sem_rich_20260527_221000",
    "paper_eval_l2_sem_rich_20260527_225159",
    "paper_eval_l2_sem_rich_20260527_231211",
    "paper_eval_l2_sem_rich_20260528_204324",  # empty
    "paper_eval_l2_sem_rich_20260528_204344",
    "paper_eval_l2_sem_rich_20260528_212705",
    "paper_eval_l2_sem_rich_20260528_222902",
    "paper_eval_v2.tar.gz",
]

OPTIONAL_SLAM = "paper_eval_l2_slam_rgbd"


def load_env(path: Path) -> None:
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip())


def run(client: paramiko.SSHClient, cmd: str, timeout: int = 600) -> tuple[int, str]:
    wrapped = f"bash -lc {repr(cmd)}"
    _, stdout, stderr = client.exec_command(wrapped, timeout=timeout)
    out = stdout.read().decode(errors="replace")
    err = stderr.read().decode(errors="replace")
    code = stdout.channel.recv_exit_status()
    return code, out + err


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Actually delete (default is dry-run)",
    )
    parser.add_argument(
        "--include-slam",
        action="store_true",
        help=f"Also delete {OPTIONAL_SLAM} (~39G)",
    )
    args = parser.parse_args()

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

    to_delete = list(DELETE_SEQUENCES)
    if args.include_slam:
        to_delete.append(OPTIONAL_SLAM)

    print("=== KEEP ===")
    for name in sorted(KEEP_SEQUENCES):
        path = f"{SEQ_ROOT}/{name}"
        _, out = run(client, f"[ -d {path} ] && du -sh {path} || echo MISSING {path}")
        print(out.strip())

    print("\n=== PLAN TO DELETE ===")
    total_human = []
    for name in to_delete:
        path = f"{SEQ_ROOT}/{name}"
        _, out = run(client, f"[ -e {path} ] && du -sh {path} || echo '  (not found) {path}'")
        line = out.strip()
        print(line)
        total_human.append(line)

    if not args.apply:
        print("\n[DRY-RUN] Pass --apply to delete. Add --include-slam to also remove SLAM dataset.")
        client.close()
        return 0

    print("\n=== DELETING ===")
    for name in to_delete:
        path = f"{SEQ_ROOT}/{name}"
        code, out = run(client, f"rm -rf {path} && echo OK: {name} || echo FAIL: {name}")
        print(out.strip())
        if code != 0:
            print(f"warning: exit {code} for {name}", file=sys.stderr)

    print("\n=== AFTER ===")
    _, out = run(client, "df -h /root/autodl-tmp; echo; du -sh /root/autodl-tmp/CARLA-Air-demo/AirGroundRelay-Sim/sequences/* 2>/dev/null | sort -hr")
    print(out)

    client.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
