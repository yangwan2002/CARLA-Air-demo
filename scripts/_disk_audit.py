"""Audit remote disk usage for CARLA-Air data (read-only)."""
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


def run(client: paramiko.SSHClient, cmd: str, timeout: int = 120) -> str:
    wrapped = f"bash -lc {repr(cmd)}"
    _, stdout, stderr = client.exec_command(wrapped, timeout=timeout)
    out = stdout.read().decode(errors="replace")
    err = stderr.read().decode(errors="replace")
    code = stdout.channel.recv_exit_status()
    if code != 0 and err.strip():
        print(err, file=sys.stderr)
    return out


def main() -> int:
    load_env(COLLECT_ROOT / ".env")
    host = os.environ["SSH_HOST"]
    port = int(os.environ["SSH_PORT"])
    user = os.environ["SSH_USER"]
    password = os.environ["SSH_PASSWORD"]

    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(host, port=port, username=user, password=password, timeout=60)

    print("=== df -h ===")
    print(run(client, "df -h /root/autodl-tmp /root 2>/dev/null; df -h"))

    seq_root = "/root/autodl-tmp/CARLA-Air-demo/AirGroundRelay-Sim/sequences"
    print(f"\n=== sequences under {seq_root} ===")
    print(
        run(
            client,
            f"if [ -d {seq_root} ]; then "
            f"du -sh {seq_root}/* 2>/dev/null | sort -hr; "
            f"else echo 'MISSING: {seq_root}'; fi",
            timeout=300,
        )
    )

    print("\n=== other large dirs under /root/autodl-tmp ===")
    print(
        run(
            client,
            "du -sh /root/autodl-tmp/* 2>/dev/null | sort -hr | head -25",
            timeout=300,
        )
    )

    print("\n=== frame counts (paper_eval_l2_sem_rich*) ===")
    print(
        run(
            client,
            f"for d in {seq_root}/paper_eval_l2_sem_rich_*; do "
            f"[ -d \"$d\" ] || continue; "
            f"n=$(ls \"$d/ugv/front_rgb\" 2>/dev/null | wc -l); "
            f"echo \"$(du -sh \"$d\" | cut -f1)\t$n frames\t$(basename \"$d\")\"; "
            f"done | sort -k2 -hr",
            timeout=300,
        )
    )

    client.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
