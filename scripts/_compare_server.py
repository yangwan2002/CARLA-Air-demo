"""Compare local project files with AutoDL server copies."""
from __future__ import annotations

import hashlib
import os
from datetime import datetime

import paramiko

HOST, PORT, USER, PWD = "connect.nmb2.seetacloud.com", 18563, "root", "qil5pBerup1G"
REMOTE = "/root/autodl-tmp/CARLA-Air-demo"
LOCAL = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

FILES = [
    "src/trajectory.py",
    "scripts/main_collect.py",
    "configs/paper_eval_v2.yaml",
    "configs/paper_eval_v2_40s.yaml",
]


def md5(path: str) -> str:
    with open(path, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()


def main() -> None:
    print("Checked at", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    c = paramiko.SSHClient()
    c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    c.connect(HOST, port=PORT, username=USER, password=PWD, timeout=25)
    sftp = c.open_sftp()

    all_ok = True
    print(f"{'file':28} match  local_mtime           remote_mtime")
    for rel in FILES:
        lp = os.path.join(LOCAL, rel.replace("/", os.sep))
        rp = REMOTE + "/" + rel
        lhash = md5(lp)
        with sftp.open(rp, "rb") as f:
            rhash = hashlib.md5(f.read()).hexdigest()
        lmt = datetime.fromtimestamp(os.path.getmtime(lp)).strftime("%Y-%m-%d %H:%M:%S")
        rmt = datetime.fromtimestamp(sftp.stat(rp).st_mtime).strftime("%Y-%m-%d %H:%M:%S")
        ok = lhash == rhash
        all_ok &= ok
        tag = "YES" if ok else "NO"
        print(f"{rel:28} {tag:4}  {lmt}   {rmt}")
        if not ok:
            print("  local ", lhash)
            print("  remote", rhash)

    sftp.close()
    c.close()
    print("RESULT:", "ALL MATCH" if all_ok else "MISMATCH — need sync")


if __name__ == "__main__":
    main()
