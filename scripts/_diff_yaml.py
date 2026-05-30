"""Compare local + both server copies of paper_eval_l2_sem_rich.yaml."""
import hashlib
import os

import paramiko

HOST = "connect.nmb2.seetacloud.com"
PORT = 18563
USER = "root"
PWD = os.environ["SSH_PASSWORD"]

SERVER_PATHS = [
    "/root/autodl-tmp/CARLA-Air-demo/configs/paper_eval_l2_sem_rich.yaml",
    "/root/autodl-tmp/CARLA-Air-demo-fix/configs/paper_eval_l2_sem_rich.yaml",
]
LOCAL = r"d:/Users/yangwan/mProject/CARLA-Air/air_ground_relay_collect/configs/paper_eval_l2_sem_rich.yaml"


def grep_relay(text: str) -> str:
    lines = []
    for i, line in enumerate(text.splitlines(), 1):
        if any(k in line for k in ("- name: relay_", "location:", "look_at:")):
            lines.append(f"{i:4d}: {line.rstrip()}")
    return "\n".join(lines)


def main() -> None:
    with open(LOCAL, "rb") as f:
        local_bytes = f.read()
    print("LOCAL", hashlib.md5(local_bytes).hexdigest(), len(local_bytes))
    print("--- LOCAL relay block ---")
    print(grep_relay(local_bytes.decode("utf-8")))
    print()

    c = paramiko.SSHClient()
    c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    c.connect(HOST, PORT, USER, PWD, timeout=30)
    sftp = c.open_sftp()
    for p in SERVER_PATHS:
        with sftp.open(p, "rb") as f:
            data = f.read()
        print("SERVER", p)
        print("  md5:", hashlib.md5(data).hexdigest(), "size:", len(data))
        print("--- SERVER relay block ---")
        print(grep_relay(data.decode("utf-8")))
        print()
    sftp.close()
    c.close()


if __name__ == "__main__":
    main()
