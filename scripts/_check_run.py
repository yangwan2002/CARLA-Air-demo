"""Show relay block of config_snapshot.yaml inside the latest run."""
import os
import paramiko

HOST, PORT, USER = "connect.nmb2.seetacloud.com", 18563, "root"
PWD = os.environ["SSH_PASSWORD"]

SEQ = "/root/autodl-tmp/CARLA-Air-demo/AirGroundRelay-Sim/sequences/paper_eval_l2_sem_rich_20260527_221000"

c = paramiko.SSHClient()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect(HOST, PORT, USER, PWD, timeout=30)
sftp = c.open_sftp()
print("=== seq dir ===")
for f in sorted(sftp.listdir(SEQ)):
    try:
        st = sftp.stat(f"{SEQ}/{f}")
        print(f"  {f}  {st.st_size}")
    except Exception:
        print(f"  {f}")
snap = f"{SEQ}/config_snapshot.yaml"
try:
    with sftp.open(snap, "rb") as fp:
        text = fp.read().decode("utf-8", errors="replace")
    print("\n=== relay block (flattened) in snapshot ===")
    lines = text.splitlines()
    for i, line in enumerate(lines, 1):
        if "- name: relay_" in line:
            print(f"\n{i:4d}: {line.rstrip()}")
            for j in range(1, 15):
                if i - 1 + j < len(lines):
                    print(f"      {lines[i - 1 + j].rstrip()}")
except Exception as e:
    print("snapshot read err:", e)
sftp.close(); c.close()
