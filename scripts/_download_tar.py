"""One-shot script: SFTP download the remote tar with progress.

Credentials are read from environment variables (SSH_HOST/PORT/USER/PASSWORD).
"""
import os, sys, time
import paramiko

HOST = os.environ["SSH_HOST"]
PORT = int(os.environ["SSH_PORT"])
USER = os.environ["SSH_USER"]
PASSWORD = os.environ["SSH_PASSWORD"]

REMOTE = "/root/autodl-tmp/paper_eval_v2.tar"
LOCAL = r"D:\Users\yangwan\mProject\CARLA-Air\air_ground_relay_collect\samples\paper_eval_v2_20260522\paper_eval_v2.tar"

os.makedirs(os.path.dirname(LOCAL), exist_ok=True)

client = paramiko.SSHClient()
client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
client.connect(HOST, port=PORT, username=USER, password=PASSWORD)
sftp = client.open_sftp()

total = sftp.stat(REMOTE).st_size
print(f"Remote size: {total/1e9:.2f} GB")

start = time.time()
last = [0, start]

def cb(transferred, total):
    now = time.time()
    if now - last[1] >= 5.0 or transferred == total:
        dt = now - last[1]
        speed_mb = (transferred - last[0]) / dt / 1e6 if dt > 0 else 0
        avg_mb = transferred / (now - start) / 1e6
        pct = transferred / total * 100
        eta = (total - transferred) / (avg_mb * 1e6) if avg_mb > 0 else 0
        print(f"  {pct:5.1f}%  {transferred/1e9:5.2f}/{total/1e9:.2f} GB  inst={speed_mb:5.1f} MB/s  avg={avg_mb:5.1f} MB/s  eta={eta:6.0f}s", flush=True)
        last[0] = transferred
        last[1] = now

sftp.get(REMOTE, LOCAL, callback=cb)
elapsed = time.time() - start
print(f"DONE in {elapsed:.0f}s ({total/elapsed/1e6:.1f} MB/s avg) -> {LOCAL}")
sftp.close()
client.close()
