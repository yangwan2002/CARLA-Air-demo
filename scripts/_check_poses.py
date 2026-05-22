import csv
import os
import paramiko

HOST = os.environ["SSH_HOST"]
PORT = int(os.environ["SSH_PORT"])
USER = os.environ["SSH_USER"]
PWD = os.environ["SSH_PASSWORD"]
BASE = "/root/autodl-tmp/CARLA-Air-demo/AirGroundRelay-Sim/sequences/paper_eval_v2"
ids = [0, 58, 116, 175, 233, 291, 349, 408, 466, 524]

c = paramiko.SSHClient()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect(HOST, port=PORT, username=USER, password=PWD, timeout=20)
sftp = c.open_sftp()

with sftp.open(f"{BASE}/ugv/pose.csv") as f:
    ugv = {int(r["frame_id"]): (float(r["x"]), float(r["y"])) for r in csv.DictReader(f)}
with sftp.open(f"{BASE}/uav/pose.csv") as f:
    uav = {int(r["frame_id"]): (float(r["x"]), float(r["y"])) for r in csv.DictReader(f)}

print("frame      ugv_xy              uav_xy              dist_m")
for i in ids:
    gx, gy = ugv[i]
    ax, ay = uav[i]
    d = ((gx - ax) ** 2 + (gy - ay) ** 2) ** 0.5
    print(f"{i:06d}   ({gx:7.1f},{gy:6.1f})   ({ax:7.1f},{ay:6.1f})   {d:5.1f}")

sftp.close()
c.close()
