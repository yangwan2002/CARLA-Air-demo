"""Run ORB-SLAM2 RGBD in foreground, capturing all output."""
import os, paramiko, sys

HOST = os.environ["SSH_HOST"]
PORT = int(os.environ["SSH_PORT"])
USER = os.environ["SSH_USER"]
PASSWORD = os.environ["SSH_PASSWORD"]

EXE  = "/root/autodl-tmp/slam_ws/ORB_SLAM2/Examples/RGB-D/rgbd_tum"
VOC  = "/root/autodl-tmp/slam_ws/ORB_SLAM2/Vocabulary/ORBvoc.txt"
CFG  = "/root/autodl-tmp/CARLA-Air-demo/CARLA_UGV_orbslam2.yaml"
SEQ  = "/root/autodl-tmp/CARLA-Air-demo/AirGroundRelay-Sim/sequences/paper_eval_l2_slam_rgbd"
ASSOC= SEQ + "/ugv/association.txt"
LOG  = "/root/autodl-tmp/CARLA-Air-demo/orbslam2_ugv.log"

cmd = f"pkill -f rgbd_tum 2>/dev/null; sleep 1; cd /root/autodl-tmp/CARLA-Air-demo && {EXE} {VOC} {CFG} {SEQ} {ASSOC} no_viewer 2>&1 | tee {LOG}"

client = paramiko.SSHClient()
client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
client.connect(HOST, port=PORT, username=USER, password=PASSWORD, timeout=15)
stdin, stdout, stderr = client.exec_command(cmd, timeout=300)
for line in iter(stdout.readline, ""):
    if not line:
        break
    print(line, end="", flush=True)
exit_code = stdout.channel.recv_exit_status()
print(f"\nExit code: {exit_code}")
client.close()
