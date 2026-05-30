"""Check depth PNG format on remote via exec_command directly."""
import os, paramiko

HOST = os.environ["SSH_HOST"]
PORT = int(os.environ["SSH_PORT"])
USER = os.environ["SSH_USER"]
PASSWORD = os.environ["SSH_PASSWORD"]

SCRIPT = (
    "cd /root/autodl-tmp/CARLA-Air-demo/AirGroundRelay-Sim/sequences/paper_eval_l2_slam_rgbd/ugv && "
    "ls front_depth | head -5 && ls front_rgb | head -5 && "
    "python3 -c \""
    "from PIL import Image; import numpy as np; "
    "d=Image.open('front_depth/000000.png'); "
    "r=Image.open('front_rgb/000000.png'); "
    "da=__import__('numpy').array(d); "
    "print('depth mode=%s size=%s dtype=%s max=%d min=%d'%(d.mode,str(d.size),str(da.dtype),da.max(),da.min())); "
    "print('rgb mode=%s size=%s'%(r.mode,str(r.size)))"
    "\""
)

client = paramiko.SSHClient()
client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
client.connect(HOST, port=PORT, username=USER, password=PASSWORD, timeout=15)
stdin, stdout, stderr = client.exec_command(SCRIPT, timeout=30)
print(stdout.read().decode(errors="replace"))
print(stderr.read().decode(errors="replace"))
client.close()
