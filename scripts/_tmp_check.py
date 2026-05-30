import numpy as np
from PIL import Image

seq = "/root/autodl-tmp/CARLA-Air-demo/AirGroundRelay-Sim/sequences/paper_eval_l2_slam_rgbd/ugv"
d = Image.open(seq + "/front_depth/000000.png")
r = Image.open(seq + "/front_rgb/000000.png")
da = np.array(d)
print("depth mode=" + d.mode + " size=" + str(d.size) + " dtype=" + str(da.dtype) + " max=" + str(int(da.max())) + " min=" + str(int(da.min())))
print("rgb mode=" + r.mode + " size=" + str(r.size))
