"""Annotate the top-down photo with the CARLA xy axes, spawn points and
drivable waypoints, so we can hand-pick relay/UAV/UGV positions visually.

The camera is at world (0, 0, 250 m) facing -z with FOV=90, output 2048x2048.
At z=250 the half-angle FOV covers tan(45°) * 250 = 250 m to each side, so
the full 2048 px image spans 500 m × 500 m. CARLA: +x forward (north), +y
right (east), +z up. After the camera looks straight down with yaw=0, the
default screen mapping is:

  image x (column) <- world +y (i.e. CARLA east)
  image y (row)    <- world +x (i.e. CARLA north),  but inverted because
                      image rows grow downward in pixel space.

So the conversion world->pixel (with image_size = 2048, span = 500):

  px = 1024 + (world_y / 500) * 2048      (east -> right on image)
  py = 1024 - (world_x / 500) * 2048      (north -> up on image)

We'll render markers + a coordinate grid on top of the photo and save
/tmp/town10hd_annotated.png locally.
"""
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path("d:/Users/yangwan/mProject/CARLA-Air/air_ground_relay_collect/samples")
src_img = cv2.imread(str(ROOT / "town10hd_topdown.png"))
H, W = src_img.shape[:2]
print(f"image size: {W}x{H}")

# Camera config used for the shot
ALT = 250.0
FOV = 90.0
import math
half_fov = math.radians(FOV / 2.0)
half_span_m = ALT * math.tan(half_fov)  # = 250 m
SPAN_M = 2.0 * half_span_m  # 500 m total
PX_PER_M = W / SPAN_M       # ≈ 4.096 px/m

print(f"covers {SPAN_M:.0f} m x {SPAN_M:.0f} m, scale = {PX_PER_M:.3f} px/m")

def world_to_px(x, y):
    """CARLA world (x_north, y_east) -> image pixel (col, row)."""
    # World center (0,0) sits at image center (W/2, H/2)
    # +y_east -> +col (right);  +x_north -> -row (up)
    col = W / 2 + y * PX_PER_M
    row = H / 2 - x * PX_PER_M
    return int(round(col)), int(round(row))

# 1. Draw a CARLA coordinate grid every 10 m, labels every 50 m
img = src_img.copy()
GRID_M = 10
LABEL_M = 50
LIMIT_M = 120  # only annotate central region

for v in range(-LIMIT_M, LIMIT_M + 1, GRID_M):
    # vertical lines: world_y = v (constant east)
    p1 = world_to_px(-LIMIT_M, v)
    p2 = world_to_px( LIMIT_M, v)
    color = (60, 60, 60) if v % LABEL_M else (140, 70, 0)
    cv2.line(img, p1, p2, color, 1, cv2.LINE_AA)
    # horizontal lines: world_x = v (constant north)
    p3 = world_to_px(v, -LIMIT_M)
    p4 = world_to_px(v,  LIMIT_M)
    cv2.line(img, p3, p4, color, 1, cv2.LINE_AA)

# Axis labels
for v in range(-LIMIT_M, LIMIT_M + 1, LABEL_M):
    if v == 0:
        continue
    # x-axis label (along bottom edge)
    px, py = world_to_px(v, 0)
    cv2.putText(img, f"x={v}", (px + 4, py + 14),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1, cv2.LINE_AA)
    # y-axis label
    px, py = world_to_px(0, v)
    cv2.putText(img, f"y={v}", (px + 4, py + 14),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 255), 1, cv2.LINE_AA)

# Origin marker
ox, oy = world_to_px(0, 0)
cv2.circle(img, (ox, oy), 8, (0, 0, 255), 2)
cv2.putText(img, "(0,0)", (ox + 10, oy + 5),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1, cv2.LINE_AA)

# 2. Drivable waypoints (small green dots)
wps = json.load(open(ROOT / "town10hd_waypoints.json"))
print(f"drawing {len(wps)} drivable waypoints")
for w in wps:
    if abs(w["x"]) > LIMIT_M or abs(w["y"]) > LIMIT_M:
        continue
    p = world_to_px(w["x"], w["y"])
    cv2.circle(img, p, 1, (0, 255, 0), -1)

# 3. Spawn points (numbered cyan circles)
sps = json.load(open(ROOT / "town10hd_spawnpoints.json"))
for s in sps:
    if abs(s["x"]) > LIMIT_M or abs(s["y"]) > LIMIT_M:
        continue
    p = world_to_px(s["x"], s["y"])
    cv2.circle(img, p, 4, (255, 255, 0), 1)
    cv2.putText(img, f"#{s['i']}", (p[0] + 5, p[1] - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 0), 1, cv2.LINE_AA)

# 4. Big legend block (top-left)
legend = [
    "Town10HD top-down (z=250m, FOV=90deg)",
    f"Span: {SPAN_M:.0f}m x {SPAN_M:.0f}m   Scale: {PX_PER_M:.2f} px/m",
    "+x = north (up on image)   +y = east (right)",
    "Green dots = drivable waypoints",
    "Cyan circles = spawn points (with index)",
    "Yellow lines = grid (10m), red origin = (0,0)",
]
y0 = 30
for i, line in enumerate(legend):
    cv2.putText(img, line, (10, y0 + i * 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(img, line, (10, y0 + i * 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA)

cv2.imwrite(str(ROOT / "town10hd_annotated.png"), img)
print("saved town10hd_annotated.png")

# Also save a zoomed crop centered on origin (200x200 m, 1.6x scale)
zoom_m = 100
z_x0 = W // 2 - int(zoom_m * PX_PER_M)
z_y0 = H // 2 - int(zoom_m * PX_PER_M)
z_x1 = W // 2 + int(zoom_m * PX_PER_M)
z_y1 = H // 2 + int(zoom_m * PX_PER_M)
crop = img[z_y0:z_y1, z_x0:z_x1]
cv2.imwrite(str(ROOT / "town10hd_zoomed.png"), crop)
print(f"saved town10hd_zoomed.png ({crop.shape[1]}x{crop.shape[0]})")
