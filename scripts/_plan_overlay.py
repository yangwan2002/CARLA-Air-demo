"""Render a planning overlay on top of Town10HD's top-down photo:
  * 5 relay positions + their look_at + FOV cone
  * UGV path (4 phases, color-coded)
  * UAV path (4 phases, color-coded)
  * Phase legend
"""
import math
from pathlib import Path
import cv2
import numpy as np

ROOT = Path("d:/Users/yangwan/mProject/CARLA-Air/air_ground_relay_collect/samples")
src = cv2.imread(str(ROOT / "town10hd_topdown.png"))
H, W = src.shape[:2]

# Top-down camera was at z=250m, FOV=90, so 500m x 500m field of view.
ALT, FOV = 250.0, 90.0
SPAN_M = 2.0 * ALT * math.tan(math.radians(FOV / 2.0))   # 500
PX_PER_M = W / SPAN_M

def w2p(x, y):
    """CARLA (x_north, y_east) -> image pixel (col, row)."""
    col = W / 2 + y * PX_PER_M
    row = H / 2 - x * PX_PER_M
    return int(round(col)), int(round(row))

# -- Crop to central 80m x 80m (160 px per m would be too zoom; keep 80m) --
CROP_M = 80
half_px = int(CROP_M * PX_PER_M)
cx, cy = W // 2, H // 2
img = src[cy - half_px:cy + half_px, cx - half_px:cx + half_px].copy()
H2, W2 = img.shape[:2]

def w2p_crop(x, y):
    col = W2 / 2 + y * PX_PER_M
    row = H2 / 2 - x * PX_PER_M
    return int(round(col)), int(round(row))

# Upscale 1.5x for legibility
scale = 1.5
img = cv2.resize(img, (int(W2 * scale), int(H2 * scale)), interpolation=cv2.INTER_LINEAR)
H3, W3 = img.shape[:2]
PX_PER_M_FINAL = PX_PER_M * scale

def w2p_final(x, y):
    col = W3 / 2 + y * PX_PER_M_FINAL
    row = H3 / 2 - x * PX_PER_M_FINAL
    return int(round(col)), int(round(row))

# Slight darkening so overlays pop
img = cv2.addWeighted(img, 0.65, np.zeros_like(img), 0.35, 0)

# -------------------- Grid --------------------
for v in range(-CROP_M, CROP_M + 1, 10):
    color = (60, 60, 60) if v % 50 else (180, 130, 50)
    cv2.line(img, w2p_final(-CROP_M, v), w2p_final(CROP_M, v), color, 1, cv2.LINE_AA)
    cv2.line(img, w2p_final(v, -CROP_M), w2p_final(v, CROP_M), color, 1, cv2.LINE_AA)

# -------------------- Relays --------------------
relays = [
    ("relay_01", (-30, 40, 10),  (-30, 25.5, 0.4)),
    ("relay_02", (  0, 36, 10),  (  0, 25.5, 0.4)),
    ("relay_03", ( 30, 40, 10),  ( 30, 25.5, 0.4)),
    ("relay_04", (-15,  0, 10),  (-15, 15.0, 0.4)),
    ("relay_05", ( 15,  0, 10),  ( 15, 15.0, 0.4)),
]
RELAY_COLOR = (0, 200, 255)   # orange-ish (BGR)
for name, cam, look in relays:
    pcam = w2p_final(cam[0], cam[1])
    plook = w2p_final(look[0], look[1])
    # Camera triangle marker
    cv2.drawMarker(img, pcam, RELAY_COLOR, cv2.MARKER_TRIANGLE_DOWN, 16, 2)
    # Look-at line (where the camera is pointing on the lane)
    cv2.line(img, pcam, plook, RELAY_COLOR, 1, cv2.LINE_AA)
    cv2.circle(img, plook, 5, RELAY_COLOR, 2)
    # FOV footprint (35m wide, 20m tall on the lane plane)
    half_w_m, half_h_m = 35 / 2, 20 / 2
    # Direction from cam to look (in CARLA xy)
    dx, dy = look[0] - cam[0], look[1] - cam[1]
    L = math.hypot(dx, dy) or 1
    fx, fy = dx / L, dy / L                # forward
    px, py = -fy, fx                       # perpendicular along-lane
    corners = [
        (look[0] - fx * half_h_m + px * half_w_m, look[1] - fy * half_h_m + py * half_w_m),
        (look[0] - fx * half_h_m - px * half_w_m, look[1] - fy * half_h_m - py * half_w_m),
        (look[0] + fx * half_h_m - px * half_w_m, look[1] + fy * half_h_m - py * half_w_m),
        (look[0] + fx * half_h_m + px * half_w_m, look[1] + fy * half_h_m + py * half_w_m),
    ]
    poly = np.array([w2p_final(c[0], c[1]) for c in corners], dtype=np.int32)
    cv2.polylines(img, [poly], True, RELAY_COLOR, 1, cv2.LINE_AA)
    # Label
    label = name + f"\n{cam}"
    cv2.putText(img, name, (pcam[0] + 10, pcam[1] - 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(img, name, (pcam[0] + 10, pcam[1] - 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, RELAY_COLOR, 1, cv2.LINE_AA)

# -------------------- UGV path --------------------
# 4 phases — colors increasing in saturation
UGV_PHASES = [
    ("P1", [(-0.76, 24.61), (30.0, 25.0)],                              (255,  90,  90)),  # blue-ish
    ("P2", [(30.0, 25.0), (40.0, 25.0), (40.0, 17.0), (0.0, 17.0)],     (255, 165,   0)),  # orange-ish
    ("P3", [(0.0, 17.0), (-30.0, 17.0)],                                ( 50, 200,  50)),  # green
    ("P4", [(-30.0, 17.0), (-15.0, 17.0)],                              ( 60,  60, 230)),  # red
]
for label, pts, color in UGV_PHASES:
    poly = [w2p_final(p[0], p[1]) for p in pts]
    for i in range(len(poly) - 1):
        cv2.arrowedLine(img, poly[i], poly[i + 1], color, 4, cv2.LINE_AA, tipLength=0.05)

# UGV start marker
sp = w2p_final(-0.76, 24.61)
cv2.circle(img, sp, 9, (0, 255, 255), 3)
cv2.putText(img, "UGV START", (sp[0] + 12, sp[1] - 12),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 4, cv2.LINE_AA)
cv2.putText(img, "UGV START", (sp[0] + 12, sp[1] - 12),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1, cv2.LINE_AA)

# -------------------- UAV path --------------------
UAV_PHASES = [
    ("P1", [(-40, 10), (-10, 18)],                       (220, 100, 220)),  # magenta
    ("P2", [(-10, 18), (0, 25)],                          (220, 220,   0)),  # cyan
    ("P3", [(0, 25), (30, 25), (15, 17)],                 ( 90, 255, 200)),  # yellow-ish
    ("P4", [(15, 17), (-15, 17)],                         (180,  90, 255)),  # purple
]
for label, pts, color in UAV_PHASES:
    poly = [w2p_final(p[0], p[1]) for p in pts]
    for i in range(len(poly) - 1):
        # dashed-arrow style: thicker so it's clearly the UAV path
        cv2.arrowedLine(img, poly[i], poly[i + 1], color, 3, cv2.LINE_AA, tipLength=0.05)

# UAV start
us = w2p_final(-40, 10)
cv2.drawMarker(img, us, (0, 255, 255), cv2.MARKER_STAR, 16, 2)
cv2.putText(img, "UAV START (z=30m)", (us[0] + 12, us[1] - 8),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 4, cv2.LINE_AA)
cv2.putText(img, "UAV START (z=30m)", (us[0] + 12, us[1] - 8),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1, cv2.LINE_AA)

# -------------------- Legend / title --------------------
panel = np.zeros((220, 470, 3), dtype=np.uint8)
panel[:] = (30, 30, 30)
lines = [
    ("CARLA Town10HD - SAGENet plan v3 (60s)",  (255, 255, 255), 0.7),
    ("Relay (10m, FOV=90, 35x20m footprint)",   RELAY_COLOR,     0.5),
    ("UGV  P1 east on y=25",                   (255,  90,  90), 0.5),
    ("UGV  P2 turn -> y=17 east",              (255, 165,   0), 0.5),
    ("UGV  P3 west on y=17 (covis decay)",     ( 50, 200,  50), 0.5),
    ("UGV  P4 stop near relay_04",             ( 60,  60, 230), 0.5),
    ("UAV  P1 SW corner -> approach",          (220, 100, 220), 0.5),
    ("UAV  P2 -> over relay_02 (covis 1)",     (220, 220,   0), 0.5),
    ("UAV  P3 east + south (scale_var)",       ( 90, 255, 200), 0.5),
    ("UAV  P4 west to relay_04 (covis 2)",     (180,  90, 255), 0.5),
]
for i, (text, color, sz) in enumerate(lines):
    cv2.putText(panel, text, (10, 24 + i * 19),
                cv2.FONT_HERSHEY_SIMPLEX, sz, color, 1, cv2.LINE_AA)

# Stamp panel onto top-left corner
img[10:10 + panel.shape[0], 10:10 + panel.shape[1]] = panel

# -------------------- Coordinate axis labels --------------------
cv2.putText(img, "+x (north)", (W3 // 2 - 60, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
cv2.putText(img, "+y (east)", (W3 - 110, H3 // 2),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
cv2.arrowedLine(img, (W3 // 2, 60), (W3 // 2, 20), (255, 255, 255), 2, cv2.LINE_AA)
cv2.arrowedLine(img, (W3 - 130, H3 // 2 + 20), (W3 - 30, H3 // 2 + 20),
                (255, 255, 255), 2, cv2.LINE_AA)

out_path = ROOT / "plan_v3_overlay.png"
cv2.imwrite(str(out_path), img)
print(f"saved {out_path}, size={W3}x{H3}")
