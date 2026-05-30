"""One-off viz: 172625 uav@000000 + relay_01@000037 (red bbox style)."""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from compute_uav_relay_overlap import load_frames, overlap_on_ground, pose_dict_to_cam
from visualize_uav_relay_overlap import (
    INTRINSIC,
    draw_covis_red_box,
    overlap_grids,
    render_topdown,
)

ROOT = Path(__file__).resolve().parent.parent
frames = load_frames(ROOT / "samples/_overlap_tmp/frames.jsonl")
uav_pose = frames[0]["uav_camera_poses_carla"]["front_rgb"]
relay_pose = frames[37]["relay_camera_poses_carla"]["relay_01"]
zg = float(frames[0]["ugv_pose_carla"]["z"])
metrics = overlap_on_ground(uav_pose, relay_pose, INTRINSIC, zg)
xs, ys, ma, mb, inter = overlap_grids(uav_pose, relay_pose, zg)
Cu, Ru = pose_dict_to_cam(uav_pose)
Cr, Rr = pose_dict_to_cam(relay_pose)

uav_bgr = cv2.imread(
    str(ROOT / "samples/paper_eval_l2_sem_rich_20260527_172625_preview/uav/front_rgb/000000.png")
)
relay_bgr = cv2.imread(
    str(
        ROOT
        / "samples/paper_eval_l2_sem_rich_20260527_172625_preview/relay/relay_01/rgb/000037.png"
    )
)
out = ROOT / "samples/uav_relay_overlap_viz/user_172625_uav0_relay37"
out.mkdir(parents=True, exist_ok=True)

title = f"172625 uav@0 + relay@37 | IoU={metrics['iou_ground'] * 100:.1f}%"
uav_ov, uav_box = draw_covis_red_box(uav_bgr, xs, ys, inter, zg, Cu, Ru, "covis bbox (UAV)")
relay_ov, relay_box = draw_covis_red_box(relay_bgr, xs, ys, inter, zg, Cr, Rr, "covis bbox (Relay)")
topdown = render_topdown(xs, ys, ma, mb, inter, uav_pose, relay_pose, title, metrics)

uav_s = cv2.resize(uav_ov, (640, 360))
relay_s = cv2.resize(relay_ov, (640, 360))
h = max(topdown.shape[0], 360)

def pad(img: np.ndarray, height: int) -> np.ndarray:
    if img.shape[0] == height:
        return img
    out = np.full((height, img.shape[1], 3), 30, dtype=np.uint8)
    out[: img.shape[0], : img.shape[1]] = img
    return out

row1 = np.hstack([pad(topdown, h), pad(uav_s, h)])
row2 = np.hstack([cv2.resize(topdown, (topdown.shape[1], 360)), relay_s])
composite = np.vstack([row1, row2])

cv2.imwrite(str(out / "composite.png"), composite)
cv2.imwrite(str(out / "topdown.png"), topdown)
cv2.imwrite(str(out / "uav_overlap.png"), uav_ov)
cv2.imwrite(str(out / "relay_overlap.png"), relay_ov)
with (out / "meta.json").open("w", encoding="utf-8") as f:
    json.dump(
        {
            "uav_image": "172625_preview/uav/front_rgb/000000.png",
            "relay_image": "172625_preview/relay/relay_01/rgb/000037.png",
            "note": "172625 deleted on server; poses proxy from 155623 (relay_01 fixed).",
            "metrics": metrics,
            "image_bbox_xyxy": {"uav": uav_box, "relay": relay_box},
        },
        f,
        indent=2,
    )
print("saved", out)
