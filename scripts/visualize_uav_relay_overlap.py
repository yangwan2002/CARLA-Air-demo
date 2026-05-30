"""
Visualize UAV vs relay ground overlap for two example frames.

Outputs under samples/uav_relay_overlap_viz/:
  ex1_high_iou_*  (frame 300, relay_05)
  ex2_low_iou_*   (frame 50,  relay_01)
"""
from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np

from compute_uav_relay_overlap import (
    fov_v_from_h,
    ground_visible_mask,
    load_frames,
    pose_dict_to_cam,
)

SCRIPT_DIR = Path(__file__).resolve().parent
COLLECT_ROOT = SCRIPT_DIR.parent
OUT_DIR = COLLECT_ROOT / "samples" / "uav_relay_overlap_viz"
JSONL = COLLECT_ROOT / "samples" / "_overlap_tmp" / "frames.jsonl"
SEQ_REMOTE = (
    "/root/autodl-tmp/CARLA-Air-demo/AirGroundRelay-Sim/sequences/"
    "paper_eval_l2_sem_rich_20260528_155623"
)
INTRINSIC = {"width": 1920, "height": 1080, "fov_deg": 90.0, "fx": 960.0, "fy": 960.0, "cx": 960.0, "cy": 540.0}
RED_BGR = (0, 0, 255)  # OpenCV BGR — red box like detection overlays
BOX_THICKNESS = 3

EXAMPLES = [
    {
        "tag": "ex1_high_iou",
        "title": "Example 1: high overlap (t=60.2s)",
        "frame_id": 300,
        "relay": "relay_05",
    },
    {
        "tag": "ex2_low_iou",
        "title": "Example 2: low overlap (t=10.2s)",
        "frame_id": 50,
        "relay": "relay_01",
    },
]


def load_env(path: Path) -> None:
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip())


def overlap_grids(
    uav_pose: Dict[str, float],
    relay_pose: Dict[str, float],
    z_ground: float,
    grid_step_m: float = 0.5,
    margin_m: float = 80.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    w, h = int(INTRINSIC["width"]), int(INTRINSIC["height"])
    fov = float(INTRINSIC["fov_deg"])
    Cu, Ru = pose_dict_to_cam(uav_pose)
    Cr, Rr = pose_dict_to_cam(relay_pose)
    cx = (Cu[0] + Cr[0]) / 2.0
    cy = (Cu[1] + Cr[1]) / 2.0
    xs = np.arange(cx - margin_m, cx + margin_m + grid_step_m, grid_step_m)
    ys = np.arange(cy - margin_m, cy + margin_m + grid_step_m, grid_step_m)
    gx, gy = np.meshgrid(xs, ys)
    grid = np.column_stack([gx.ravel(), gy.ravel()])
    ma = ground_visible_mask(grid, z_ground, Cu, Ru, fov, w, h).reshape(gx.shape)
    mb = ground_visible_mask(grid, z_ground, Cr, Rr, fov, w, h).reshape(gx.shape)
    inter = ma & mb
    return xs, ys, ma, mb, inter


def world_to_uv(
    xy: np.ndarray, z_ground: float, C: np.ndarray, R: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    n = xy.shape[0]
    P = np.column_stack([xy[:, 0], xy[:, 1], np.full(n, z_ground)])
    rel = (P - C) @ R
    ok = rel[:, 0] > 0.1
    u = INTRINSIC["fx"] * rel[:, 1] / rel[:, 0] + INTRINSIC["cx"]
    v = INTRINSIC["cy"] - INTRINSIC["fy"] * rel[:, 2] / rel[:, 0]
    return np.column_stack([u, v]), ok


def collect_overlap_uv(
    xs: np.ndarray,
    ys: np.ndarray,
    mask: np.ndarray,
    z_ground: float,
    C: np.ndarray,
    R: np.ndarray,
    w_img: int,
    h_img: int,
) -> np.ndarray:
    """Project overlap ground cells into this camera; return Nx2 pixel coords."""
    pts: List[List[float]] = []
    for j in range(len(ys)):
        for i in range(len(xs)):
            if not mask[j, i]:
                continue
            xy = np.array([[xs[i], ys[j]]], dtype=np.float64)
            uv, ok = world_to_uv(xy, z_ground, C, R)
            if not ok[0]:
                continue
            u, v = float(uv[0, 0]), float(uv[0, 1])
            if 0 <= u < w_img and 0 <= v < h_img:
                pts.append([u, v])
    if not pts:
        return np.zeros((0, 2), dtype=np.float32)
    return np.array(pts, dtype=np.float32)


def uv_axis_aligned_bbox(uv: np.ndarray, w_img: int, h_img: int) -> Tuple[int, int, int, int] | None:
    if uv.shape[0] == 0:
        return None
    x0, y0 = int(np.floor(uv[:, 0].min())), int(np.floor(uv[:, 1].min()))
    x1, y1 = int(np.ceil(uv[:, 0].max())), int(np.ceil(uv[:, 1].max()))
    x0 = max(0, x0)
    y0 = max(0, y0)
    x1 = min(w_img - 1, x1)
    y1 = min(h_img - 1, y1)
    if x1 <= x0 or y1 <= y0:
        return None
    return x0, y0, x1, y1


def draw_covis_red_box(
    bgr: np.ndarray,
    xs: np.ndarray,
    ys: np.ndarray,
    mask: np.ndarray,
    z_ground: float,
    C: np.ndarray,
    R: np.ndarray,
    label: str,
) -> Tuple[np.ndarray, Tuple[int, int, int, int] | None]:
    """Draw axis-aligned red rectangle around projected overlap (det-box style)."""
    h_img, w_img = bgr.shape[:2]
    out = bgr.copy()
    uv = collect_overlap_uv(xs, ys, mask, z_ground, C, R, w_img, h_img)
    bbox = uv_axis_aligned_bbox(uv, w_img, h_img)
    if bbox is not None:
        x0, y0, x1, y1 = bbox
        cv2.rectangle(out, (x0, y0), (x1, y1), RED_BGR, BOX_THICKNESS, cv2.LINE_AA)
    cv2.putText(
        out,
        label,
        (16, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (255, 255, 255),
        3,
        cv2.LINE_AA,
    )
    cv2.putText(
        out,
        label,
        (16, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        RED_BGR,
        2,
        cv2.LINE_AA,
    )
    return out, bbox


def overlap_world_bbox(
    xs: np.ndarray, ys: np.ndarray, inter: np.ndarray
) -> Tuple[float, float, float, float] | None:
    pts = []
    for j in range(len(ys)):
        for i in range(len(xs)):
            if inter[j, i]:
                pts.append((float(xs[i]), float(ys[j])))
    if not pts:
        return None
    arr = np.array(pts)
    return float(arr[:, 0].min()), float(arr[:, 1].min()), float(arr[:, 0].max()), float(arr[:, 1].max())


def render_topdown(
    xs: np.ndarray,
    ys: np.ndarray,
    ma: np.ndarray,
    mb: np.ndarray,
    inter: np.ndarray,
    uav_pose: Dict[str, float],
    relay_pose: Dict[str, float],
    title: str,
    metrics: Dict[str, float],
) -> np.ndarray:
    size = 720
    x0, x1 = float(xs[0]), float(xs[-1])
    y0, y1 = float(ys[0]), float(ys[-1])
    pad = 8

    def to_px(x: float, y: float) -> Tuple[int, int]:
        px = int((x - x0) / (x1 - x0 + 1e-9) * (size - 2 * pad) + pad)
        py = int((y1 - y) / (y1 - y0 + 1e-9) * (size - 2 * pad) + pad)
        return px, py

    canvas = np.full((size, size, 3), 245, dtype=np.uint8)

    # Faint footprint outlines (boundary only, no fill)
    for mask, color in ((ma, (200, 160, 120)), (mb, (120, 160, 200))):
        ys_idx, xs_idx = np.where(mask)
        for j, i in zip(ys_idx.tolist(), xs_idx.tolist()):
            if j > 0 and mask[j - 1, i] and j < len(ys) - 1 and mask[j + 1, i]:
                continue
            if i > 0 and mask[j, i - 1] and i < len(xs) - 1 and mask[j, i + 1]:
                continue
            p = to_px(xs[i], ys[j])
            cv2.circle(canvas, p, 1, color, -1)

    wb = overlap_world_bbox(xs, ys, inter)
    if wb is not None:
        xmin, ymin, xmax, ymax = wb
        p0 = to_px(xmin, ymax)
        p1 = to_px(xmax, ymin)
        cv2.rectangle(canvas, p0, p1, RED_BGR, BOX_THICKNESS, cv2.LINE_AA)

    for name, pose, color in (
        ("UAV", uav_pose, (255, 120, 0)),
        ("Relay", relay_pose, (200, 80, 0)),
    ):
        px, py = to_px(pose["x"], pose["y"])
        cv2.circle(canvas, (px, py), 7, color, -1)
        cv2.putText(canvas, name, (px + 10, py - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

    legend = (
        f"{title} | ground IoU={metrics['iou_ground']*100:.1f}% | "
        f"overlap {metrics['overlap_ground_m2']:.0f} m^2"
    )
    cv2.rectangle(canvas, (0, 0), (size - 1, 78), (40, 40, 40), -1)
    cv2.putText(canvas, legend, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
    cv2.putText(
        canvas,
        "red box = covis ground footprint (same as image boxes)",
        (10, 56),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        (180, 180, 255),
        1,
    )
    return canvas


def fetch_images(frame_id: int, relay: str, local_dir: Path) -> Tuple[Path, Path]:
    import paramiko

    load_env(COLLECT_ROOT / ".env")
    local_dir.mkdir(parents=True, exist_ok=True)
    fid = f"{frame_id:06d}"
    uav_local = local_dir / f"uav_{fid}.png"
    relay_local = local_dir / f"{relay}_{fid}.png"
    if uav_local.exists() and relay_local.exists():
        return uav_local, relay_local

    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(
        os.environ["SSH_HOST"],
        int(os.environ["SSH_PORT"]),
        os.environ["SSH_USER"],
        os.environ["SSH_PASSWORD"],
        timeout=60,
    )
    sftp = client.open_sftp()
    sftp.get(f"{SEQ_REMOTE}/uav/front_rgb/{fid}.png", str(uav_local))
    sftp.get(f"{SEQ_REMOTE}/relay/{relay}/rgb/{fid}.png", str(relay_local))
    sftp.close()
    client.close()
    return uav_local, relay_local


def process_example(frames: List[Dict[str, Any]], ex: Dict[str, Any]) -> None:
    from compute_uav_relay_overlap import overlap_on_ground

    fid = ex["frame_id"]
    relay = ex["relay"]
    fr = frames[fid]
    zg = float(fr["ugv_pose_carla"]["z"])
    uav_pose = fr["uav_camera_poses_carla"]["front_rgb"]
    relay_pose = fr["relay_camera_poses_carla"][relay]
    metrics = overlap_on_ground(uav_pose, relay_pose, INTRINSIC, zg)

    xs, ys, ma, mb, inter = overlap_grids(uav_pose, relay_pose, zg)
    Cu, Ru = pose_dict_to_cam(uav_pose)
    Cr, Rr = pose_dict_to_cam(relay_pose)

    img_dir = OUT_DIR / ex["tag"]
    uav_path, relay_path = fetch_images(fid, relay, img_dir)
    uav_bgr = cv2.imread(str(uav_path))
    relay_bgr = cv2.imread(str(relay_path))

    uav_ov, uav_box = draw_covis_red_box(
        uav_bgr, xs, ys, inter, zg, Cu, Ru, "covis bbox (UAV)"
    )
    relay_ov, relay_box = draw_covis_red_box(
        relay_bgr, xs, ys, inter, zg, Cr, Rr, "covis bbox (Relay)"
    )
    topdown = render_topdown(xs, ys, ma, mb, inter, uav_pose, relay_pose, ex["title"], metrics)

    # side-by-side composite
    h = max(uav_ov.shape[0], relay_ov.shape[0], topdown.shape[0])
    def pad(img: np.ndarray) -> np.ndarray:
        if img.shape[0] == h:
            return img
        out = np.full((h, img.shape[1], 3), 30, dtype=np.uint8)
        out[: img.shape[0], : img.shape[1]] = img
        return out

    row1 = np.hstack([pad(topdown), pad(cv2.resize(uav_ov, (640, 360)))])
    row2_img = pad(cv2.resize(relay_ov, (row1.shape[1], 360)))
    composite = np.vstack([row1, row2_img])

    tag = ex["tag"]
    cv2.imwrite(str(OUT_DIR / f"{tag}_topdown.png"), topdown)
    cv2.imwrite(str(OUT_DIR / f"{tag}_uav_overlap.png"), uav_ov)
    cv2.imwrite(str(OUT_DIR / f"{tag}_relay_overlap.png"), relay_ov)
    cv2.imwrite(str(OUT_DIR / f"{tag}_composite.png"), composite)

    meta = {
        "tag": tag,
        "title": ex["title"],
        "frame_id": fid,
        "sim_time": fr["sim_time"],
        "relay": relay,
        "metrics": metrics,
        "image_bbox_xyxy": {"uav": uav_box, "relay": relay_box},
        "files": {
            "topdown": str(OUT_DIR / f"{tag}_topdown.png"),
            "uav": str(OUT_DIR / f"{tag}_uav_overlap.png"),
            "relay": str(OUT_DIR / f"{tag}_relay_overlap.png"),
            "composite": str(OUT_DIR / f"{tag}_composite.png"),
        },
    }
    with (OUT_DIR / f"{tag}_meta.json").open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(f"Wrote {tag} -> {OUT_DIR}")


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    frames = load_frames(JSONL)
    for ex in EXAMPLES:
        process_example(frames, ex)
    print(f"\nAll outputs: {OUT_DIR.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
