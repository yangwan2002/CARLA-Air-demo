"""
Estimate UAV <-> relay ground-plane footprint overlap from frames.jsonl poses.

Method: sample a 2D ground grid; a cell is visible in a camera if the world
point lies in front of the optical axis and inside the horizontal/vertical FOV.
Overlap metrics are reported vs each camera's visible ground area.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np

# CARLA / UE4 camera: +X forward, +Y right, +Z up.
# Pixel (u,v) -> unit ray in camera frame (x forward, y right, z up; v down).
def pixel_to_ray_cam(u: float, v: float, fx: float, fy: float, cx: float, cy: float) -> np.ndarray:
    x = 1.0
    y = (u - cx) / fx
    z = -(v - cy) / fy
    d = np.array([x, y, z], dtype=np.float64)
    return d / np.linalg.norm(d)


def rpy_deg_to_R(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """CARLA Rotation (roll,pitch,yaw in degrees) -> world rotation matrix.

    CARLA/UE4 pitch sign is opposite to our Ry convention; negate pitch so
    pitch=-90 yields optical axis [0,0,-1] (nadir).
    """
    r, p, y = map(math.radians, (roll, -pitch, yaw))
    cr, sr = math.cos(r), math.sin(r)
    cp, sp = math.cos(p), math.sin(p)
    cy, sy = math.cos(y), math.sin(y)
    # Rz(yaw) @ Ry(pitch) @ Rx(roll)
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    return Rz @ Ry @ Rx


def pose_dict_to_cam(pose: Dict[str, float]) -> Tuple[np.ndarray, np.ndarray]:
    C = np.array([pose["x"], pose["y"], pose["z"]], dtype=np.float64)
    R = rpy_deg_to_R(pose["roll"], pose["pitch"], pose["yaw"])
    return C, R


def fov_v_from_h(fov_h_deg: float, width: int, height: int) -> float:
    fh = math.radians(fov_h_deg)
    fv = 2.0 * math.atan(math.tan(fh / 2.0) * height / width)
    return math.degrees(fv)


def ground_visible_mask(
    grid_xy: np.ndarray,
    z_ground: float,
    C: np.ndarray,
    R: np.ndarray,
    fov_h_deg: float,
    width: int,
    height: int,
) -> np.ndarray:
    """Return bool mask (N,) for grid points visible from camera."""
    n = grid_xy.shape[0]
    P = np.column_stack([grid_xy[:, 0], grid_xy[:, 1], np.full(n, z_ground)])
    rel = (P - C) @ R  # world -> camera (R columns are camera axes in world)
    x, y, z = rel[:, 0], rel[:, 1], rel[:, 2]
    in_front = x > 0.1
    horiz = np.degrees(np.arctan2(np.abs(y), x))
    vert = np.degrees(np.arctan2(np.abs(z), x))
    fov_v = fov_v_from_h(fov_h_deg, width, height)
    return in_front & (horiz <= fov_h_deg / 2.0 + 1e-6) & (vert <= fov_v / 2.0 + 1e-6)


def overlap_on_ground(
    pose_a: Dict[str, float],
    pose_b: Dict[str, float],
    intrinsic: Dict[str, float],
    z_ground: float,
    grid_step_m: float = 0.5,
    margin_m: float = 80.0,
) -> Dict[str, float]:
    w, h = int(intrinsic["width"]), int(intrinsic["height"])
    fov = float(intrinsic["fov_deg"])
    Ca, Ra = pose_dict_to_cam(pose_a)
    Cb, Rb = pose_dict_to_cam(pose_b)

    cx = (Ca[0] + Cb[0]) / 2.0
    cy = (Ca[1] + Cb[1]) / 2.0
    xs = np.arange(cx - margin_m, cx + margin_m + grid_step_m, grid_step_m)
    ys = np.arange(cy - margin_m, cy + margin_m + grid_step_m, grid_step_m)
    gx, gy = np.meshgrid(xs, ys)
    grid = np.column_stack([gx.ravel(), gy.ravel()])

    ma = ground_visible_mask(grid, z_ground, Ca, Ra, fov, w, h)
    mb = ground_visible_mask(grid, z_ground, Cb, Rb, fov, w, h)
    inter = ma & mb
    union = ma | mb
    area_a = float(ma.sum()) * grid_step_m**2
    area_b = float(mb.sum()) * grid_step_m**2
    area_i = float(inter.sum()) * grid_step_m**2
    area_u = float(union.sum()) * grid_step_m**2
    return {
        "ground_z_m": z_ground,
        "grid_step_m": grid_step_m,
        "area_uav_ground_m2": area_a,
        "area_relay_ground_m2": area_b,
        "overlap_ground_m2": area_i,
        "union_ground_m2": area_u,
        "iou_ground": area_i / area_u if area_u > 0 else 0.0,
        "overlap_vs_uav": area_i / area_a if area_a > 0 else 0.0,
        "overlap_vs_relay": area_i / area_b if area_b > 0 else 0.0,
    }


def load_frames(path: Path) -> List[Dict[str, Any]]:
    frames = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                frames.append(json.loads(line))
    return frames


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--jsonl",
        type=Path,
        default=Path(__file__).resolve().parent.parent
        / "samples/_overlap_tmp/frames.jsonl",
    )
    parser.add_argument("--frame-a", type=int, default=250)
    parser.add_argument("--frame-b", type=int, default=250)
    parser.add_argument("--relay-a", default="relay_05")
    parser.add_argument("--relay-b", default="relay_03")
    parser.add_argument("--scan-relay", default="relay_05", help="find max IoU frame for this relay")
    args = parser.parse_args()

    intrinsic = {
        "width": 1920,
        "height": 1080,
        "fov_deg": 90.0,
    }
    frames = load_frames(args.jsonl)

    def z_ground_at(fr: Dict[str, Any]) -> float:
        return float(fr["ugv_pose_carla"]["z"])

    # Example 1 & 2 at user-specified frames
    examples = [
        ("例1", args.frame_a, args.relay_a),
        ("例2", args.frame_b, args.relay_b),
    ]
    print(f"序列: {frames[0]['sequence']}, 共 {len(frames)} 帧\n")
    print("=" * 72)
    print("UAV (front_rgb) vs Relay — 地面采样重叠（平面近似 z=UGV 高度）")
    print("=" * 72)

    for label, fid, relay in examples:
        if fid >= len(frames):
            print(f"{label}: frame {fid} 超出范围")
            continue
        fr = frames[fid]
        uav_pose = fr["uav_camera_poses_carla"]["front_rgb"]
        relay_pose = fr["relay_camera_poses_carla"][relay]
        zg = z_ground_at(fr)
        m = overlap_on_ground(uav_pose, relay_pose, intrinsic, zg)
        dist = math.hypot(
            uav_pose["x"] - relay_pose["x"],
            uav_pose["y"] - relay_pose["y"],
        )
        print(f"\n{label}: frame_id={fid}, sim_time={fr['sim_time']:.1f}s, relay={relay}")
        print(f"  UAV 相机: ({uav_pose['x']:.1f}, {uav_pose['y']:.1f}, {uav_pose['z']:.1f}) m, pitch={uav_pose['pitch']:.1f}°")
        print(f"  Relay 相机: ({relay_pose['x']:.1f}, {relay_pose['y']:.1f}, {relay_pose['z']:.1f}) m, pitch={relay_pose['pitch']:.1f}°")
        print(f"  水平距离 UAV-Relay: {dist:.1f} m")
        print(f"  地面高度假设 z={zg:.2f} m")
        print(f"  UAV 地面可视面积: {m['area_uav_ground_m2']:.0f} m^2")
        print(f"  Relay 地面可视面积: {m['area_relay_ground_m2']:.0f} m^2")
        print(f"  重叠地面面积: {m['overlap_ground_m2']:.0f} m^2")
        print(f"  IoU (地面): {100*m['iou_ground']:.1f}%")
        print(f"  重叠占 UAV 地面视野: {100*m['overlap_vs_uav']:.1f}%")
        print(f"  重叠占 Relay 地面视野: {100*m['overlap_vs_relay']:.1f}%")

    # Scan for best overlap frame on palm-street relay
    relay = args.scan_relay
    best_iou, best_fid = -1.0, -1
    for fid, fr in enumerate(frames):
        if relay not in fr["relay_camera_poses_carla"]:
            continue
        uav_pose = fr["uav_camera_poses_carla"]["front_rgb"]
        relay_pose = fr["relay_camera_poses_carla"][relay]
        m = overlap_on_ground(uav_pose, relay_pose, intrinsic, z_ground_at(fr), grid_step_m=1.0)
        if m["iou_ground"] > best_iou:
            best_iou, best_fid = m["iou_ground"], fid

    if best_fid >= 0:
        fr = frames[best_fid]
        m = overlap_on_ground(
            fr["uav_camera_poses_carla"]["front_rgb"],
            fr["relay_camera_poses_carla"][relay],
            intrinsic,
            z_ground_at(fr),
        )
        print(f"\n--- 全序列扫描: UAV vs {relay} 地面 IoU 最大帧 ---")
        print(f"  frame_id={best_fid}, sim_time={fr['sim_time']:.1f}s, IoU={100*m['iou_ground']:.1f}%")
        print(f"  重叠占 UAV: {100*m['overlap_vs_uav']:.1f}%, 占 Relay: {100*m['overlap_vs_relay']:.1f}%")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
