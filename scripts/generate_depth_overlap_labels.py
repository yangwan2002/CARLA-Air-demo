"""
Generate pairwise overlap labels from CARLA RGB-D frames by depth reprojection.

This script avoids feature matching. It reconstructs 3D points from one view's
depth image, reprojects them into the other view, validates visibility against
the target depth, and outputs per-image overlap masks, bounding boxes, and
debug visualizations.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import cv2
import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
COLLECT_ROOT = SCRIPT_DIR.parent


def load_frames(path: Path) -> List[Dict[str, Any]]:
    frames: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                frames.append(json.loads(line))
    return frames


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def pose_dict_to_cam_axes(pose: Dict[str, float]) -> Tuple[np.ndarray, np.ndarray]:
    """Return camera origin and axes-in-world rotation matrix.

    Coordinate convention follows the collector repo:
      camera x = forward, y = right, z = up
    """
    roll, pitch, yaw = pose["roll"], -pose["pitch"], pose["yaw"]
    r, p, y = map(math.radians, (roll, pitch, yaw))
    cr, sr = math.cos(r), math.sin(r)
    cp, sp = math.cos(p), math.sin(p)
    cy, sy = math.cos(y), math.sin(y)
    rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]], dtype=np.float64)
    ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]], dtype=np.float64)
    rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]], dtype=np.float64)
    return (
        np.array([pose["x"], pose["y"], pose["z"]], dtype=np.float64),
        rz @ ry @ rx,
    )


def pixel_rays(width: int, height: int, fx: float, fy: float, cx: float, cy: float) -> np.ndarray:
    """Return normalized camera rays for all pixels."""
    us, vs = np.meshgrid(np.arange(width, dtype=np.float32), np.arange(height, dtype=np.float32))
    x = np.ones_like(us, dtype=np.float32)
    y = (us - cx) / fx
    z = -(vs - cy) / fy
    rays = np.stack([x, y, z], axis=-1)
    norms = np.linalg.norm(rays, axis=-1, keepdims=True)
    return rays / np.maximum(norms, 1e-8)


def decode_depth_png(path: Path, depth_max_m: float) -> np.ndarray:
    raw = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if raw is None:
        raise FileNotFoundError(f"Failed to read depth image: {path}")
    if raw.dtype != np.uint16:
        raise ValueError(f"Expected uint16 depth PNG at {path}, got {raw.dtype}")
    return (raw.astype(np.float32) / 65535.0) * float(depth_max_m)


def resolve_image_path(seq_dir: Path, rel_path: str) -> Path:
    path = seq_dir / rel_path
    if path.exists():
        return path
    alt_png = path.with_suffix(".png")
    if alt_png.exists():
        return alt_png
    alt_npy = path.with_suffix(".npy")
    if alt_npy.exists():
        return alt_npy
    raise FileNotFoundError(f"Missing file for {rel_path} under {seq_dir}")


def get_intrinsic(seq_dir: Path, sensor_name: str) -> Dict[str, float]:
    if sensor_name == "uav_front":
        return load_json(seq_dir / "calibration" / "uav_intrinsics.json")["intrinsic"]
    if sensor_name == "ugv_front":
        return load_json(seq_dir / "calibration" / "ugv_intrinsics.json")["intrinsic"]
    if sensor_name.startswith("relay_"):
        relay_cfg = load_json(seq_dir / "calibration" / "relay_intrinsics.json")["cameras"]
        return relay_cfg[sensor_name]["intrinsic"]
    raise KeyError(f"Unsupported sensor: {sensor_name}")


def get_sensor_pose(frame: Dict[str, Any], sensor_name: str) -> Dict[str, float]:
    if sensor_name == "uav_front":
        return frame["uav_camera_poses_carla"]["front_rgb"]
    if sensor_name == "ugv_front":
        return frame["ugv_camera_pose_carla"]
    if sensor_name.startswith("relay_"):
        return frame["relay_camera_poses_carla"][sensor_name]
    raise KeyError(f"Unsupported sensor: {sensor_name}")


def get_sensor_image_key(sensor_name: str) -> Tuple[str, str]:
    if sensor_name == "uav_front":
        return "uav_front_rgb", "uav_front_depth"
    if sensor_name == "ugv_front":
        return "ugv_front_rgb", "ugv_front_depth"
    if sensor_name.startswith("relay_"):
        return f"{sensor_name}_rgb", f"{sensor_name}_depth"
    raise KeyError(f"Unsupported sensor: {sensor_name}")


def project_cam_points(cam_points: np.ndarray, intrinsic: Dict[str, float]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    x = cam_points[:, 0]
    y = cam_points[:, 1]
    z = cam_points[:, 2]
    valid = x > 1e-3
    u = intrinsic["fx"] * y / np.maximum(x, 1e-8) + intrinsic["cx"]
    v = intrinsic["cy"] - intrinsic["fy"] * z / np.maximum(x, 1e-8)
    rng = np.linalg.norm(cam_points, axis=1)
    return u, v, rng, valid


def bbox_from_mask(mask: np.ndarray) -> Optional[List[int]]:
    ys, xs = np.where(mask)
    if xs.size == 0:
        return None
    return [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]


def overlay_mask_and_box(image: np.ndarray, mask: np.ndarray, bbox: Optional[List[int]], title: str) -> np.ndarray:
    out = image.copy()
    if mask.any():
        tint = np.zeros_like(out)
        tint[..., 2] = 255
        blended = cv2.addWeighted(out, 0.72, tint, 0.28, 0.0)
        out[mask] = blended[mask]
    if bbox is not None:
        x0, y0, x1, y1 = bbox
        cv2.rectangle(out, (x0, y0), (x1, y1), (0, 0, 255), 2, cv2.LINE_AA)
    cv2.putText(out, title, (18, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 3, cv2.LINE_AA)
    cv2.putText(out, title, (18, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 1, cv2.LINE_AA)
    return out


def overlap_from_source_to_target(
    depth_src: np.ndarray,
    intrinsic_src: Dict[str, float],
    pose_src: Dict[str, float],
    depth_tgt: np.ndarray,
    intrinsic_tgt: Dict[str, float],
    pose_tgt: Dict[str, float],
    ray_cache: Dict[Tuple[int, int, float, float, float, float], np.ndarray],
    min_depth_m: float,
    max_depth_m: float,
    depth_tol_m: float,
    depth_tol_ratio: float,
) -> Tuple[np.ndarray, np.ndarray]:
    h, w = depth_src.shape
    key = (w, h, intrinsic_src["fx"], intrinsic_src["fy"], intrinsic_src["cx"], intrinsic_src["cy"])
    rays = ray_cache.get(key)
    if rays is None:
        rays = pixel_rays(w, h, intrinsic_src["fx"], intrinsic_src["fy"], intrinsic_src["cx"], intrinsic_src["cy"])
        ray_cache[key] = rays

    valid_src = np.isfinite(depth_src) & (depth_src > min_depth_m) & (depth_src < max_depth_m)
    if not np.any(valid_src):
        return np.zeros_like(valid_src, dtype=bool), np.zeros_like(depth_tgt, dtype=bool)

    C_src, R_src = pose_dict_to_cam_axes(pose_src)
    C_tgt, R_tgt = pose_dict_to_cam_axes(pose_tgt)

    pix_v, pix_u = np.where(valid_src)
    src_depth_vals = depth_src[pix_v, pix_u].astype(np.float32)
    src_rays = rays[pix_v, pix_u]
    cam_points_src = src_rays * src_depth_vals[:, None]
    world_points = cam_points_src @ R_src.T + C_src[None, :]
    cam_points_tgt = (world_points - C_tgt[None, :]) @ R_tgt

    u_tgt, v_tgt, pred_range_tgt, in_front = project_cam_points(cam_points_tgt, intrinsic_tgt)
    finite = np.isfinite(u_tgt) & np.isfinite(v_tgt) & np.isfinite(pred_range_tgt)
    castable = finite & (np.abs(u_tgt) < (2**30)) & (np.abs(v_tgt) < (2**30))
    u_int = np.zeros_like(u_tgt, dtype=np.int32)
    v_int = np.zeros_like(v_tgt, dtype=np.int32)
    u_int[castable] = np.rint(u_tgt[castable]).astype(np.int32)
    v_int[castable] = np.rint(v_tgt[castable]).astype(np.int32)
    inside = (
        castable
        & in_front
        & (u_int >= 0)
        & (u_int < depth_tgt.shape[1])
        & (v_int >= 0)
        & (v_int < depth_tgt.shape[0])
    )
    if not np.any(inside):
        return np.zeros_like(valid_src, dtype=bool), np.zeros_like(depth_tgt, dtype=bool)

    src_u_kept = pix_u[inside]
    src_v_kept = pix_v[inside]
    tgt_u_kept = u_int[inside]
    tgt_v_kept = v_int[inside]
    pred_kept = pred_range_tgt[inside]
    tgt_depth_vals = depth_tgt[tgt_v_kept, tgt_u_kept]
    tgt_depth_ok = np.isfinite(tgt_depth_vals) & (tgt_depth_vals > min_depth_m) & (tgt_depth_vals < max_depth_m)
    tol = np.maximum(depth_tol_m, depth_tol_ratio * tgt_depth_vals)
    consistent = tgt_depth_ok & (np.abs(pred_kept - tgt_depth_vals) <= tol)

    mask_src = np.zeros_like(valid_src, dtype=bool)
    mask_tgt = np.zeros_like(depth_tgt, dtype=bool)
    if np.any(consistent):
        mask_src[src_v_kept[consistent], src_u_kept[consistent]] = True
        mask_tgt[tgt_v_kept[consistent], tgt_u_kept[consistent]] = True
    return mask_src, mask_tgt


def pair_overlap(
    seq_dir: Path,
    frame: Dict[str, Any],
    src_sensor: str,
    tgt_sensor: str,
    depth_max_m: float,
    min_depth_m: float,
    depth_tol_m: float,
    depth_tol_ratio: float,
    ray_cache: Dict[Tuple[int, int, float, float, float, float], np.ndarray],
) -> Dict[str, Any]:
    src_rgb_key, src_depth_key = get_sensor_image_key(src_sensor)
    tgt_rgb_key, tgt_depth_key = get_sensor_image_key(tgt_sensor)
    src_img_path = resolve_image_path(seq_dir, frame["images"][src_rgb_key])
    src_depth_path = resolve_image_path(seq_dir, frame["images"][src_depth_key])
    tgt_img_path = resolve_image_path(seq_dir, frame["images"][tgt_rgb_key])
    tgt_depth_path = resolve_image_path(seq_dir, frame["images"][tgt_depth_key])

    src_rgb = cv2.imread(str(src_img_path), cv2.IMREAD_COLOR)
    tgt_rgb = cv2.imread(str(tgt_img_path), cv2.IMREAD_COLOR)
    if src_rgb is None or tgt_rgb is None:
        raise FileNotFoundError("Failed to load RGB image for pair")

    src_depth = decode_depth_png(src_depth_path, depth_max_m)
    tgt_depth = decode_depth_png(tgt_depth_path, depth_max_m)
    intrinsic_src = get_intrinsic(seq_dir, src_sensor)
    intrinsic_tgt = get_intrinsic(seq_dir, tgt_sensor)
    pose_src = get_sensor_pose(frame, src_sensor)
    pose_tgt = get_sensor_pose(frame, tgt_sensor)

    src_mask, src_proj_on_tgt = overlap_from_source_to_target(
        src_depth,
        intrinsic_src,
        pose_src,
        tgt_depth,
        intrinsic_tgt,
        pose_tgt,
        ray_cache,
        min_depth_m=min_depth_m,
        max_depth_m=depth_max_m,
        depth_tol_m=depth_tol_m,
        depth_tol_ratio=depth_tol_ratio,
    )
    tgt_mask, tgt_proj_on_src = overlap_from_source_to_target(
        tgt_depth,
        intrinsic_tgt,
        pose_tgt,
        src_depth,
        intrinsic_src,
        pose_src,
        ray_cache,
        min_depth_m=min_depth_m,
        max_depth_m=depth_max_m,
        depth_tol_m=depth_tol_m,
        depth_tol_ratio=depth_tol_ratio,
    )

    src_bbox = bbox_from_mask(src_mask)
    tgt_bbox = bbox_from_mask(tgt_mask)
    valid_src_pixels = int(np.count_nonzero((src_depth > min_depth_m) & (src_depth < depth_max_m)))
    valid_tgt_pixels = int(np.count_nonzero((tgt_depth > min_depth_m) & (tgt_depth < depth_max_m)))
    overlap_src_pixels = int(np.count_nonzero(src_mask))
    overlap_tgt_pixels = int(np.count_nonzero(tgt_mask))

    return {
        "frame_id": int(frame["frame_id"]),
        "sim_time": float(frame["sim_time"]),
        "src_sensor": src_sensor,
        "tgt_sensor": tgt_sensor,
        "src_rgb_path": str(src_img_path),
        "tgt_rgb_path": str(tgt_img_path),
        "src_mask": src_mask,
        "tgt_mask": tgt_mask,
        "src_bbox": src_bbox,
        "tgt_bbox": tgt_bbox,
        "src_overlap_pixels": overlap_src_pixels,
        "tgt_overlap_pixels": overlap_tgt_pixels,
        "src_overlap_ratio": overlap_src_pixels / valid_src_pixels if valid_src_pixels else 0.0,
        "tgt_overlap_ratio": overlap_tgt_pixels / valid_tgt_pixels if valid_tgt_pixels else 0.0,
        "pair_score": min(
            overlap_src_pixels / valid_src_pixels if valid_src_pixels else 0.0,
            overlap_tgt_pixels / valid_tgt_pixels if valid_tgt_pixels else 0.0,
        ),
        "src_image": src_rgb,
        "tgt_image": tgt_rgb,
        "src_projected_mask_from_target": tgt_proj_on_src,
        "tgt_projected_mask_from_source": src_proj_on_tgt,
    }


def save_visualization(result: Dict[str, Any], out_dir: Path, stem: str) -> Dict[str, str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    src_overlay = overlay_mask_and_box(
        result["src_image"],
        result["src_mask"],
        result["src_bbox"],
        f"{result['src_sensor']} overlap {result['src_overlap_ratio']:.3f}",
    )
    tgt_overlay = overlay_mask_and_box(
        result["tgt_image"],
        result["tgt_mask"],
        result["tgt_bbox"],
        f"{result['tgt_sensor']} overlap {result['tgt_overlap_ratio']:.3f}",
    )
    composite = cv2.hconcat([src_overlay, tgt_overlay])

    src_mask_path = out_dir / f"{stem}_{result['src_sensor']}_mask.png"
    tgt_mask_path = out_dir / f"{stem}_{result['tgt_sensor']}_mask.png"
    src_overlay_path = out_dir / f"{stem}_{result['src_sensor']}_overlay.png"
    tgt_overlay_path = out_dir / f"{stem}_{result['tgt_sensor']}_overlay.png"
    composite_path = out_dir / f"{stem}_composite.png"

    cv2.imwrite(str(src_mask_path), (result["src_mask"].astype(np.uint8) * 255))
    cv2.imwrite(str(tgt_mask_path), (result["tgt_mask"].astype(np.uint8) * 255))
    cv2.imwrite(str(src_overlay_path), src_overlay)
    cv2.imwrite(str(tgt_overlay_path), tgt_overlay)
    cv2.imwrite(str(composite_path), composite)

    return {
        "src_mask": str(src_mask_path),
        "tgt_mask": str(tgt_mask_path),
        "src_overlay": str(src_overlay_path),
        "tgt_overlay": str(tgt_overlay_path),
        "composite": str(composite_path),
    }


def sensor_pairs(relays: Iterable[str]) -> List[Tuple[str, str]]:
    return [("uav_front", relay) for relay in relays]


def list_relays(frame: Dict[str, Any]) -> List[str]:
    return sorted(frame["relay_camera_poses_carla"].keys())


def find_example_pairs(
    seq_dir: Path,
    frames: List[Dict[str, Any]],
    frame_ids: Optional[List[int]],
    relays: Optional[List[str]],
    depth_max_m: float,
    min_depth_m: float,
    depth_tol_m: float,
    depth_tol_ratio: float,
) -> List[Dict[str, Any]]:
    ray_cache: Dict[Tuple[int, int, float, float, float, float], np.ndarray] = {}
    selected_frames = frames if not frame_ids else [frames[idx] for idx in frame_ids]
    relay_names = relays if relays else list_relays(frames[0])
    results: List[Dict[str, Any]] = []
    for frame in selected_frames:
        for src_sensor, tgt_sensor in sensor_pairs(relay_names):
            try:
                results.append(
                    pair_overlap(
                        seq_dir,
                        frame,
                        src_sensor,
                        tgt_sensor,
                        depth_max_m,
                        min_depth_m,
                        depth_tol_m,
                        depth_tol_ratio,
                        ray_cache,
                    )
                )
            except FileNotFoundError:
                continue
    results.sort(key=lambda item: item["pair_score"])
    return results


def choose_low_example(results: List[Dict[str, Any]], min_nonzero: float) -> Dict[str, Any]:
    for item in results:
        if item["pair_score"] >= min_nonzero:
            return item
    return results[0]


def choose_high_example(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    return results[-1]


def write_json(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate overlap labels by depth reprojection.")
    parser.add_argument(
        "--sequence-dir",
        type=Path,
        required=True,
        help="Sequence directory containing frames.jsonl and RGB-D images.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Output directory. Defaults to <sequence-dir>/overlap_depth_labels",
    )
    parser.add_argument("--frame-id", type=int, action="append", default=None, help="Specific frame id(s) to process.")
    parser.add_argument("--relay", action="append", default=None, help="Specific relay camera(s) to process.")
    parser.add_argument("--depth-max-m", type=float, default=80.0, help="Depth PNG clipping distance used during saving.")
    parser.add_argument("--min-depth-m", type=float, default=0.1, help="Minimum valid metric depth.")
    parser.add_argument("--depth-tol-m", type=float, default=1.0, help="Absolute occlusion tolerance in meters.")
    parser.add_argument("--depth-tol-ratio", type=float, default=0.05, help="Relative occlusion tolerance.")
    parser.add_argument(
        "--auto-examples",
        action="store_true",
        help="Scan candidate pairs and write one low-overlap and one high-overlap example.",
    )
    parser.add_argument(
        "--min-nonzero-score",
        type=float,
        default=0.01,
        help="Minimum pair score for the low-overlap example when auto-selecting.",
    )
    args = parser.parse_args()

    seq_dir = args.sequence_dir.resolve()
    out_dir = args.out_dir.resolve() if args.out_dir else (seq_dir / "overlap_depth_labels")
    frames = load_frames(seq_dir / "frames.jsonl")

    results = find_example_pairs(
        seq_dir=seq_dir,
        frames=frames,
        frame_ids=args.frame_id,
        relays=args.relay,
        depth_max_m=args.depth_max_m,
        min_depth_m=args.min_depth_m,
        depth_tol_m=args.depth_tol_m,
        depth_tol_ratio=args.depth_tol_ratio,
    )
    if not results:
        raise RuntimeError("No pair results generated.")

    if args.auto_examples:
        chosen = [
            ("low", choose_low_example(results, args.min_nonzero_score)),
            ("high", choose_high_example(results)),
        ]
    else:
        chosen = [("result", results[-1])]

    summary: Dict[str, Any] = {"sequence_dir": str(seq_dir), "examples": []}
    for tag, result in chosen:
        stem = f"{tag}_frame{result['frame_id']:04d}_{result['src_sensor']}_{result['tgt_sensor']}"
        files = save_visualization(result, out_dir, stem)
        example_meta = {
            "tag": tag,
            "frame_id": result["frame_id"],
            "sim_time": result["sim_time"],
            "src_sensor": result["src_sensor"],
            "tgt_sensor": result["tgt_sensor"],
            "src_bbox_xyxy": result["src_bbox"],
            "tgt_bbox_xyxy": result["tgt_bbox"],
            "src_overlap_pixels": result["src_overlap_pixels"],
            "tgt_overlap_pixels": result["tgt_overlap_pixels"],
            "src_overlap_ratio": result["src_overlap_ratio"],
            "tgt_overlap_ratio": result["tgt_overlap_ratio"],
            "pair_score": result["pair_score"],
            "files": files,
        }
        summary["examples"].append(example_meta)
        write_json(out_dir / f"{stem}.json", example_meta)
        print(json.dumps(example_meta, ensure_ascii=False))

    write_json(out_dir / "summary.json", summary)
    print(f"Wrote outputs to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
