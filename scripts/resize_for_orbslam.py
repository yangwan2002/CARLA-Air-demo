"""
Resize CARLA sequence images (RGB + depth) to half resolution for ORB-SLAM2.

ORB-SLAM2's ORBextractor::DistributeOctTree crashes on 1920x1080 images due to
an implicit 640x480 design assumption. Downsampling to 960x540 (factor 0.5)
fixes this while preserving the 16:9 aspect ratio.

Usage:
    python3 resize_for_orbslam.py <sequence_dir> [--camera ugv|uav] [--scale 0.5]

Output:
    Creates <sequence_dir>/<camera>/front_rgb_960/ and front_depth_960/ with
    resized images. The originals are NOT touched.

Depth resizing uses INTER_NEAREST to avoid interpolation artifacts on uint16.
RGB resizing uses INTER_AREA (best for downscaling).
"""
import argparse
import os
from pathlib import Path

import cv2
import numpy as np


def resize_images(src_dir: Path, dst_dir: Path, scale: float, is_depth: bool):
    dst_dir.mkdir(parents=True, exist_ok=True)
    files = sorted(src_dir.glob("*.png"))
    print(f"Resizing {len(files)} images: {src_dir} -> {dst_dir} (scale={scale})")
    for i, fp in enumerate(files):
        img = cv2.imread(str(fp), cv2.IMREAD_UNCHANGED)
        if img is None:
            print(f"  WARNING: failed to read {fp}, skipping")
            continue
        interp = cv2.INTER_NEAREST if is_depth else cv2.INTER_AREA
        resized = cv2.resize(img, None, fx=scale, fy=scale, interpolation=interp)
        out = dst_dir / fp.name
        cv2.imwrite(str(out), resized)
        if (i + 1) % 100 == 0:
            print(f"  {i+1}/{len(files)}")
    print(f"  Done ({len(files)} files)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("seq_dir", help="Path to sequence directory")
    parser.add_argument("--camera", default="ugv", help="'ugv' or 'uav'")
    parser.add_argument("--scale", type=float, default=0.5)
    args = parser.parse_args()

    seq = Path(args.seq_dir)
    cam = seq / args.camera
    scale = args.scale

    suffix = f"_{int(args.scale * 100)}"

    resize_images(cam / "front_rgb",   cam / f"front_rgb{suffix}",   scale, False)
    resize_images(cam / "front_depth", cam / f"front_depth{suffix}", scale, True)

    # Generate association.txt for the resized version
    import json
    timestamps = {}
    jsonl = seq / "frames.jsonl"
    if jsonl.exists():
        with open(jsonl) as f:
            for line in f:
                if not line.strip():
                    continue
                obj = json.loads(line)
                timestamps[int(obj["frame_id"])] = float(obj["sim_time"])

    rgb_files   = sorted((cam / f"front_rgb{suffix}").glob("*.png"))
    depth_files = sorted((cam / f"front_depth{suffix}").glob("*.png"))
    n = min(len(rgb_files), len(depth_files))

    assoc_path = cam / f"association{suffix}.txt"
    with open(assoc_path, "w") as f:
        for i in range(n):
            fid = int(rgb_files[i].stem)
            ts = timestamps.get(fid, fid / 10.0)
            ts_str = f"{ts:.6f}"
            f.write(f"{ts_str} {args.camera}/front_rgb{suffix}/{rgb_files[i].name} "
                    f"{ts_str} {args.camera}/front_depth{suffix}/{depth_files[i].name}\n")
    print(f"Written association: {assoc_path}  ({n} pairs)")
    print(f"First: {open(assoc_path).readline().strip()}")


if __name__ == "__main__":
    main()
