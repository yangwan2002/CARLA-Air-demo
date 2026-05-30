"""
Generate association.txt for ORB-SLAM2 RGB-D from a CARLA sequence directory.

Usage:
    python generate_association.py <sequence_dir> [--rgb front_rgb] [--depth front_depth]

Output:
    <sequence_dir>/association.txt

Format (one line per frame):
    <timestamp> <rgb_path> <timestamp> <depth_path>

Timestamps are taken from frames.jsonl (sim_time field). If frames.jsonl is not
found, timestamps are generated as frame_id / save_hz.
"""
import argparse
import json
import os
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("seq_dir", help="Path to sequence directory")
    parser.add_argument("--rgb",   default="front_rgb",   help="RGB subdir under ugv/ or uav/")
    parser.add_argument("--depth", default="front_depth", help="Depth subdir")
    parser.add_argument("--camera", default="ugv", help="'ugv' or 'uav'")
    parser.add_argument("--hz",    type=float, default=10.0,
                        help="save_hz (fallback if no frames.jsonl)")
    args = parser.parse_args()

    seq = Path(args.seq_dir)
    cam_dir = seq / args.camera
    rgb_dir   = cam_dir / args.rgb
    depth_dir = cam_dir / args.depth

    if not rgb_dir.exists():
        sys.exit(f"ERROR: RGB dir not found: {rgb_dir}")
    if not depth_dir.exists():
        sys.exit(f"ERROR: Depth dir not found: {depth_dir}")

    rgb_files   = sorted(rgb_dir.glob("*.png"))
    depth_files = sorted(depth_dir.glob("*.png"))

    if len(rgb_files) == 0:
        sys.exit("ERROR: No PNG files found in RGB dir")
    if len(rgb_files) != len(depth_files):
        print(f"WARNING: RGB count={len(rgb_files)} != Depth count={len(depth_files)}, "
              f"will pair only overlapping frames")

    n = min(len(rgb_files), len(depth_files))

    # Load timestamps from frames.jsonl if available
    timestamps = {}
    jsonl_path = seq / "frames.jsonl"
    if jsonl_path.exists():
        with open(jsonl_path) as f:
            for line in f:
                if not line.strip():
                    continue
                obj = json.loads(line)
                fid = int(obj["frame_id"])
                timestamps[fid] = float(obj["sim_time"])
        print(f"Loaded {len(timestamps)} timestamps from frames.jsonl")
    else:
        print("frames.jsonl not found, generating timestamps from frame index / hz")

    out_path = cam_dir / "association.txt"
    with open(out_path, "w") as f:
        for i in range(n):
            rgb_file   = rgb_files[i]
            depth_file = depth_files[i]
            frame_id = int(rgb_file.stem)
            if frame_id in timestamps:
                ts = timestamps[frame_id]
            else:
                ts = frame_id / args.hz
            ts_str = f"{ts:.6f}"
            rgb_rel   = f"{args.camera}/{args.rgb}/{rgb_file.name}"
            depth_rel = f"{args.camera}/{args.depth}/{depth_file.name}"
            f.write(f"{ts_str} {rgb_rel} {ts_str} {depth_rel}\n")

    print(f"Written {n} pairs -> {out_path}")
    print(f"First line: {open(out_path).readline().strip()}")


if __name__ == "__main__":
    main()
