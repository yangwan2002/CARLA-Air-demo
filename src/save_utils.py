"""
save_utils.py
-------------
File-system layout + image writing helpers.

All paths returned by :class:`SequencePaths` are *absolute* and ready to be
passed straight to opencv / numpy.
"""

from __future__ import annotations

import csv
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import numpy as np

try:
    import cv2  # type: ignore
except ImportError:  # pragma: no cover - cv2 is a hard dep at runtime
    cv2 = None  # type: ignore


# ---------------------------------------------------------------------------
# Directory layout
# ---------------------------------------------------------------------------
@dataclass
class SequencePaths:
    """Resolved absolute paths for a single recording sequence."""

    root: Path                 # AirGroundRelay-Sim/sequences/seq_001
    dataset_root: Path         # AirGroundRelay-Sim/
    calibration_dir: Path
    ugv_dir: Path
    uav_dir: Path
    relay_dir: Path
    frames_jsonl: Path
    trajectory_yaml: Path

    ugv_subdirs: Dict[str, Path] = field(default_factory=dict)
    uav_subdirs: Dict[str, Path] = field(default_factory=dict)
    relay_subdirs: Dict[str, Dict[str, Path]] = field(default_factory=dict)

    @staticmethod
    def build(
        output_dir: str | os.PathLike,
        sequence_name: str,
        ugv_sensor_mode: str,
        ugv_capture_depth_stereo: bool,
        uav_camera_flags: Dict[str, bool],
        relay_camera_specs: Iterable[Dict[str, Any]],
        jsonl_filename: str = "frames.jsonl",
    ) -> "SequencePaths":
        dataset_root = Path(output_dir).resolve()
        root = dataset_root / "sequences" / sequence_name

        calibration_dir = root / "calibration"
        ugv_dir = root / "ugv"
        uav_dir = root / "uav"
        relay_dir = root / "relay"

        ugv_subdirs: Dict[str, Path] = {}
        if ugv_sensor_mode == "rgbd":
            ugv_subdirs["front_rgb"] = ugv_dir / "front_rgb"
            ugv_subdirs["front_depth"] = ugv_dir / "front_depth"
        else:  # stereo
            ugv_subdirs["stereo_left"] = ugv_dir / "stereo_left"
            ugv_subdirs["stereo_right"] = ugv_dir / "stereo_right"
            if ugv_capture_depth_stereo:
                ugv_subdirs["front_depth"] = ugv_dir / "front_depth"

        uav_subdirs: Dict[str, Path] = {}
        if uav_camera_flags.get("front_rgb"):
            uav_subdirs["front_rgb"] = uav_dir / "front_rgb"
        if uav_camera_flags.get("front_depth"):
            uav_subdirs["front_depth"] = uav_dir / "front_depth"
        if uav_camera_flags.get("down_rgb"):
            uav_subdirs["down_rgb"] = uav_dir / "down_rgb"
        if uav_camera_flags.get("down_depth"):
            uav_subdirs["down_depth"] = uav_dir / "down_depth"

        relay_subdirs: Dict[str, Dict[str, Path]] = {}
        for spec in relay_camera_specs:
            name = spec["name"]
            entry: Dict[str, Path] = {"rgb": relay_dir / name / "rgb"}
            if spec.get("use_depth"):
                entry["depth"] = relay_dir / name / "depth"
            relay_subdirs[name] = entry

        return SequencePaths(
            root=root,
            dataset_root=dataset_root,
            calibration_dir=calibration_dir,
            ugv_dir=ugv_dir,
            uav_dir=uav_dir,
            relay_dir=relay_dir,
            frames_jsonl=root / jsonl_filename,
            trajectory_yaml=root / "trajectory.yaml",
            ugv_subdirs=ugv_subdirs,
            uav_subdirs=uav_subdirs,
            relay_subdirs=relay_subdirs,
        )

    def ensure(self) -> None:
        """Create every directory referenced by this object."""
        for p in [
            self.root,
            self.dataset_root,
            self.calibration_dir,
            self.ugv_dir,
            self.uav_dir,
            self.relay_dir,
        ]:
            p.mkdir(parents=True, exist_ok=True)
        for p in self.ugv_subdirs.values():
            p.mkdir(parents=True, exist_ok=True)
        for p in self.uav_subdirs.values():
            p.mkdir(parents=True, exist_ok=True)
        for cam in self.relay_subdirs.values():
            for p in cam.values():
                p.mkdir(parents=True, exist_ok=True)

    def rel(self, p: Path) -> str:
        """Path relative to the sequence root, using forward slashes."""
        return p.relative_to(self.root).as_posix()


# ---------------------------------------------------------------------------
# Image writers
# ---------------------------------------------------------------------------
def _require_cv2() -> None:
    if cv2 is None:
        raise RuntimeError(
            "opencv-python is required for image I/O. Install via `pip install opencv-python`."
        )


def save_rgb_png(path: Path, rgb: np.ndarray) -> None:
    """Save an HxWx3 uint8 RGB image as PNG."""
    _require_cv2()
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError(f"save_rgb_png expects HxWx3, got {rgb.shape}")
    path.parent.mkdir(parents=True, exist_ok=True)
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    if not cv2.imwrite(str(path), bgr):
        raise IOError(f"cv2.imwrite failed for {path}")


def save_depth_npy(path: Path, depth_m: np.ndarray) -> None:
    """Save metric depth as a float32 .npy file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, depth_m.astype(np.float32))


def save_depth_png_viz(path: Path, depth_m: np.ndarray, max_m: float = 80.0) -> None:
    """Save a uint16 PNG visualization of metric depth (depth_mm * 256 / max).

    Values >= ``max_m`` are clipped. Useful for quickly viewing in any image
    viewer. Not a lossless representation - keep the .npy for that.
    """
    _require_cv2()
    path.parent.mkdir(parents=True, exist_ok=True)
    clipped = np.clip(depth_m, 0.0, max_m)
    scaled = (clipped / max_m * 65535.0).astype(np.uint16)
    if not cv2.imwrite(str(path), scaled):
        raise IOError(f"cv2.imwrite failed for {path}")


# ---------------------------------------------------------------------------
# JSON / YAML / CSV writers
# ---------------------------------------------------------------------------
def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fp:
        json.dump(obj, fp, indent=2, sort_keys=False, ensure_ascii=False)


class JsonlWriter:
    """Append-only writer for newline-delimited JSON."""

    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self._fp = path.open("a", encoding="utf-8")
        self.count = 0

    def write(self, obj: Dict[str, Any]) -> None:
        self._fp.write(json.dumps(obj, ensure_ascii=False))
        self._fp.write("\n")
        self._fp.flush()
        self.count += 1

    def close(self) -> None:
        try:
            self._fp.close()
        except Exception:
            pass


class PoseCsvWriter:
    """CSV writer for time-stamped 6/7-DoF poses.

    Columns: ``frame_id, sim_time, x, y, z, [roll, pitch, yaw | qw, qx, qy, qz]``.
    """

    EULER_HEADER = ["frame_id", "sim_time", "x", "y", "z", "roll", "pitch", "yaw"]
    QUAT_HEADER = ["frame_id", "sim_time", "x", "y", "z", "qw", "qx", "qy", "qz"]

    def __init__(self, path: Path, mode: str = "euler") -> None:
        if mode not in {"euler", "quat"}:
            raise ValueError("mode must be 'euler' or 'quat'")
        self.mode = mode
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        new_file = not path.exists() or path.stat().st_size == 0
        self._fp = path.open("a", encoding="utf-8", newline="")
        self._writer = csv.writer(self._fp)
        if new_file:
            self._writer.writerow(self.EULER_HEADER if mode == "euler" else self.QUAT_HEADER)
        self.count = 0

    def write(self, frame_id: int, sim_time: float, pose: Dict[str, float]) -> None:
        if self.mode == "euler":
            row = [
                frame_id,
                f"{sim_time:.6f}",
                f"{pose['x']:.6f}",
                f"{pose['y']:.6f}",
                f"{pose['z']:.6f}",
                f"{pose['roll']:.6f}",
                f"{pose['pitch']:.6f}",
                f"{pose['yaw']:.6f}",
            ]
        else:
            row = [
                frame_id,
                f"{sim_time:.6f}",
                f"{pose['x']:.6f}",
                f"{pose['y']:.6f}",
                f"{pose['z']:.6f}",
                f"{pose['qw']:.6f}",
                f"{pose['qx']:.6f}",
                f"{pose['qy']:.6f}",
                f"{pose['qz']:.6f}",
            ]
        self._writer.writerow(row)
        self._fp.flush()
        self.count += 1

    def close(self) -> None:
        try:
            self._fp.close()
        except Exception:
            pass


def frame_filename(frame_id: int, ext: str) -> str:
    """``frame_id=120, ext='png'`` -> ``'000120.png'``."""
    if not ext.startswith("."):
        ext = "." + ext
    return f"{frame_id:06d}{ext}"
