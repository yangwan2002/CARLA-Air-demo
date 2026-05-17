"""
geometry.py
-----------
Geometric helpers shared between CARLA and AirSim.

Coordinate-system reminders
---------------------------
* CARLA uses a UE4 left-handed coordinate system: +X forward, +Y right,
  +Z up. Rotations are roll-pitch-yaw in degrees, and CARLA's positive yaw
  rotates from +X toward +Y (i.e. "clockwise when viewed from above").
* AirSim uses NED: +X north, +Y east, +Z down. Orientations are quaternions
  (w, x, y, z).

Because CARLA-Air does NOT guarantee that the CARLA world origin coincides
with the AirSim PlayerStart, we keep CARLA poses and AirSim poses in their
native frames everywhere in the dataset, and only convert at the boundary
between the two simulators (e.g. when telling AirSim "go fly above the
UGV"). The :func:`carla_to_airsim_ned` helper is intentionally configurable.
"""

from __future__ import annotations

import math
from typing import Dict, Iterable, Tuple

# We deliberately import carla lazily so this file can be imported by
# tooling/tests even when the CARLA python egg is not on PYTHONPATH.
try:
    import carla  # type: ignore
except Exception:  # pragma: no cover - tooling fallback
    carla = None  # type: ignore


# ---------------------------------------------------------------------------
# Intrinsics
# ---------------------------------------------------------------------------
def compute_camera_intrinsic(width: int, height: int, fov_deg: float) -> Dict[str, float]:
    """Return a pinhole intrinsic dict for a CARLA / AirSim camera.

    CARLA renders with a horizontal FOV. The intrinsic focal length therefore
    follows the textbook formula:

        fx = (width / 2) / tan(fov / 2)

    By convention fy == fx, and the principal point sits at the image center.
    """
    fov_rad = math.radians(fov_deg)
    fx = (width / 2.0) / math.tan(fov_rad / 2.0)
    fy = fx
    return {
        "width": int(width),
        "height": int(height),
        "fx": float(fx),
        "fy": float(fy),
        "cx": float(width) / 2.0,
        "cy": float(height) / 2.0,
        "fov_deg": float(fov_deg),
    }


# ---------------------------------------------------------------------------
# look_at for fixed CARLA relay cameras
# ---------------------------------------------------------------------------
def look_at_yaw_pitch(location: Iterable[float], target: Iterable[float]) -> Tuple[float, float]:
    """Compute (yaw_deg, pitch_deg) for a CARLA camera pointed at `target`.

    CARLA's convention:
        * yaw = 0  ->  +X (forward in world space)
        * positive yaw rotates toward +Y (left-handed, "right" in screen)
        * positive pitch points DOWNWARD (this is non-standard but it is how
          CARLA's UE4 backend defines it)

    Therefore for a vector (dx, dy, dz) from camera to target:
        yaw   =  atan2(dy, dx)                      [degrees]
        pitch = -atan2(dz, sqrt(dx*dx + dy*dy))     [degrees]

    The minus sign in pitch is what makes "looking down at the street"
    correspond to a positive pitch value in CARLA.
    """
    lx, ly, lz = (float(v) for v in location)
    tx, ty, tz = (float(v) for v in target)
    dx, dy, dz = tx - lx, ty - ly, tz - lz

    yaw = math.degrees(math.atan2(dy, dx))
    horiz = math.hypot(dx, dy)
    # CARLA pitch is negated relative to the "+pitch up" convention.
    pitch = -math.degrees(math.atan2(dz, horiz))
    return yaw, pitch


def look_at_transform(location: Iterable[float], target: Iterable[float]):
    """Build a ``carla.Transform`` whose camera looks from ``location`` at ``target``.

    Raises:
        RuntimeError: if the ``carla`` package is unavailable.
    """
    if carla is None:
        raise RuntimeError(
            "The `carla` python package is not importable; cannot build a Transform."
        )

    yaw, pitch = look_at_yaw_pitch(location, target)
    lx, ly, lz = (float(v) for v in location)
    return carla.Transform(
        carla.Location(x=lx, y=ly, z=lz),
        carla.Rotation(roll=0.0, pitch=pitch, yaw=yaw),
    )


# ---------------------------------------------------------------------------
# CARLA Transform <-> dict serialization
# ---------------------------------------------------------------------------
def carla_transform_to_dict(tf) -> Dict[str, float]:
    """Serialize a ``carla.Transform`` to the schema used in frames.jsonl."""
    loc = tf.location
    rot = tf.rotation
    return {
        "x": float(loc.x),
        "y": float(loc.y),
        "z": float(loc.z),
        "roll": float(rot.roll),
        "pitch": float(rot.pitch),
        "yaw": float(rot.yaw),
    }


def airsim_pose_to_dict(pose) -> Dict[str, float]:
    """Serialize an ``airsim.Pose`` to NED + quaternion JSON-friendly dict."""
    p = pose.position
    q = pose.orientation
    return {
        "x": float(p.x_val),
        "y": float(p.y_val),
        "z": float(p.z_val),
        "qw": float(q.w_val),
        "qx": float(q.x_val),
        "qy": float(q.y_val),
        "qz": float(q.z_val),
    }


# ---------------------------------------------------------------------------
# CARLA -> AirSim NED helper (intentionally approximate / configurable)
# ---------------------------------------------------------------------------
def carla_to_airsim_ned(
    carla_xyz: Iterable[float],
    origin_offset_carla: Iterable[float] = (0.0, 0.0, 0.0),
    yaw_offset_deg: float = 0.0,
) -> Tuple[float, float, float]:
    """Approximate conversion from CARLA world XYZ to AirSim NED XYZ.

    .. warning::
        CARLA-Air does **not** guarantee that the CARLA world frame and the
        AirSim PlayerStart frame coincide. This function therefore exposes
        two knobs that you should measure for your build:

        * ``origin_offset_carla`` - CARLA (x, y, z) of the AirSim NED origin.
        * ``yaw_offset_deg``      - rotation about z to align CARLA +X with
          AirSim +N (north).

    The general transform is:

        1. translate so the AirSim origin sits at (0, 0, 0) in CARLA
        2. rotate around z by ``yaw_offset_deg`` (CARLA convention: +yaw
           rotates +X toward +Y)
        3. map left-handed CARLA (x, y, z) to NED via
                N =  x'
                E =  y'
                D = -z'

    Returns:
        Tuple ``(n, e, d)`` suitable for ``moveToPositionAsync``.
    """
    cx, cy, cz = (float(v) for v in carla_xyz)
    ox, oy, oz = (float(v) for v in origin_offset_carla)

    cx -= ox
    cy -= oy
    cz -= oz

    yaw = math.radians(yaw_offset_deg)
    rx =  math.cos(yaw) * cx + math.sin(yaw) * cy
    ry = -math.sin(yaw) * cx + math.cos(yaw) * cy
    rz = cz

    n = rx
    e = ry
    d = -rz
    return n, e, d


# ---------------------------------------------------------------------------
# Depth conversion utilities
# ---------------------------------------------------------------------------
def decode_carla_depth(raw_bgra) -> "Any":
    """Decode a CARLA depth image (BGRA uint8) to metric depth (float32).

    CARLA encodes a 24-bit normalized depth value across the R, G, B channels:

        normalized = (R + G * 256 + B * 256 * 256) / (256**3 - 1)
        depth_m    = normalized * far_plane_m   # far plane = 1000 m by default

    ``raw_bgra`` is expected to be an ``np.ndarray`` of shape (H, W, 4) and
    dtype uint8, as returned by ``np.frombuffer(image.raw_data, ...)`` then
    reshaped. CARLA emits images in BGRA order.

    .. note::
        Some CARLA forks deviate from the 1000 m far-plane convention. If you
        find that your decoded depth is off by a constant factor on CARLA-Air,
        adjust ``far_plane_m`` here (or expose it via config).
    """
    import numpy as np

    if raw_bgra.dtype != np.uint8 or raw_bgra.ndim != 3 or raw_bgra.shape[2] < 3:
        raise ValueError(
            f"decode_carla_depth expects an (H, W, 4) uint8 array, got "
            f"shape={raw_bgra.shape}, dtype={raw_bgra.dtype}"
        )

    b = raw_bgra[:, :, 0].astype(np.float32)
    g = raw_bgra[:, :, 1].astype(np.float32)
    r = raw_bgra[:, :, 2].astype(np.float32)

    normalized = (r + g * 256.0 + b * 256.0 * 256.0) / (256.0 ** 3 - 1.0)
    far_plane_m = 1000.0  # TODO: verify on the exact CARLA-Air build you use
    depth_m = normalized * far_plane_m
    return depth_m.astype(np.float32)
