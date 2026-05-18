"""
sensors.py
----------
CARLA sensor wrappers.

Every sensor's callback simply puts the raw ``carla.Image`` (or
``carla.SensorData``) onto a thread-safe queue keyed by the CARLA frame
number. The synchronous main loop then drains that queue and converts the
chosen frame into numpy at save time.

We deliberately do *no* image decoding inside the callback - decoding can be
expensive enough to backpressure the CARLA streaming thread, which in turn
can stall the world tick.
"""

from __future__ import annotations

import logging
import queue
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

try:
    import carla  # type: ignore
except ImportError:  # pragma: no cover
    carla = None  # type: ignore

from .actors import ActorRegistry


LOGGER = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Sensor wrapper
# ---------------------------------------------------------------------------
@dataclass
class CarlaSensor:
    """A single CARLA sensor with its own bounded image queue."""

    name: str
    sensor_type: str                          # "rgb" or "depth"
    actor: "carla.Sensor"
    width: int
    height: int
    fov: float
    queue: "queue.Queue[Any]" = field(default_factory=lambda: queue.Queue(maxsize=32))

    def start(self) -> None:
        # The closure captures ``self`` so the callback always knows where
        # to deposit the image.
        def _callback(data):
            try:
                self.queue.put_nowait(data)
            except queue.Full:
                # We are running behind. Drop the oldest frame and try once
                # more so we always keep the most recent data available.
                try:
                    _ = self.queue.get_nowait()
                except queue.Empty:
                    pass
                try:
                    self.queue.put_nowait(data)
                except queue.Full:
                    LOGGER.warning("Sensor %s queue full; dropping frame %d", self.name, data.frame)

        self.actor.listen(_callback)

    def stop(self) -> None:
        try:
            if self.actor.is_alive and self.actor.is_listening:
                self.actor.stop()
        except Exception as e:  # pragma: no cover
            LOGGER.warning("stop() failed for sensor %s: %s", self.name, e)


# ---------------------------------------------------------------------------
# Sensor factories
# ---------------------------------------------------------------------------
def _make_camera_blueprint(
    world: "carla.World",
    sensor_type: str,
    width: int,
    height: int,
    fov: float,
) -> "carla.ActorBlueprint":
    bp_lib = world.get_blueprint_library()
    if sensor_type == "rgb":
        bp = bp_lib.find("sensor.camera.rgb")
    elif sensor_type == "depth":
        bp = bp_lib.find("sensor.camera.depth")
    else:
        raise ValueError(f"Unsupported sensor_type {sensor_type!r}")

    bp.set_attribute("image_size_x", str(int(width)))
    bp.set_attribute("image_size_y", str(int(height)))
    bp.set_attribute("fov", str(float(fov)))
    bp.set_attribute("sensor_tick", "0.0")

    if sensor_type == "rgb":
        # MegaDepth-style training data has no motion blur / bloom / lens
        # flare. Disable them so the evaluation distribution matches.
        for k, v in (
            ("motion_blur_intensity", "0.0"),
            ("motion_blur_max_distortion", "0.0"),
            ("motion_blur_min_object_screen_size", "0.0"),
            ("lens_flare_intensity", "0.0"),
            ("bloom_intensity", "0.0"),
            ("chromatic_aberration_intensity", "0.0"),
        ):
            try:
                bp.set_attribute(k, v)
            except Exception:
                pass
    return bp


def spawn_attached_camera(
    world: "carla.World",
    parent: "carla.Actor",
    name: str,
    sensor_type: str,
    transform: "carla.Transform",
    width: int,
    height: int,
    fov: float,
    registry: ActorRegistry,
) -> CarlaSensor:
    """Spawn a camera attached to ``parent`` (e.g. the UGV)."""
    bp = _make_camera_blueprint(world, sensor_type, width, height, fov)
    actor = world.spawn_actor(bp, transform, attach_to=parent)
    registry.register(actor)
    return CarlaSensor(
        name=name, sensor_type=sensor_type, actor=actor,
        width=width, height=height, fov=fov,
    )


def spawn_static_camera(
    world: "carla.World",
    name: str,
    sensor_type: str,
    transform: "carla.Transform",
    width: int,
    height: int,
    fov: float,
    registry: ActorRegistry,
) -> CarlaSensor:
    """Spawn a static (world-anchored) camera, e.g. a relay surveillance cam."""
    bp = _make_camera_blueprint(world, sensor_type, width, height, fov)
    actor = world.spawn_actor(bp, transform)
    registry.register(actor)
    return CarlaSensor(
        name=name, sensor_type=sensor_type, actor=actor,
        width=width, height=height, fov=fov,
    )


# ---------------------------------------------------------------------------
# UGV camera rig
# ---------------------------------------------------------------------------
@dataclass
class UgvCameraRig:
    mode: str                                # "rgbd" or "stereo"
    sensors: Dict[str, CarlaSensor]          # logical-name -> sensor


def build_ugv_camera_rig(
    world: "carla.World",
    parent: "carla.Vehicle",
    ugv_cfg: Dict[str, Any],
    registry: ActorRegistry,
) -> UgvCameraRig:
    """Construct the UGV's RGB-D *or* stereo rig according to config."""
    cam_cfg = ugv_cfg["camera"]
    w, h = cam_cfg["resolution"]
    fov = cam_cfg["fov"]
    loc = cam_cfg["location"]
    rot = cam_cfg["rotation"]

    base_tf = carla.Transform(
        carla.Location(x=loc[0], y=loc[1], z=loc[2]),
        carla.Rotation(roll=rot[0], pitch=rot[1], yaw=rot[2]),
    )

    mode = ugv_cfg["sensor_mode"]
    sensors: Dict[str, CarlaSensor] = {}

    if mode == "rgbd":
        sensors["front_rgb"] = spawn_attached_camera(
            world, parent, "ugv_front_rgb", "rgb", base_tf, w, h, fov, registry,
        )
        sensors["front_depth"] = spawn_attached_camera(
            world, parent, "ugv_front_depth", "depth", base_tf, w, h, fov, registry,
        )
    elif mode == "stereo":
        baseline = float(cam_cfg.get("stereo_baseline_m", 0.2))
        left_tf = carla.Transform(
            carla.Location(x=loc[0], y=loc[1] - baseline / 2.0, z=loc[2]),
            carla.Rotation(roll=rot[0], pitch=rot[1], yaw=rot[2]),
        )
        right_tf = carla.Transform(
            carla.Location(x=loc[0], y=loc[1] + baseline / 2.0, z=loc[2]),
            carla.Rotation(roll=rot[0], pitch=rot[1], yaw=rot[2]),
        )
        sensors["stereo_left"] = spawn_attached_camera(
            world, parent, "ugv_stereo_left", "rgb", left_tf, w, h, fov, registry,
        )
        sensors["stereo_right"] = spawn_attached_camera(
            world, parent, "ugv_stereo_right", "rgb", right_tf, w, h, fov, registry,
        )
        if cam_cfg.get("stereo_capture_depth", False):
            sensors["front_depth"] = spawn_attached_camera(
                world, parent, "ugv_stereo_depth", "depth", left_tf, w, h, fov, registry,
            )
    else:
        raise ValueError(f"Unknown UGV sensor_mode {mode!r}")

    return UgvCameraRig(mode=mode, sensors=sensors)


# ---------------------------------------------------------------------------
# Relay cameras
# ---------------------------------------------------------------------------
@dataclass
class RelayCamera:
    name: str
    transform: "carla.Transform"
    rgb: CarlaSensor
    depth: Optional[CarlaSensor]
    use_depth: bool


def build_relay_cameras(
    world: "carla.World",
    relay_cfg: Dict[str, Any],
    registry: ActorRegistry,
) -> List[RelayCamera]:
    """Spawn every relay camera and return them in config order."""
    from .geometry import look_at_transform  # local import to avoid carla coupling on tooling

    default_res = relay_cfg.get("default_resolution", [1280, 720])
    default_fov = relay_cfg.get("default_fov", 90.0)
    default_use_depth = relay_cfg.get("default_use_depth", True)

    out: List[RelayCamera] = []
    for spec in relay_cfg["cameras"]:
        name = spec["name"]
        res = spec.get("resolution", default_res)
        fov = spec.get("fov", default_fov)
        use_depth = spec.get("use_depth", default_use_depth)

        tf = look_at_transform(spec["location"], spec["look_at"])

        rgb = spawn_static_camera(
            world, f"{name}_rgb", "rgb", tf, res[0], res[1], fov, registry,
        )
        depth_sensor: Optional[CarlaSensor] = None
        if use_depth:
            depth_sensor = spawn_static_camera(
                world, f"{name}_depth", "depth", tf, res[0], res[1], fov, registry,
            )

        out.append(RelayCamera(
            name=name, transform=tf, rgb=rgb, depth=depth_sensor, use_depth=bool(use_depth),
        ))

    return out


# ---------------------------------------------------------------------------
# Queue helpers
# ---------------------------------------------------------------------------
def pull_image_for_frame(
    sensor: CarlaSensor,
    target_frame: int,
    timeout_s: float = 2.0,
    max_skips: int = 64,
) -> Optional["carla.Image"]:
    """Drain ``sensor.queue`` until we find an image with ``image.frame == target_frame``.

    Older frames are discarded with a warning. Returns ``None`` if no matching
    frame arrives within ``timeout_s``.
    """
    deadline = None  # set on first wait
    import time as _time

    skipped = 0
    while skipped < max_skips:
        try:
            data = sensor.queue.get(timeout=timeout_s)
        except queue.Empty:
            LOGGER.warning(
                "Sensor %s: timed out waiting for frame %d (timeout=%.2fs)",
                sensor.name, target_frame, timeout_s,
            )
            return None

        if data.frame == target_frame:
            return data
        elif data.frame < target_frame:
            skipped += 1
            LOGGER.debug(
                "Sensor %s: discarding stale frame %d (waiting for %d)",
                sensor.name, data.frame, target_frame,
            )
            continue
        else:
            # We somehow overshot - keep the future frame, log, and bail.
            LOGGER.warning(
                "Sensor %s: got future frame %d while waiting for %d - "
                "the world tick may have been missed.",
                sensor.name, data.frame, target_frame,
            )
            return data

    LOGGER.warning(
        "Sensor %s: discarded %d stale frames without seeing %d. Giving up.",
        sensor.name, max_skips, target_frame,
    )
    return None


def pull_latest_image(
    sensor: CarlaSensor,
    timeout_s: float = 2.0,
) -> Optional["carla.Image"]:
    """Block for the next image on the sensor queue, then drain any stale ones.

    Returns the freshest available image, or None on timeout. Use this in async
    mode where sensor frames are not aligned with a specific world tick.
    """
    try:
        latest = sensor.queue.get(timeout=timeout_s)
    except queue.Empty:
        LOGGER.warning(
            "Sensor %s: timed out waiting for any frame (timeout=%.2fs)",
            sensor.name, timeout_s,
        )
        return None
    while True:
        try:
            latest = sensor.queue.get_nowait()
        except queue.Empty:
            break
    return latest


# ---------------------------------------------------------------------------
# Image decoding
# ---------------------------------------------------------------------------
def carla_rgb_image_to_array(image: "carla.Image") -> np.ndarray:
    """Convert a CARLA RGB ``carla.Image`` (BGRA) to a contiguous HxWx3 RGB uint8 array."""
    buf = np.frombuffer(image.raw_data, dtype=np.uint8)
    bgra = buf.reshape((image.height, image.width, 4))
    rgb = bgra[:, :, :3][:, :, ::-1]  # BGRA -> RGB
    return np.ascontiguousarray(rgb)


def carla_depth_image_to_array(image: "carla.Image") -> np.ndarray:
    """Convert a CARLA depth ``carla.Image`` to HxWx4 uint8 (ready for decode_carla_depth)."""
    buf = np.frombuffer(image.raw_data, dtype=np.uint8)
    bgra = buf.reshape((image.height, image.width, 4))
    return bgra
