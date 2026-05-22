"""
carla_uav.py
------------
A CARLA-native "drone": a kinematic actor that we teleport every tick to a
target pose, with RGB + Depth cameras attached.

Why a kinematic actor instead of plain sensor.camera spawned in the world:
  * attaching the cameras to a parent gives them stable per-tick transforms
    (CARLA computes child poses on the simulator side).
  * the parent itself (a small bicycle blueprint with physics disabled) is
    visible to the relay cameras when the UAV is low, which makes a few
    qualitative figures possible without affecting the rest of the dataset.
  * the bicycle's footprint is ~1m, which is essentially invisible (~1-2 px)
    to ground-level cameras when the UAV is at the configured 30 m cruise
    altitude, so it does not pollute SLAM on the UGV.

Coordinate convention: pure CARLA, +X forward / +Y right / +Z up. UAV
"camera_pitch_deg" of -90 means the camera looks straight down.

Public API:
  * ``CarlaUav.spawn(...)`` — build the actor + cameras and register them
  * ``CarlaUav.set_pose(x, y, z, yaw_deg)`` — teleport for the next tick
  * ``CarlaUav.get_transform()`` / ``CarlaUav.cameras`` — query state
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

try:
    import carla  # type: ignore
except ImportError:  # pragma: no cover
    carla = None  # type: ignore

from .actors import ActorRegistry
from .sensors import CarlaSensor, make_camera_blueprint


LOGGER = logging.getLogger(__name__)


@dataclass
class CarlaUav:
    """The UAV's parent actor + its attached camera rig."""

    body: "carla.Actor"
    cameras: Dict[str, CarlaSensor]            # logical name -> sensor
    cfg: Dict[str, Any]
    min_altitude_m: float = 5.0

    @staticmethod
    def spawn(
        world: "carla.World",
        uav_cfg: Dict[str, Any],
        registry: ActorRegistry,
        spawn_xy: Optional[List[float]] = None,
        spawn_altitude_m: Optional[float] = None,
    ) -> "CarlaUav":
        """Spawn the UAV body + attached cameras and return a wrapper."""
        if carla is None:
            raise ImportError("carla python package is not importable")

        body_bp_id = uav_cfg.get("body_blueprint", "vehicle.diamondback.century")
        bp_lib = world.get_blueprint_library()
        candidates = bp_lib.filter(body_bp_id)
        if not candidates:
            raise ValueError(
                f"No CARLA blueprint matches {body_bp_id!r}. "
                f"Try one of the bicycles: vehicle.diamondback.century, "
                f"vehicle.gazelle.omafiets, vehicle.bh.crossbike."
            )
        body_bp = candidates[0]
        if body_bp.has_attribute("role_name"):
            body_bp.set_attribute("role_name", uav_cfg.get("role_name", "uav_collector"))

        # Initial pose: well above ground so spawn does not collide.
        init_xy = spawn_xy or uav_cfg.get("spawn_xy", [0.0, 0.0])
        init_alt = float(spawn_altitude_m if spawn_altitude_m is not None
                         else uav_cfg.get("spawn_altitude_m", uav_cfg.get("altitude_m", 30.0)))
        init_alt = max(init_alt, float(uav_cfg.get("min_altitude_m", 5.0)))

        spawn_tf = carla.Transform(
            carla.Location(x=float(init_xy[0]), y=float(init_xy[1]), z=init_alt),
            carla.Rotation(roll=0.0, pitch=0.0, yaw=float(uav_cfg.get("spawn_yaw_deg", 0.0))),
        )

        body = world.try_spawn_actor(body_bp, spawn_tf)
        if body is None:
            # Try a couple of nearby points if the spawn cell happened to be busy.
            for dz in (5.0, 10.0, 15.0):
                spawn_tf.location.z = init_alt + dz
                body = world.try_spawn_actor(body_bp, spawn_tf)
                if body is not None:
                    break
        if body is None:
            raise RuntimeError(f"Failed to spawn UAV body {body_bp_id!r} at {init_xy} alt={init_alt}")

        # Critical: disable physics so it does not fall, and freeze it so
        # the simulator never resets our teleports.
        try:
            body.set_simulate_physics(False)
        except Exception as e:  # pragma: no cover
            LOGGER.warning("set_simulate_physics(False) failed for UAV body: %s", e)

        registry.register(body)
        LOGGER.info("Spawned UAV body %s at (%.2f, %.2f, %.2f).",
                    body_bp.id, init_xy[0], init_xy[1], init_alt)

        # ---- attach cameras ----------------------------------------------
        cam_cfg = uav_cfg.get("camera", {}) or {}
        res = cam_cfg.get("resolution", [1280, 720])
        fov = float(cam_cfg.get("fov", 90.0))
        pitch_deg = float(uav_cfg.get("camera_pitch_deg", -90.0))

        # Camera mounted slightly below the body, pointing down (or whatever pitch).
        cam_loc = cam_cfg.get("location", [0.0, 0.0, -0.3])
        cam_tf = carla.Transform(
            carla.Location(x=float(cam_loc[0]), y=float(cam_loc[1]), z=float(cam_loc[2])),
            carla.Rotation(roll=0.0, pitch=pitch_deg, yaw=0.0),
        )

        cameras: Dict[str, CarlaSensor] = {}
        cam_flags = uav_cfg.get("cameras", {"front_rgb": True, "front_depth": True}) or {}

        if cam_flags.get("front_rgb"):
            rgb_bp = make_camera_blueprint(world, "rgb", int(res[0]), int(res[1]), fov)
            rgb_actor = world.spawn_actor(rgb_bp, cam_tf, attach_to=body)
            registry.register(rgb_actor)
            cameras["front_rgb"] = CarlaSensor(
                name="uav_front_rgb", sensor_type="rgb", actor=rgb_actor,
                width=int(res[0]), height=int(res[1]), fov=fov,
            )

        if cam_flags.get("front_depth"):
            d_bp = make_camera_blueprint(world, "depth", int(res[0]), int(res[1]), fov)
            d_actor = world.spawn_actor(d_bp, cam_tf, attach_to=body)
            registry.register(d_actor)
            cameras["front_depth"] = CarlaSensor(
                name="uav_front_depth", sensor_type="depth", actor=d_actor,
                width=int(res[0]), height=int(res[1]), fov=fov,
            )

        for cam in cameras.values():
            cam.start()

        LOGGER.info("UAV camera rig: %s, fov=%.1f, pitch=%.1f deg",
                    list(cameras.keys()), fov, pitch_deg)

        return CarlaUav(
            body=body,
            cameras=cameras,
            cfg=uav_cfg,
            min_altitude_m=float(uav_cfg.get("min_altitude_m", 5.0)),
        )

    # ------------------------------------------------------------------
    def set_pose(self, x: float, y: float, z: float, yaw_deg: float = 0.0) -> None:
        """Teleport the UAV to a new world pose. Called once per CARLA tick."""
        if carla is None:
            return
        z = max(float(z), self.min_altitude_m)
        tf = carla.Transform(
            carla.Location(x=float(x), y=float(y), z=z),
            carla.Rotation(roll=0.0, pitch=0.0, yaw=float(yaw_deg)),
        )
        try:
            self.body.set_transform(tf)
        except Exception as e:  # pragma: no cover
            LOGGER.warning("UAV set_transform failed: %s", e)

    def get_transform(self) -> "carla.Transform":
        return self.body.get_transform()

    def get_camera_transform(self, logical_name: str) -> Optional["carla.Transform"]:
        cam = self.cameras.get(logical_name)
        if cam is None:
            return None
        try:
            return cam.actor.get_transform()
        except Exception:
            return None
