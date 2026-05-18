"""
actors.py
---------
Helpers for spawning the UGV and (later) destroying every actor we create.

We keep a single :class:`ActorRegistry` that owns every spawned actor /
sensor so the main script only has to call ``registry.destroy_all()`` in its
``finally`` block.
"""

from __future__ import annotations

import logging
import random
from typing import Any, Dict, List, Optional

try:
    import carla  # type: ignore
except ImportError:  # pragma: no cover
    carla = None  # type: ignore


LOGGER = logging.getLogger(__name__)


class ActorRegistry:
    """Tracks every CARLA actor we spawn so we can destroy them cleanly."""

    def __init__(self, world: "carla.World") -> None:
        self.world = world
        self._actors: List["carla.Actor"] = []

    def register(self, actor: "carla.Actor") -> "carla.Actor":
        self._actors.append(actor)
        return actor

    def __len__(self) -> int:
        return len(self._actors)

    def destroy_all(self) -> None:
        """Destroy every registered actor. Safe to call multiple times."""
        if not self._actors:
            return

        LOGGER.info("Destroying %d CARLA actors...", len(self._actors))
        # Destroy sensors before their parent vehicles.
        sensors = [a for a in self._actors if a.type_id.startswith("sensor.")]
        other = [a for a in self._actors if not a.type_id.startswith("sensor.")]

        for s in sensors:
            try:
                if s.is_alive and s.is_listening:
                    s.stop()
            except Exception as e:
                LOGGER.warning("sensor.stop() failed for %s: %s", s, e)
            try:
                if s.is_alive:
                    s.destroy()
            except Exception as e:
                LOGGER.warning("sensor.destroy() failed for %s: %s", s, e)

        for a in other:
            try:
                if a.is_alive:
                    a.destroy()
            except Exception as e:
                LOGGER.warning("actor.destroy() failed for %s: %s", a, e)

        self._actors.clear()


# ---------------------------------------------------------------------------
# UGV
# ---------------------------------------------------------------------------
def spawn_ugv(
    world: "carla.World",
    ugv_cfg: Dict[str, Any],
    registry: ActorRegistry,
    seed: Optional[int] = None,
) -> "carla.Vehicle":
    """Spawn the ground vehicle described by ``ugv_cfg`` and return it."""
    if carla is None:
        raise ImportError("carla python package is not importable")

    blueprint_id = ugv_cfg.get("blueprint", "vehicle.tesla.model3")
    bp_lib = world.get_blueprint_library()
    candidates = bp_lib.filter(blueprint_id)
    if not candidates:
        raise ValueError(
            f"No CARLA blueprints match {blueprint_id!r}. "
            f"Try one of: {[b.id for b in bp_lib.filter('vehicle.*')][:5]} ..."
        )
    bp = candidates[0]

    role_name = ugv_cfg.get("role_name", "ugv_collector")
    if bp.has_attribute("role_name"):
        bp.set_attribute("role_name", role_name)
    if bp.has_attribute("color"):
        # Deterministic but unique color per role.
        rng = random.Random(seed if seed is not None else hash(role_name))
        recommended = bp.get_attribute("color").recommended_values
        if recommended:
            bp.set_attribute("color", rng.choice(recommended))

    spawn_points = world.get_map().get_spawn_points()
    if not spawn_points:
        raise RuntimeError("CARLA map returned no spawn points!")

    spawn_xy = ugv_cfg.get("spawn_xy", None)
    if spawn_xy is not None and len(spawn_xy) >= 2:
        sx, sy = float(spawn_xy[0]), float(spawn_xy[1])
        sz = 0.6  # small z offset above ground to avoid collision
        yaw = float(ugv_cfg.get("spawn_yaw_deg", 0.0))
        spawn_tf = carla.Transform(
            carla.Location(x=sx, y=sy, z=sz),
            carla.Rotation(roll=0.0, pitch=0.0, yaw=yaw),
        )
        LOGGER.info(
            "Spawning UGV %s at explicit spawn_xy=(%.2f, %.2f, %.2f), yaw=%.1f.",
            bp.id, sx, sy, sz, yaw,
        )
        vehicle = world.try_spawn_actor(bp, spawn_tf)
        if vehicle is None:
            LOGGER.warning(
                "Explicit spawn_xy failed; falling back to spawn_point_index.",
            )
    else:
        vehicle = None

    if vehicle is None:
        idx = int(ugv_cfg.get("spawn_point_index", 0))
        if idx < 0 or idx >= len(spawn_points):
            LOGGER.warning(
                "spawn_point_index=%d out of range (0..%d); using 0 instead.",
                idx, len(spawn_points) - 1,
            )
            idx = 0
        spawn_tf = spawn_points[idx]

        LOGGER.info(
            "Spawning UGV %s at spawn point %d (xyz=(%.2f, %.2f, %.2f)).",
            bp.id, idx, spawn_tf.location.x, spawn_tf.location.y, spawn_tf.location.z,
        )

        vehicle = world.try_spawn_actor(bp, spawn_tf)
        if vehicle is None:
            # Try a few neighbouring spawn points if the chosen one is blocked.
            LOGGER.warning("Spawn at index %d failed (occupied); trying neighbours.", idx)
            for offset in range(1, 6):
                for sign in (1, -1):
                    j = (idx + sign * offset) % len(spawn_points)
                    vehicle = world.try_spawn_actor(bp, spawn_points[j])
                    if vehicle is not None:
                        LOGGER.info("Spawned UGV at fallback spawn point %d.", j)
                        break
                if vehicle is not None:
                    break
    if vehicle is None:
        raise RuntimeError("Failed to spawn UGV after exhausting fallbacks.")

    registry.register(vehicle)
    return vehicle
