"""
scene_dressing.py
-----------------
Populate the scene with semantic content for the SAGENet evaluation:

* "static cars" — vehicles spawned with physics off, parked at fixed CARLA
  poses (e.g. along the curb near each relay's look_at point). They are the
  most reliable semantic anchors for cross-view matching.
* "static props" — bicycles, scooters, benches, trash cans, advertisements,
  trees. Variety per relay is configurable so we can support a per-relay
  semantic-density ablation (relay_01 dense ... relay_05 sparse).
* "moving NPCs" — vehicles on autopilot via the TrafficManager, plus a
  small group of pedestrians on the walker controller.

The :func:`dress_scene` driver reads ``scene.dressing`` from the YAML and
produces all three groups in one pass. Every spawned actor is registered
with the supplied :class:`ActorRegistry` so it gets cleaned up automatically.
"""

from __future__ import annotations

import logging
import math
import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

try:
    import carla  # type: ignore
except ImportError:  # pragma: no cover
    carla = None  # type: ignore

from .actors import ActorRegistry


LOGGER = logging.getLogger(__name__)


# ------------------------------------------------------------------------ #
# Static placement helpers
# ------------------------------------------------------------------------ #
def _yaw_pointing_outward(cx: float, cy: float, x: float, y: float) -> float:
    """Yaw (CARLA deg) so the actor faces away from a center point."""
    return math.degrees(math.atan2(y - cy, x - cx))


def _spawn_static_actor(
    world: "carla.World",
    bp_id: str,
    x: float, y: float, z: float, yaw_deg: float,
    registry: ActorRegistry,
    role_name: Optional[str] = None,
) -> Optional["carla.Actor"]:
    """Spawn a single static actor with physics disabled. Returns None on failure."""
    if carla is None:
        return None
    bp_lib = world.get_blueprint_library()
    candidates = bp_lib.filter(bp_id)
    if not candidates:
        LOGGER.warning("Static blueprint %r not found; skipping.", bp_id)
        return None
    bp = candidates[0]
    if role_name and bp.has_attribute("role_name"):
        bp.set_attribute("role_name", role_name)
    if bp.has_attribute("color"):
        recommended = bp.get_attribute("color").recommended_values
        if recommended:
            bp.set_attribute("color", random.choice(recommended))

    tf = carla.Transform(
        carla.Location(x=float(x), y=float(y), z=float(z)),
        carla.Rotation(roll=0.0, pitch=0.0, yaw=float(yaw_deg)),
    )

    actor = world.try_spawn_actor(bp, tf)
    if actor is None:
        # Try a few small jitters before giving up — tight street curbs are
        # often partially blocked.
        for dx, dy in [(0.5, 0.0), (-0.5, 0.0), (0.0, 0.5), (0.0, -0.5),
                       (1.0, 0.0), (-1.0, 0.0), (0.0, 1.0), (0.0, -1.0)]:
            tf.location.x = float(x) + dx
            tf.location.y = float(y) + dy
            actor = world.try_spawn_actor(bp, tf)
            if actor is not None:
                break
    if actor is None:
        LOGGER.debug("Could not spawn %s near (%.1f, %.1f) after jitters.", bp_id, x, y)
        return None

    if hasattr(actor, "set_simulate_physics"):
        try:
            actor.set_simulate_physics(False)
        except Exception:
            pass
    registry.register(actor)
    return actor


def _far_sidewalk_positions(
    center: Tuple[float, float],
    away_dir: Tuple[float, float],
    offset_m: float,
    count: int,
    z: float = 0.05,
    yaw_offset_deg: float = 0.0,
) -> List[Tuple[float, float, float, float]]:
    """Place actors on the FAR sidewalk (opposite the relay) so they appear
    BEHIND the lane in the relay's image.

    ``center`` is the relay's look_at point on the lane.
    ``away_dir`` is a unit vector pointing AWAY from the relay (i.e. across
    the road from the relay's pole). Actors land at center + offset_m * away_dir
    plus a small along-street spread. This guarantees props never sit IN the
    lane between the relay and its target — they sit beyond it on the
    far sidewalk.

    The "along-street" axis is the direction perpendicular to ``away_dir``;
    we spread actors evenly along it.
    """
    import math as _m
    out: List[Tuple[float, float, float, float]] = []
    cx, cy = center
    ax, ay = away_dir
    norm = _m.hypot(ax, ay) or 1.0
    ax, ay = ax / norm, ay / norm
    # Perpendicular along-street unit vector
    px, py = -ay, ax
    for i in range(count):
        # Spread along the street: -count/2 .. +count/2 with 3 m spacing
        s = (i - (count - 1) / 2.0) * 3.0
        x = cx + ax * offset_m + px * s
        y = cy + ay * offset_m + py * s
        # Yaw points along the street so static cars look parked
        yaw = _m.degrees(_m.atan2(py, px)) + yaw_offset_deg
        out.append((x, y, z, yaw))
    return out


def _sidewalk_positions(
    center: Tuple[float, float],
    offset_y_m: float,
    count: int,
    z: float = 0.05,
    yaw_offset_deg: float = 0.0,
) -> List[Tuple[float, float, float, float]]:
    """Place actors on the SIDEWALK either side of an east/west street.

    The y=13/17 streets in Town10HD run along the x axis, so sidewalks are
    parallel lines at y = look_at_y ± offset_y_m. We alternate sides and
    spread along x. Yaw points along the street so vehicles look parked.
    """
    out: List[Tuple[float, float, float, float]] = []
    cx, cy = center
    for i in range(count):
        side = -1 if (i % 2 == 0) else +1
        # Distribute along x, +/- 6 m around the relay center
        dx = ((i // 2) - max(count // 4, 1) * 0.5) * 4.0
        x = cx + dx
        y = cy + side * offset_y_m
        yaw = 0.0 + (180.0 if side > 0 else 0.0) + yaw_offset_deg
        out.append((x, y, z, yaw))
    return out


def _ring_positions(
    center: Tuple[float, float],
    radius: float,
    count: int,
    z: float = 0.05,
    yaw_offset_deg: float = 0.0,
) -> List[Tuple[float, float, float, float]]:
    """Evenly spaced positions around a circle. Returns (x, y, z, yaw)."""
    out: List[Tuple[float, float, float, float]] = []
    cx, cy = center
    for i in range(count):
        theta = 2.0 * math.pi * (i / max(count, 1))
        x = cx + radius * math.cos(theta)
        y = cy + radius * math.sin(theta)
        yaw = math.degrees(theta) + 90.0 + yaw_offset_deg  # tangent to the ring
        out.append((x, y, z, yaw))
    return out


# ------------------------------------------------------------------------ #
# Per-relay dressing
# ------------------------------------------------------------------------ #
@dataclass
class DressingResult:
    static_vehicles: List["carla.Actor"] = field(default_factory=list)
    static_props: List["carla.Actor"] = field(default_factory=list)
    npc_vehicles: List["carla.Actor"] = field(default_factory=list)
    walkers: List["carla.Actor"] = field(default_factory=list)
    walker_controllers: List["carla.Actor"] = field(default_factory=list)


def _dress_one_relay(
    world: "carla.World",
    relay_name: str,
    look_at: Tuple[float, float, float],
    relay_location: Tuple[float, float, float],
    density_cfg: Dict[str, Any],
    registry: ActorRegistry,
    rng: random.Random,
    result: DressingResult,
) -> None:
    """Place static cars + props on the FAR sidewalk relative to the relay.

    density_cfg keys:
        static_vehicles: int
        static_vehicle_blueprints: list[str]
        props: list[{blueprint, count}]
        ring_radius_m: float — used as the far-sidewalk offset distance
    """
    cx, cy, cz = look_at
    rx, ry, _rz = relay_location
    # Unit vector from relay to look_at, then keep going to land on the far
    # sidewalk.
    ax, ay = cx - rx, cy - ry
    away_offset = float(density_cfg.get("ring_radius_m", 5.0))

    n_veh = int(density_cfg.get("static_vehicles", 0))
    veh_bps = density_cfg.get(
        "static_vehicle_blueprints",
        ["vehicle.tesla.model3", "vehicle.audi.tt", "vehicle.mercedes.coupe",
         "vehicle.nissan.micra", "vehicle.mini.cooper_s", "vehicle.toyota.prius"],
    ) or []
    if n_veh > 0 and veh_bps:
        # Park static vehicles on the FAR sidewalk (across the lane from the
        # relay), so they appear behind/beside the lane in the relay image.
        positions = _far_sidewalk_positions((cx, cy), (ax, ay), away_offset, n_veh, z=cz - 1.0)
        for (x, y, z, yaw) in positions:
            bp_id = rng.choice(veh_bps)
            actor = _spawn_static_actor(world, bp_id, x, y, z, yaw, registry,
                                        role_name=f"static_veh_{relay_name}")
            if actor is not None:
                result.static_vehicles.append(actor)

    # Props: same far-sidewalk distribution, slightly farther out so they
    # don't overlap the static vehicles.
    prop_specs = density_cfg.get("props", []) or []
    prop_offset = away_offset + 2.5
    total_props = sum(int(p.get("count", 0)) for p in prop_specs)
    if total_props > 0:
        positions = _far_sidewalk_positions(
            (cx, cy), (ax, ay), prop_offset, total_props,
            z=cz - 1.5, yaw_offset_deg=rng.uniform(0.0, 30.0),
        )
        idx = 0
        for spec in prop_specs:
            bp_id = spec.get("blueprint")
            count = int(spec.get("count", 0))
            if not bp_id or count <= 0:
                continue
            for _ in range(count):
                x, y, z, yaw = positions[idx]
                idx += 1
                actor = _spawn_static_actor(world, bp_id, x, y, z, yaw, registry,
                                            role_name=f"prop_{relay_name}")
                if actor is not None:
                    result.static_props.append(actor)


# ------------------------------------------------------------------------ #
# Moving NPCs (vehicles + pedestrians)
# ------------------------------------------------------------------------ #
def _spawn_npc_vehicles(
    world: "carla.World",
    count: int,
    blueprint_filters: List[str],
    tm: Optional["carla.TrafficManager"],
    registry: ActorRegistry,
    rng: random.Random,
    result: DressingResult,
) -> None:
    if count <= 0 or carla is None:
        return
    bp_lib = world.get_blueprint_library()
    candidates: List["carla.ActorBlueprint"] = []
    for f in blueprint_filters:
        candidates.extend(list(bp_lib.filter(f)))
    if not candidates:
        LOGGER.warning("No blueprints matched filters %s; no NPC vehicles spawned.",
                       blueprint_filters)
        return

    spawn_points = world.get_map().get_spawn_points()
    rng.shuffle(spawn_points)

    spawned = 0
    for sp in spawn_points:
        if spawned >= count:
            break
        bp = rng.choice(candidates)
        if bp.has_attribute("role_name"):
            bp.set_attribute("role_name", f"npc_veh_{spawned:03d}")
        if bp.has_attribute("color"):
            recommended = bp.get_attribute("color").recommended_values
            if recommended:
                bp.set_attribute("color", rng.choice(recommended))
        actor = world.try_spawn_actor(bp, sp)
        if actor is None:
            continue
        registry.register(actor)
        result.npc_vehicles.append(actor)
        try:
            actor.set_autopilot(True, tm.get_port() if tm is not None else 8000)
        except Exception as e:  # pragma: no cover
            LOGGER.warning("set_autopilot failed for NPC %s: %s", actor.id, e)
        spawned += 1
    LOGGER.info("Spawned %d / %d moving NPC vehicles.", spawned, count)


def _spawn_walkers(
    world: "carla.World",
    count: int,
    registry: ActorRegistry,
    rng: random.Random,
    result: DressingResult,
) -> None:
    if count <= 0 or carla is None:
        return
    bp_lib = world.get_blueprint_library()
    walker_bps = list(bp_lib.filter("walker.pedestrian.*"))
    controller_bp = bp_lib.find("controller.ai.walker")
    if not walker_bps or controller_bp is None:
        LOGGER.warning("No walker blueprints; skipping pedestrians.")
        return

    # Find spawn points on the navigation mesh.
    targets: List["carla.Location"] = []
    for _ in range(count * 4):  # over-sample, navigation may return None
        loc = world.get_random_location_from_navigation()
        if loc is not None:
            targets.append(loc)
        if len(targets) >= count:
            break
    if not targets:
        LOGGER.warning("Navigation returned no walker spawn locations.")
        return

    walkers: List["carla.Actor"] = []
    for loc in targets[:count]:
        bp = rng.choice(walker_bps)
        if bp.has_attribute("is_invincible"):
            bp.set_attribute("is_invincible", "false")
        tf = carla.Transform(loc, carla.Rotation())
        actor = world.try_spawn_actor(bp, tf)
        if actor is None:
            continue
        registry.register(actor)
        walkers.append(actor)
        result.walkers.append(actor)

    # Now attach AI controllers and start them walking.
    for w in walkers:
        ctrl = world.try_spawn_actor(controller_bp, carla.Transform(), attach_to=w)
        if ctrl is None:
            continue
        registry.register(ctrl)
        result.walker_controllers.append(ctrl)
        try:
            ctrl.start()
            target = world.get_random_location_from_navigation()
            if target is not None:
                ctrl.go_to_location(target)
            ctrl.set_max_speed(1.4)  # ~normal walking
        except Exception as e:  # pragma: no cover
            LOGGER.warning("Walker controller setup failed: %s", e)
    LOGGER.info("Spawned %d walkers + %d controllers.",
                len(walkers), len(result.walker_controllers))


# ------------------------------------------------------------------------ #
# Public driver
# ------------------------------------------------------------------------ #
def dress_scene(
    world: "carla.World",
    scene_cfg: Dict[str, Any],
    relay_specs: List[Dict[str, Any]],
    tm: Optional["carla.TrafficManager"],
    registry: ActorRegistry,
    seed: Optional[int] = None,
) -> DressingResult:
    """Apply a full ``scene.dressing`` configuration to the world.

    Expected ``scene_cfg`` shape::

        moving_npc_vehicles: 12
        npc_vehicle_filters: ["vehicle.audi.*", "vehicle.tesla.*", ...]
        walkers: 18
        per_relay:
          relay_01: { static_vehicles: 3, props: [...] }
          ...
          default: { static_vehicles: 0, props: [] }
    """
    rng = random.Random(seed)
    result = DressingResult()

    per_relay = scene_cfg.get("per_relay", {}) or {}
    default_density = per_relay.get("default", {}) or {}
    for spec in relay_specs:
        name = spec["name"]
        look_at = spec.get("look_at", [0.0, 0.0, 1.5])
        relay_loc = spec.get("location", [0.0, 0.0, 8.0])
        density = per_relay.get(name, default_density)
        if not density:
            continue
        _dress_one_relay(
            world, name,
            (float(look_at[0]), float(look_at[1]), float(look_at[2])),
            (float(relay_loc[0]), float(relay_loc[1]), float(relay_loc[2])),
            density, registry, rng, result,
        )

    _spawn_npc_vehicles(
        world,
        int(scene_cfg.get("moving_npc_vehicles", 0)),
        list(scene_cfg.get("npc_vehicle_filters", ["vehicle.*"])),
        tm, registry, rng, result,
    )
    _spawn_walkers(
        world,
        int(scene_cfg.get("walkers", 0)),
        registry, rng, result,
    )

    LOGGER.info(
        "Scene dressing summary: static_vehicles=%d, static_props=%d, "
        "moving_npcs=%d, walkers=%d",
        len(result.static_vehicles), len(result.static_props),
        len(result.npc_vehicles), len(result.walkers),
    )
    return result


def stop_walker_controllers(result: DressingResult) -> None:
    """Stop all walker AI controllers (call before destroying actors)."""
    for ctrl in result.walker_controllers:
        try:
            ctrl.stop()
        except Exception:  # pragma: no cover
            pass
