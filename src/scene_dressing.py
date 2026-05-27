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


def _bike_rack_positions(
    center: Tuple[float, float],
    toward_dir: Tuple[float, float],
    offset_m: float,
    count: int,
    spacing_m: float = 1.5,
    z: float = 0.05,
) -> List[Tuple[float, float, float, float]]:
    """Place a row of bicycles on the NEAR sidewalk like a parking rack.

    All bikes share the same yaw (perpendicular to the street, pointing
    toward the lane) and are spaced ``spacing_m`` apart along the street,
    centered on ``center + offset_m * toward_dir``.
    """
    import math as _m
    out: List[Tuple[float, float, float, float]] = []
    cx, cy = center
    ax, ay = toward_dir
    norm = _m.hypot(ax, ay) or 1.0
    ax, ay = ax / norm, ay / norm
    px, py = -ay, ax  # along-street axis
    # Uniform yaw: bikes face the lane (i.e. opposite of toward_dir).
    yaw = _m.degrees(_m.atan2(-ay, -ax))
    for i in range(count):
        s = (i - (count - 1) / 2.0) * spacing_m
        x = cx + ax * offset_m + px * s
        y = cy + ay * offset_m + py * s
        out.append((x, y, z, yaw))
    return out


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
    when ``away_dir`` is the lane normal, this naturally becomes the local
    lane tangent.
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


def _nearest_path_tangent(
    path_points: List[Tuple[float, float]],
    query_xy: Tuple[float, float],
) -> Optional[Tuple[float, float]]:
    """Return the unit tangent of the path segment nearest to ``query_xy``."""
    if len(path_points) < 2:
        return None

    qx, qy = query_xy
    best_dist2 = float("inf")
    best_tangent: Optional[Tuple[float, float]] = None
    for (ax, ay), (bx, by) in zip(path_points, path_points[1:]):
        vx, vy = bx - ax, by - ay
        seg_len2 = vx * vx + vy * vy
        if seg_len2 <= 1e-9:
            continue
        wx, wy = qx - ax, qy - ay
        u = max(0.0, min(1.0, (wx * vx + wy * vy) / seg_len2))
        px, py = ax + u * vx, ay + u * vy
        dist2 = (qx - px) ** 2 + (qy - py) ** 2
        if dist2 < best_dist2:
            seg_len = math.sqrt(seg_len2)
            best_dist2 = dist2
            best_tangent = (vx / seg_len, vy / seg_len)
    return best_tangent


def _normalize_xy(vec: Tuple[float, float]) -> Optional[Tuple[float, float]]:
    x, y = vec
    norm = math.hypot(x, y)
    if norm <= 1e-9:
        return None
    return x / norm, y / norm


def _min_dist2_to_path(
    path_points: List[Tuple[float, float]],
    query_xy: Tuple[float, float],
) -> float:
    """Squared distance from a 2D point to the nearest segment of a polyline."""
    if not path_points:
        return float("inf")
    if len(path_points) == 1:
        dx = query_xy[0] - path_points[0][0]
        dy = query_xy[1] - path_points[0][1]
        return dx * dx + dy * dy

    qx, qy = query_xy
    best_dist2 = float("inf")
    for (ax, ay), (bx, by) in zip(path_points, path_points[1:]):
        vx, vy = bx - ax, by - ay
        seg_len2 = vx * vx + vy * vy
        if seg_len2 <= 1e-9:
            dx = qx - ax
            dy = qy - ay
            best_dist2 = min(best_dist2, dx * dx + dy * dy)
            continue
        wx, wy = qx - ax, qy - ay
        u = max(0.0, min(1.0, (wx * vx + wy * vy) / seg_len2))
        px, py = ax + u * vx, ay + u * vy
        dx = qx - px
        dy = qy - py
        best_dist2 = min(best_dist2, dx * dx + dy * dy)
    return best_dist2


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
    ugv_path: Optional[List[Tuple[float, float]]] = None,
    loop_center_xy: Optional[Tuple[float, float]] = None,
) -> None:
    """Place static cars + props around a relay without intruding into the lane.

    density_cfg keys:
        static_vehicles: int
        static_vehicle_blueprints: list[str]
        props: list[{blueprint, count}]
        ring_radius_m: float — far-sidewalk offset for static vehicles / props
        bike_offset_m: float — near-sidewalk offset for two-wheeler vehicles
        bike_spacing_m: float — gap between adjacent bikes in the rack
    """
    cx, cy, cz = look_at
    rx, ry, _rz = relay_location
    toward_dir = (rx - cx, ry - cy)
    bike_dir = _normalize_xy(toward_dir) or (0.0, 1.0)
    outward_dir = None
    if loop_center_xy is not None:
        outward_dir = _normalize_xy((cx - loop_center_xy[0], cy - loop_center_xy[1]))
    tangent = _nearest_path_tangent(ugv_path or [], (cx, cy))
    if tangent is not None:
        tx, ty = tangent
        nx, ny = -ty, tx
        if outward_dir is not None and nx * outward_dir[0] + ny * outward_dir[1] < 0.0:
            nx, ny = -nx, -ny
        elif outward_dir is None and nx * toward_dir[0] + ny * toward_dir[1] < 0.0:
            nx, ny = -nx, -ny
        outward_dir = (nx, ny)
    if outward_dir is None:
        outward_dir = bike_dir
    if bike_dir[0] * outward_dir[0] + bike_dir[1] * outward_dir[1] < 0.25:
        bike_dir = outward_dir
    away_offset = float(density_cfg.get("ring_radius_m", 5.0))

    vehicles_before = len(result.static_vehicles)
    n_veh = int(density_cfg.get("static_vehicles", 0))
    veh_bps = density_cfg.get(
        "static_vehicle_blueprints",
        ["vehicle.tesla.model3", "vehicle.audi.tt", "vehicle.mercedes.coupe",
         "vehicle.nissan.micra", "vehicle.mini.cooper_s", "vehicle.toyota.prius"],
    ) or []
    if n_veh > 0 and veh_bps:
        positions = _far_sidewalk_positions(
            (cx, cy), outward_dir, away_offset, n_veh, z=max(cz + 0.2, 0.2),
        )
        for (x, y, z, yaw) in positions:
            bp_id = rng.choice(veh_bps)
            actor = _spawn_static_actor(world, bp_id, x, y, z, yaw, registry,
                                        role_name=f"static_veh_{relay_name}")
            if actor is not None:
                result.static_vehicles.append(actor)
    vehicles_spawned = len(result.static_vehicles) - vehicles_before

    # Props: split by type.
    #   * Two-wheeler vehicle blueprints (bike/scooter/moto) -> NEAR sidewalk
    #     in a tidy bike rack (uniform yaw, tight spacing).
    #   * Everything else (static.prop.*) -> FAR sidewalk.
    prop_specs = density_cfg.get("props", []) or []
    bike_prefixes = (
        "vehicle.diamondback.", "vehicle.gazelle.", "vehicle.vespa.",
        "vehicle.kawasaki.", "vehicle.bh.", "vehicle.harley-davidson.",
        "vehicle.yamaha.",
    )
    def _is_bike(bp_id: str) -> bool:
        return any(bp_id.startswith(p) for p in bike_prefixes)

    far_specs  = [s for s in prop_specs if not _is_bike(s.get("blueprint", ""))]
    near_specs = [s for s in prop_specs if     _is_bike(s.get("blueprint", ""))]
    bike_offset  = float(density_cfg.get("bike_offset_m", 4.0))
    bike_spacing = float(density_cfg.get("bike_spacing_m", 1.5))

    n_bikes_requested = sum(int(p.get("count", 0)) for p in near_specs)
    n_props_requested = sum(int(p.get("count", 0)) for p in far_specs)

    def _spawn_from_positions(specs, positions) -> int:
        n = 0
        idx = 0
        for spec in specs:
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
                    n += 1
        return n

    far_spawned = 0
    if far_specs:
        far_offset = away_offset + 2.5
        positions = _far_sidewalk_positions(
            (cx, cy), outward_dir, far_offset, n_props_requested,
            z=max(cz + 0.1, 0.1), yaw_offset_deg=rng.uniform(0.0, 30.0),
        )
        far_spawned = _spawn_from_positions(far_specs, positions)

    bike_spawned = 0
    if near_specs:
        positions = _bike_rack_positions(
            (cx, cy), bike_dir, bike_offset, n_bikes_requested,
            spacing_m=bike_spacing, z=max(cz + 0.1, 0.1),
        )
        bike_spawned = _spawn_from_positions(near_specs, positions)

    LOGGER.info(
        "Per-relay %s: static_veh=%d/%d, far_props=%d/%d, bikes=%d/%d",
        relay_name, vehicles_spawned, n_veh,
        far_spawned, n_props_requested,
        bike_spawned, n_bikes_requested,
    )


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
    ugv_xy: Optional[Tuple[float, float]] = None,
    keepout_radius_m: float = 15.0,
    ugv_path: Optional[List[Tuple[float, float]]] = None,
    route_keepout_radius_m: float = 0.0,
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
    if ugv_xy is not None and keepout_radius_m > 0:
        ux, uy = ugv_xy
        r2 = keepout_radius_m * keepout_radius_m
        before = len(spawn_points)
        spawn_points = [
            sp for sp in spawn_points
            if (sp.location.x - ux) ** 2 + (sp.location.y - uy) ** 2 > r2
        ]
        LOGGER.info(
            "NPC keepout: dropped %d / %d spawn points within %.1f m of UGV (%.2f, %.2f).",
            before - len(spawn_points), before, keepout_radius_m, ux, uy,
        )
    if ugv_path and route_keepout_radius_m > 0:
        r2 = route_keepout_radius_m * route_keepout_radius_m
        before = len(spawn_points)
        spawn_points = [
            sp for sp in spawn_points
            if _min_dist2_to_path(ugv_path, (sp.location.x, sp.location.y)) > r2
        ]
        LOGGER.info(
            "NPC route keepout: dropped %d / %d spawn points within %.1f m of UGV path.",
            before - len(spawn_points), before, route_keepout_radius_m,
        )
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
    ugv_path: Optional[List[Tuple[float, float]]] = None,
    keepout_radius_m: float = 0.0,
    target_keepout_radius_m: Optional[float] = None,
) -> None:
    if count <= 0 or carla is None:
        return
    bp_lib = world.get_blueprint_library()
    walker_bps = list(bp_lib.filter("walker.pedestrian.*"))
    controller_bp = bp_lib.find("controller.ai.walker")
    if not walker_bps or controller_bp is None:
        LOGGER.warning("No walker blueprints; skipping pedestrians.")
        return

    def _sample_nav_location(route_keepout_m: float) -> Optional["carla.Location"]:
        for _ in range(12):
            loc = world.get_random_location_from_navigation()
            if loc is None:
                continue
            if ugv_path and route_keepout_m > 0:
                if _min_dist2_to_path(ugv_path, (loc.x, loc.y)) <= route_keepout_m * route_keepout_m:
                    continue
            return loc
        return None

    # Find spawn points on the navigation mesh.
    targets: List["carla.Location"] = []
    for _ in range(count * 4):  # over-sample, navigation may return None
        loc = _sample_nav_location(keepout_radius_m)
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
            keepout_m = target_keepout_radius_m if target_keepout_radius_m is not None else keepout_radius_m
            target = _sample_nav_location(keepout_m)
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
    ugv_xy: Optional[Tuple[float, float]] = None,
    ugv_path: Optional[List[Tuple[float, float]]] = None,
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
    loop_center_xy: Optional[Tuple[float, float]] = None
    if ugv_path:
        xs = [p[0] for p in ugv_path]
        ys = [p[1] for p in ugv_path]
        loop_center_xy = ((min(xs) + max(xs)) / 2.0, (min(ys) + max(ys)) / 2.0)
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
            density, registry, rng, result, ugv_path=ugv_path,
            loop_center_xy=loop_center_xy,
        )

    _spawn_npc_vehicles(
        world,
        int(scene_cfg.get("moving_npc_vehicles", 0)),
        list(scene_cfg.get("npc_vehicle_filters", ["vehicle.*"])),
        tm, registry, rng, result,
        ugv_xy=ugv_xy,
        keepout_radius_m=float(scene_cfg.get("npc_keepout_radius_m", 15.0)),
        ugv_path=ugv_path,
        route_keepout_radius_m=float(scene_cfg.get("npc_route_keepout_radius_m", 0.0)),
    )
    _spawn_walkers(
        world,
        int(scene_cfg.get("walkers", 0)),
        registry, rng, result,
        ugv_path=ugv_path,
        keepout_radius_m=float(scene_cfg.get("walker_route_keepout_radius_m", 0.0)),
        target_keepout_radius_m=float(
            scene_cfg.get(
                "walker_target_keepout_radius_m",
                scene_cfg.get("walker_route_keepout_radius_m", 0.0),
            )
        ),
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
