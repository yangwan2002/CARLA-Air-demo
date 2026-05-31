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

# Props with large colliders — placed further from the lane centre than normal
# far-sidewalk props (only matters when ugv_path matches the outer loop).
_BLOCKING_PROP_IDS = frozenset({
    "static.prop.kiosk_01",
    "static.prop.streetbarrier",
})


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
    vehicle_snap_margin_m: float = 0.02,
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

    is_vehicle = bp_id.startswith("vehicle.")
    surf_z = _ground_z_near(world, x, y, float(z), lift=0.0)
    spawn_z = surf_z + (0.6 if is_vehicle else 0.05)
    tf = carla.Transform(
        carla.Location(x=float(x), y=float(y), z=spawn_z),
        carla.Rotation(roll=0.0, pitch=0.0, yaw=float(yaw_deg)),
    )

    actor = world.try_spawn_actor(bp, tf)
    if actor is None:
        for dx, dy, dz in [
            (0.0, 0.0, -0.25),
            (0.0, 0.0, 0.25),
            (0.5, 0.0, 0.0),
            (-0.5, 0.0, 0.0),
            (0.0, 0.5, 0.0),
            (0.0, -0.5, 0.0),
            (0.5, 0.0, -0.15),
            (-0.5, 0.0, -0.15),
            (0.0, 0.5, -0.15),
            (0.0, -0.5, -0.15),
            (1.0, 0.0, 0.0),
            (-1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            (0.0, -1.0, 0.0),
        ]:
            tf.location.x = float(x) + dx
            tf.location.y = float(y) + dy
            tf.location.z = spawn_z + dz
            actor = world.try_spawn_actor(bp, tf)
            if actor is not None:
                break
    if actor is None:
        LOGGER.debug("Could not spawn %s near (%.1f, %.1f) after jitters.", bp_id, x, y)
        return None

    try:
        tf.location.x = float(x)
        tf.location.y = float(y)
        if is_vehicle:
            _snap_static_vehicle(
                world, actor, surf_z, bp_id=bp_id, margin_m=vehicle_snap_margin_m,
            )
        else:
            tf.location.z = surf_z + 0.03
            actor.set_transform(tf)
    except Exception:
        pass

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


def _neg_xy(vec: Tuple[float, float]) -> Tuple[float, float]:
    return -vec[0], -vec[1]


def _ground_z_near(
    world: "carla.World",
    x: float,
    y: float,
    fallback_z: float,
    lift: float = 0.0,
) -> float:
    """Project an XY point to the nearest map-supported surface height."""
    if carla is None:
        return float(fallback_z)

    try:
        hits = world.cast_ray(
            carla.Location(x=float(x), y=float(y), z=50.0),
            carla.Location(x=float(x), y=float(y), z=-10.0),
        )
        if hits:
            preferred_hits = [
                hit for hit in hits
                if any(
                    token in str(hit.label)
                    for token in ("Road", "Sidewalk", "Terrain", "Ground")
                )
            ]
            ground_hit = min(preferred_hits or hits, key=lambda hit: hit.location.z)
            return float(ground_hit.location.z + lift)
    except Exception:
        pass

    world_map = world.get_map()
    lane_type = getattr(getattr(carla, "LaneType", None), "Any", None)
    probe_zs = [fallback_z, 0.0, 0.2, -0.5, 1.0]
    for probe_z in probe_zs:
        try:
            loc = carla.Location(x=float(x), y=float(y), z=float(probe_z))
            if lane_type is not None:
                waypoint = world_map.get_waypoint(
                    loc,
                    project_to_road=True,
                    lane_type=lane_type,
                )
            else:
                waypoint = world_map.get_waypoint(loc, project_to_road=True)
        except Exception:
            continue
        if waypoint is not None:
            return float(waypoint.transform.location.z + lift)
    return float(fallback_z)


def _vehicle_half_height(actor: "carla.Actor", bp_id: str = "") -> float:
    if hasattr(actor, "bounding_box"):
        half_h = float(actor.bounding_box.extent.z)
        if half_h > 0.02:
            return half_h
    two_wheel = (
        "vehicle.diamondback.", "vehicle.gazelle.", "vehicle.vespa.",
        "vehicle.kawasaki.", "vehicle.bh.", "vehicle.yamaha.",
        "vehicle.harley-davidson.",
    )
    if bp_id.startswith(two_wheel):
        return 0.45
    return 0.35


def _snap_static_vehicle(
    world: "carla.World",
    actor: "carla.Actor",
    surf_z: float,
    bp_id: str = "",
    margin_m: float = 0.02,
) -> None:
    """Re-seat a physics-off static vehicle onto ``surf_z``."""
    if carla is None:
        return
    try:
        tf = actor.get_transform()
        half_h = _vehicle_half_height(actor, bp_id=bp_id)
        tf.location.z = float(surf_z) + half_h + float(margin_m)
        actor.set_transform(tf)
    except Exception as exc:  # pragma: no cover
        LOGGER.debug("Static vehicle ground snap failed for %s: %s", actor.id, exc)


def _reconcile_static_vehicles(
    world: "carla.World",
    actors: List["carla.Actor"],
    margin_m: float = 0.02,
) -> None:
    """Post-pass ground snap for physics-off parked vehicles only."""
    for actor in actors:
        try:
            bp_id = actor.type_id if hasattr(actor, "type_id") else ""
        except Exception:
            bp_id = ""
        try:
            tf = actor.get_transform()
            surf_z = _ground_z_near(
                world, tf.location.x, tf.location.y, tf.location.z, lift=0.0,
            )
            _snap_static_vehicle(world, actor, surf_z, bp_id=bp_id, margin_m=margin_m)
        except Exception:
            pass


def _min_dist_to_path(
    path_points: List[Tuple[float, float]],
    query_xy: Tuple[float, float],
) -> float:
    """Distance from a 2D point to the nearest segment of a polyline."""
    return math.sqrt(_min_dist2_to_path(path_points, query_xy))


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


def _sample_points_along_path(
    path_points: List[Tuple[float, float]],
    spacing_m: float,
) -> List[Tuple[float, float]]:
    """Return roughly ``spacing_m``-spaced (x, y) samples along a polyline."""
    if not path_points:
        return []
    if len(path_points) == 1 or spacing_m <= 0.0:
        return list(path_points)

    spacing_m = max(float(spacing_m), 1.0)
    out: List[Tuple[float, float]] = []
    for (ax, ay), (bx, by) in zip(path_points, path_points[1:]):
        vx, vy = bx - ax, by - ay
        seg_len = math.hypot(vx, vy)
        if seg_len <= 1e-6:
            continue
        n_steps = max(1, int(math.ceil(seg_len / spacing_m)))
        for i in range(n_steps):
            t = i / n_steps
            out.append((ax + t * vx, ay + t * vy))
    out.append(path_points[-1])
    return out


def _within_ugv_path_corridor(
    xy: Tuple[float, float],
    ugv_path: Optional[List[Tuple[float, float]]],
    radius_m: float,
) -> bool:
    if not ugv_path or radius_m <= 0.0:
        return True
    return _min_dist_to_path(ugv_path, xy) <= float(radius_m)


def _waypoint_dist_to_path(
    wp: "carla.Waypoint",
    ugv_path: List[Tuple[float, float]],
) -> float:
    loc = wp.transform.location
    return _min_dist_to_path(ugv_path, (loc.x, loc.y))


def _pick_npc_driving_waypoint(
    wp: "carla.Waypoint",
    ugv_path: Optional[List[Tuple[float, float]]],
    path_center_keepout_m: float,
    rng: random.Random,
) -> Optional["carla.Waypoint"]:
    """Pick a Driving lane that keeps TM traffic off the UGV centerline."""
    if carla is None or wp is None or wp.lane_type != carla.LaneType.Driving:
        return None
    if not ugv_path or path_center_keepout_m <= 0.0:
        return wp

    candidates: List["carla.Waypoint"] = []
    seen_ids: set = set()

    def _add(w: Optional["carla.Waypoint"]) -> None:
        if w is None or w.lane_type != carla.LaneType.Driving:
            return
        loc = w.transform.location
        key = (round(loc.x, 1), round(loc.y, 1))
        if key in seen_ids:
            return
        seen_ids.add(key)
        candidates.append(w)

    _add(wp)
    for getter in (wp.get_left_lane, wp.get_right_lane):
        side = getter()
        _add(side)
        if side is not None:
            _add(side.get_left_lane())
            _add(side.get_right_lane())

    safe = [
        w for w in candidates
        if _waypoint_dist_to_path(w, ugv_path) >= float(path_center_keepout_m)
    ]
    if safe:
        return rng.choice(safe)
    if candidates:
        return max(candidates, key=lambda w: _waypoint_dist_to_path(w, ugv_path))
    return None


def _route_driving_spawn_transforms(
    world: "carla.World",
    ugv_path: List[Tuple[float, float]],
    spacing_m: float,
    ugv_xy: Optional[Tuple[float, float]] = None,
    ugv_keepout_m: float = 0.0,
    path_center_keepout_m: float = 0.0,
    rng: Optional[random.Random] = None,
) -> List["carla.Transform"]:
    """Build Driving-lane spawn transforms sampled along the UGV polyline."""
    if carla is None or not ugv_path:
        return []

    world_map = world.get_map()
    candidates: List["carla.Transform"] = []
    seen: set = set()
    for x, y in _sample_points_along_path(ugv_path, spacing_m):
        if ugv_xy is not None and ugv_keepout_m > 0.0:
            ux, uy = ugv_xy
            dx, dy = x - ux, y - uy
            if dx * dx + dy * dy <= ugv_keepout_m * ugv_keepout_m:
                continue
        try:
            wp = world_map.get_waypoint(
                carla.Location(x=float(x), y=float(y), z=0.5),
                project_to_road=True,
                lane_type=carla.LaneType.Driving,
            )
        except Exception:
            wp = world_map.get_waypoint(
                carla.Location(x=float(x), y=float(y), z=0.5),
                project_to_road=True,
            )
        if wp is None or wp.lane_type != carla.LaneType.Driving:
            continue
        wp = _pick_npc_driving_waypoint(
            wp, ugv_path, path_center_keepout_m, rng or random.Random(0),
        )
        if wp is None:
            continue
        loc = wp.transform.location
        key = (round(loc.x, 0), round(loc.y, 0))
        if key in seen:
            continue
        seen.add(key)
        candidates.append(wp.transform)
    return candidates


def _sidewalk_locations_along_route(
    world: "carla.World",
    ugv_path: List[Tuple[float, float]],
    spacing_m: float,
    sidewalk_offset_m: float = 3.8,
    ugv_xy: Optional[Tuple[float, float]] = None,
    ugv_keepout_m: float = 0.0,
) -> List[Tuple[float, float, float]]:
    """Return sidewalk (x, y, z) samples near the UGV route."""
    if carla is None or not ugv_path:
        return []

    world_map = world.get_map()
    out: List[Tuple[float, float, float]] = []
    seen: set = set()
    for x, y in _sample_points_along_path(ugv_path, spacing_m):
        if ugv_xy is not None and ugv_keepout_m > 0.0:
            ux, uy = ugv_xy
            dx, dy = x - ux, y - uy
            if dx * dx + dy * dy <= ugv_keepout_m * ugv_keepout_m:
                continue
        try:
            wp = world_map.get_waypoint(
                carla.Location(x=float(x), y=float(y), z=0.5),
                project_to_road=True,
                lane_type=carla.LaneType.Driving,
            )
        except Exception:
            wp = world_map.get_waypoint(
                carla.Location(x=float(x), y=float(y), z=0.5),
                project_to_road=True,
            )
        if wp is None:
            continue
        yaw_rad = math.radians(float(wp.transform.rotation.yaw))
        nx, ny = -math.sin(yaw_rad), math.cos(yaw_rad)
        cx, cy = float(wp.transform.location.x), float(wp.transform.location.y)
        for sign in (1.0, -1.0):
            sx = cx + sign * nx * sidewalk_offset_m
            sy = cy + sign * ny * sidewalk_offset_m
            key = (round(sx, 0), round(sy, 0))
            if key in seen:
                continue
            seen.add(key)
            surf_z = _ground_z_near(world, sx, sy, wp.transform.location.z, lift=0.0)
            out.append((sx, sy, surf_z))
    return out


def _npc_spawn_on_driving_lane(
    world: "carla.World",
    sp: "carla.Transform",
    vehicle_lift_m: float = 0.3,
    max_catalog_z_delta_m: float = 2.0,
) -> Optional["carla.Transform"]:
    """Snap a map spawn point onto a Driving lane with raycast ground Z.

    Town10HD has catalog spawn points on elevated plazas / ramps whose Z does
    not match the drivable surface below.  Static parked cars already use
    ``_ground_z_near``; moving NPCs must do the same or a fraction will float.
    """
    if carla is None:
        return sp

    world_map = world.get_map()
    try:
        wp = world_map.get_waypoint(
            sp.location,
            project_to_road=True,
            lane_type=carla.LaneType.Driving,
        )
    except Exception:
        wp = world_map.get_waypoint(sp.location, project_to_road=True)

    if wp is None or wp.lane_type != carla.LaneType.Driving:
        return None

    loc = wp.transform.location
    lane_z = float(loc.z)
    if abs(float(sp.location.z) - lane_z) > float(max_catalog_z_delta_m):
        return None

    surf_z = _ground_z_near(world, float(loc.x), float(loc.y), lane_z, lift=0.0)
    if abs(surf_z - lane_z) > 0.6:
        return None

    center_z = surf_z + float(vehicle_lift_m)
    return carla.Transform(
        carla.Location(x=float(loc.x), y=float(loc.y), z=center_z),
        carla.Rotation(
            roll=float(sp.rotation.roll),
            pitch=float(sp.rotation.pitch),
            yaw=float(wp.transform.rotation.yaw),
        ),
    )


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


def _clear_of_ugv_path(
    xy: Tuple[float, float],
    ugv_path: Optional[List[Tuple[float, float]]],
    keepout_m: float,
) -> bool:
    """True when ``xy`` is at least ``keepout_m`` away from the UGV polyline."""
    if not ugv_path or keepout_m <= 0.0:
        return True
    return _min_dist_to_path(ugv_path, xy) >= keepout_m


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
    vehicle_snap_margin_m: float = 0.02,
    static_route_keepout_m: float = 0.0,
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
    relay_side_dir = _normalize_xy(toward_dir) or (0.0, 1.0)
    tangent = _nearest_path_tangent(ugv_path or [], (cx, cy))
    if tangent is not None:
        _tx, ty = tangent
        tx = tangent[0]
        nx, ny = -ty, tx
        if nx * toward_dir[0] + ny * toward_dir[1] < 0.0:
            nx, ny = -nx, -ny
        relay_side_dir = (nx, ny)
    curb_side_dir = relay_side_dir

    away_offset = float(density_cfg.get("ring_radius_m", 5.0))
    bike_offset = float(density_cfg.get("bike_offset_m", 4.0))
    bike_sidewalk_margin = float(density_cfg.get("bike_sidewalk_margin_m", 0.8))
    relay_side_clearance = toward_dir[0] * relay_side_dir[0] + toward_dir[1] * relay_side_dir[1]
    if relay_side_clearance > bike_sidewalk_margin + 0.5:
        bike_offset = max(0.5, relay_side_clearance - bike_sidewalk_margin)

    vehicles_before = len(result.static_vehicles)
    n_veh = int(density_cfg.get("static_vehicles", 0))
    veh_bps = density_cfg.get(
        "static_vehicle_blueprints",
        ["vehicle.tesla.model3", "vehicle.audi.tt", "vehicle.mercedes.coupe",
         "vehicle.nissan.micra", "vehicle.mini.cooper_s", "vehicle.toyota.prius"],
    ) or []
    if n_veh > 0 and veh_bps:
        positions = _far_sidewalk_positions(
            (cx, cy), curb_side_dir, away_offset + 2.5, n_veh, z=max(cz + 0.2, 0.2),
        )
        for (x, y, z, yaw) in positions:
            if not _clear_of_ugv_path((x, y), ugv_path, static_route_keepout_m):
                LOGGER.debug(
                    "Per-relay %s: skip static veh at (%.1f, %.1f) — UGV path keepout.",
                    relay_name, x, y,
                )
                continue
            surf_z = _ground_z_near(world, x, y, z, lift=0.0)
            bp_id = rng.choice(veh_bps)
            actor = _spawn_static_actor(
                world, bp_id, x, y, surf_z, yaw, registry,
                role_name=f"static_veh_{relay_name}",
                vehicle_snap_margin_m=vehicle_snap_margin_m,
            )
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
                if idx >= len(positions):
                    break
                x, y, z, yaw = positions[idx]
                idx += 1
                if not _clear_of_ugv_path((x, y), ugv_path, static_route_keepout_m):
                    LOGGER.debug(
                        "Per-relay %s: skip prop %s at (%.1f, %.1f) — UGV path keepout.",
                        relay_name, bp_id, x, y,
                    )
                    continue
                surf_z = _ground_z_near(world, x, y, z, lift=0.0)
                actor = _spawn_static_actor(
                    world,
                    bp_id,
                    x,
                    y,
                    surf_z,
                    yaw,
                    registry,
                    role_name=f"prop_{relay_name}",
                    vehicle_snap_margin_m=vehicle_snap_margin_m,
                )
                if actor is not None:
                    result.static_props.append(actor)
                    n += 1
        return n

    far_spawned = 0
    if far_specs:
        normal_specs = [
            s for s in far_specs if s.get("blueprint") not in _BLOCKING_PROP_IDS
        ]
        blocking_specs = [
            s for s in far_specs if s.get("blueprint") in _BLOCKING_PROP_IDS
        ]
        if normal_specs:
            n_normal = sum(int(p.get("count", 0)) for p in normal_specs)
            positions = _far_sidewalk_positions(
                (cx, cy), curb_side_dir, away_offset + 2.5, n_normal,
                z=max(cz + 0.1, 0.1), yaw_offset_deg=rng.uniform(0.0, 30.0),
            )
            far_spawned += _spawn_from_positions(normal_specs, positions)
        if blocking_specs:
            n_block = sum(int(p.get("count", 0)) for p in blocking_specs)
            positions = _far_sidewalk_positions(
                (cx, cy), curb_side_dir, away_offset + 4.5, n_block,
                z=max(cz + 0.1, 0.1), yaw_offset_deg=rng.uniform(0.0, 30.0),
            )
            far_spawned += _spawn_from_positions(blocking_specs, positions)

    bike_spawned = 0
    if near_specs:
        positions = _bike_rack_positions(
            (cx, cy), relay_side_dir, bike_offset, n_bikes_requested,
            spacing_m=bike_spacing, z=max(cz + 0.1, 0.1),
        )
        positions = [
            pos for pos in positions
            if _clear_of_ugv_path((pos[0], pos[1]), ugv_path, static_route_keepout_m)
        ]
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
def _npc_grounding_ok(
    world: "carla.World",
    actor: "carla.Actor",
    vehicle_lift_m: float,
    max_z_err_m: float,
) -> bool:
    """Return False when an autopilot NPC is clearly floating above the road."""
    if carla is None:
        return True
    try:
        tf = actor.get_transform()
        surf_z = _ground_z_near(
            world, tf.location.x, tf.location.y, tf.location.z, lift=0.0,
        )
        expected_z = surf_z + float(vehicle_lift_m)
        return abs(float(tf.location.z) - expected_z) <= float(max_z_err_m)
    except Exception:
        return True


def _destroy_npc_actor(
    actor: "carla.Actor",
    registry: ActorRegistry,
    result: DressingResult,
) -> None:
    try:
        actor.set_autopilot(False)
    except Exception:
        pass
    try:
        if actor in result.npc_vehicles:
            result.npc_vehicles.remove(actor)
    except Exception:
        pass
    try:
        if actor in registry._actors:
            registry._actors.remove(actor)
    except Exception:
        pass
    try:
        if actor.is_alive:
            actor.destroy()
    except Exception:
        pass


def _cull_ungrounded_npcs(
    world: "carla.World",
    result: DressingResult,
    registry: ActorRegistry,
    vehicle_lift_m: float,
    max_z_err_m: float,
) -> int:
    """Remove floating autopilot NPCs without post-spawn transform snaps."""
    culled = 0
    for actor in list(result.npc_vehicles):
        if _npc_grounding_ok(world, actor, vehicle_lift_m, max_z_err_m):
            continue
        try:
            loc = actor.get_transform().location
            LOGGER.warning(
                "Culling ungrounded NPC %s at (%.1f, %.1f, %.1f).",
                actor.id, loc.x, loc.y, loc.z,
            )
        except Exception:
            LOGGER.warning("Culling ungrounded NPC %s.", actor.id)
        _destroy_npc_actor(actor, registry, result)
        culled += 1
    if culled:
        LOGGER.info("Removed %d ungrounded moving NPC vehicle(s).", culled)
    return culled


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
    route_spawn_radius_m: float = 0.0,
    route_spawn_spacing_m: float = 12.0,
    route_keepout_radius_m: float = 0.0,
    path_center_keepout_m: float = 0.0,
    vehicle_lift_m: float = 0.3,
    max_catalog_z_delta_m: float = 2.0,
    vehicle_snap_margin_m: float = 0.02,
    max_spawn_z_err_m: float = 1.2,
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

    route_mode = bool(ugv_path and route_spawn_radius_m > 0.0)
    if route_mode:
        before = len(spawn_points)
        spawn_points = [
            sp for sp in spawn_points
            if _within_ugv_path_corridor(
                (sp.location.x, sp.location.y), ugv_path, route_spawn_radius_m,
            )
        ]
        LOGGER.info(
            "NPC route corridor: kept %d / %d catalog spawn points within %.1f m of UGV path.",
            len(spawn_points), before, route_spawn_radius_m,
        )
        route_tfs = _route_driving_spawn_transforms(
            world, ugv_path,
            spacing_m=route_spawn_spacing_m,
            ugv_xy=ugv_xy,
            ugv_keepout_m=keepout_radius_m,
            path_center_keepout_m=path_center_keepout_m,
            rng=rng,
        )
        LOGGER.info(
            "NPC route corridor: generated %d driving-lane samples along UGV path.",
            len(route_tfs),
        )
    elif ugv_path and route_keepout_radius_m > 0:
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
        route_tfs = []
    else:
        route_tfs = []

    rng.shuffle(spawn_points)

    spawned = 0
    skipped_bad_ground = 0

    def _try_spawn_at_transform(sp_tf: "carla.Transform") -> bool:
        nonlocal spawned, skipped_bad_ground
        if spawned >= count:
            return False
        if ugv_path and path_center_keepout_m > 0.0:
            try:
                wp = world.get_map().get_waypoint(
                    sp_tf.location,
                    project_to_road=True,
                    lane_type=carla.LaneType.Driving,
                )
            except Exception:
                wp = world.get_map().get_waypoint(sp_tf.location, project_to_road=True)
            wp = _pick_npc_driving_waypoint(wp, ugv_path, path_center_keepout_m, rng)
            if wp is None:
                skipped_bad_ground += 1
                return False
            sp_tf = wp.transform
        grounded_tf = _npc_spawn_on_driving_lane(
            world, sp_tf,
            vehicle_lift_m=vehicle_lift_m,
            max_catalog_z_delta_m=max_catalog_z_delta_m,
        )
        if grounded_tf is None:
            skipped_bad_ground += 1
            return False
        bp = rng.choice(candidates)
        if bp.has_attribute("role_name"):
            bp.set_attribute("role_name", f"npc_veh_{spawned:03d}")
        if bp.has_attribute("color"):
            recommended = bp.get_attribute("color").recommended_values
            if recommended:
                bp.set_attribute("color", rng.choice(recommended))
        actor = world.try_spawn_actor(bp, grounded_tf)
        if actor is None:
            return False
        if not _npc_grounding_ok(world, actor, vehicle_lift_m, max_spawn_z_err_m):
            try:
                loc = actor.get_transform().location
                LOGGER.warning(
                    "NPC spawn rejected (bad Z) at (%.1f, %.1f, %.1f); destroying.",
                    loc.x, loc.y, loc.z,
                )
            except Exception:
                LOGGER.warning("NPC spawn rejected (bad Z); destroying.")
            try:
                if actor.is_alive:
                    actor.destroy()
            except Exception:
                pass
            return False
        registry.register(actor)
        result.npc_vehicles.append(actor)
        try:
            port = tm.get_port() if tm is not None else 8000
            actor.set_autopilot(True, port)
            if tm is not None:
                if hasattr(tm, "ignore_walkers_percentage"):
                    tm.ignore_walkers_percentage(actor, 100.0)
                if hasattr(tm, "ignore_vehicles_percentage"):
                    tm.ignore_vehicles_percentage(actor, 100.0)
                if hasattr(tm, "distance_to_leading_vehicle"):
                    tm.distance_to_leading_vehicle(actor, 4.0)
        except Exception as e:  # pragma: no cover
            LOGGER.warning("set_autopilot failed for NPC %s: %s", actor.id, e)
        spawned += 1
        return True

    if route_mode:
        rng.shuffle(route_tfs)
        for sp_tf in route_tfs:
            if spawned >= count:
                break
            _try_spawn_at_transform(sp_tf)

    for sp in spawn_points:
        if spawned >= count:
            break
        _try_spawn_at_transform(sp)
    if skipped_bad_ground:
        LOGGER.info(
            "NPC spawn: skipped %d catalog points (non-driving or elevated Z).",
            skipped_bad_ground,
        )
    LOGGER.info("Spawned %d / %d moving NPC vehicles.", spawned, count)


def _spawn_walkers(
    world: "carla.World",
    count: int,
    registry: ActorRegistry,
    rng: random.Random,
    result: DressingResult,
    ugv_path: Optional[List[Tuple[float, float]]] = None,
    route_spawn_radius_m: float = 0.0,
    route_spawn_spacing_m: float = 10.0,
    route_keepout_radius_m: float = 0.0,
    target_route_spawn_radius_m: Optional[float] = None,
    ugv_xy: Optional[Tuple[float, float]] = None,
    spawn_keepout_radius_m: float = 0.0,
) -> None:
    if count <= 0 or carla is None:
        return
    bp_lib = world.get_blueprint_library()
    walker_bps = list(bp_lib.filter("walker.pedestrian.*"))
    controller_bp = bp_lib.find("controller.ai.walker")
    if not walker_bps or controller_bp is None:
        LOGGER.warning("No walker blueprints; skipping pedestrians.")
        return

    route_mode = bool(ugv_path and route_spawn_radius_m > 0.0)
    target_radius = (
        target_route_spawn_radius_m
        if target_route_spawn_radius_m is not None
        else route_spawn_radius_m
    )

    def _sample_nav_location(require_near_route: bool) -> Optional["carla.Location"]:
        for _ in range(24):
            loc = world.get_random_location_from_navigation()
            if loc is None:
                continue
            if ugv_xy is not None and spawn_keepout_radius_m > 0:
                ux, uy = ugv_xy
                dx = loc.x - ux
                dy = loc.y - uy
                if dx * dx + dy * dy <= spawn_keepout_radius_m * spawn_keepout_radius_m:
                    continue
            if ugv_path and require_near_route and route_spawn_radius_m > 0:
                if not _within_ugv_path_corridor(
                    (loc.x, loc.y), ugv_path, route_spawn_radius_m,
                ):
                    continue
            elif ugv_path and route_keepout_radius_m > 0:
                if _min_dist2_to_path(ugv_path, (loc.x, loc.y)) <= route_keepout_radius_m * route_keepout_radius_m:
                    continue
            return loc
        return None

    targets: List["carla.Location"] = []
    if route_mode:
        sidewalk_pts = _sidewalk_locations_along_route(
            world, ugv_path,
            spacing_m=route_spawn_spacing_m,
            ugv_xy=ugv_xy,
            ugv_keepout_m=spawn_keepout_radius_m,
        )
        rng.shuffle(sidewalk_pts)
        LOGGER.info(
            "Walker route corridor: %d sidewalk samples along UGV path.",
            len(sidewalk_pts),
        )
        for x, y, z in sidewalk_pts[:count]:
            targets.append(carla.Location(x=float(x), y=float(y), z=float(z)))
    if len(targets) < count:
        for _ in range(count * 6):
            loc = _sample_nav_location(require_near_route=route_mode)
            if loc is not None:
                targets.append(loc)
            if len(targets) >= count:
                break
    if not targets:
        LOGGER.warning("No walker spawn locations on/near the UGV route.")
        return

    walkers: List["carla.Actor"] = []
    for loc in targets[:count]:
        bp = rng.choice(walker_bps)
        if bp.has_attribute("is_invincible"):
            bp.set_attribute("is_invincible", "true")
        surf_z = _ground_z_near(world, loc.x, loc.y, loc.z, lift=0.0)
        tf = carla.Transform(
            carla.Location(float(loc.x), float(loc.y), surf_z + 1.0),
            carla.Rotation(),
        )
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
            target = _sample_nav_location(require_near_route=route_mode and target_radius > 0)
            if target is None and route_mode and ugv_path:
                alt_pts = _sidewalk_locations_along_route(
                    world, ugv_path, spacing_m=route_spawn_spacing_m * 1.5,
                )
                if alt_pts:
                    x, y, z = rng.choice(alt_pts)
                    target = carla.Location(x=float(x), y=float(y), z=float(z))
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
    vehicle_snap_margin_m = float(scene_cfg.get("static_vehicle_snap_margin_m", 0.02))
    static_route_keepout_m = float(scene_cfg.get("static_route_keepout_radius_m", 0.0))
    npc_vehicle_lift_m = float(scene_cfg.get("npc_vehicle_lift_m", 0.3))
    npc_spawn_max_z_err_m = float(scene_cfg.get("npc_spawn_max_z_err_m", 1.2))

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
            vehicle_snap_margin_m=vehicle_snap_margin_m,
            static_route_keepout_m=static_route_keepout_m,
        )

    _spawn_npc_vehicles(
        world,
        int(scene_cfg.get("moving_npc_vehicles", 0)),
        list(scene_cfg.get("npc_vehicle_filters", ["vehicle.*"])),
        tm, registry, rng, result,
        ugv_xy=ugv_xy,
        keepout_radius_m=float(scene_cfg.get("npc_keepout_radius_m", 15.0)),
        ugv_path=ugv_path,
        route_spawn_radius_m=float(scene_cfg.get("npc_route_spawn_radius_m", 0.0)),
        route_spawn_spacing_m=float(scene_cfg.get("npc_route_spawn_spacing_m", 12.0)),
        route_keepout_radius_m=float(scene_cfg.get("npc_route_keepout_radius_m", 0.0)),
        path_center_keepout_m=float(scene_cfg.get("npc_path_center_keepout_m", 0.0)),
        vehicle_lift_m=npc_vehicle_lift_m,
        max_catalog_z_delta_m=float(scene_cfg.get("npc_spawn_max_z_delta_m", 1.0)),
        vehicle_snap_margin_m=vehicle_snap_margin_m,
        max_spawn_z_err_m=npc_spawn_max_z_err_m,
    )
    _cull_ungrounded_npcs(
        world, result, registry,
        vehicle_lift_m=npc_vehicle_lift_m,
        max_z_err_m=npc_spawn_max_z_err_m,
    )
    _spawn_walkers(
        world,
        int(scene_cfg.get("walkers", 0)),
        registry, rng, result,
        ugv_path=ugv_path,
        route_spawn_radius_m=float(scene_cfg.get("walker_route_spawn_radius_m", 0.0)),
        route_spawn_spacing_m=float(scene_cfg.get("walker_route_spawn_spacing_m", 10.0)),
        route_keepout_radius_m=float(scene_cfg.get("walker_route_keepout_radius_m", 0.0)),
        target_route_spawn_radius_m=float(
            scene_cfg.get(
                "walker_target_route_spawn_radius_m",
                scene_cfg.get("walker_route_spawn_radius_m", 0.0),
            )
        ),
        ugv_xy=ugv_xy,
        spawn_keepout_radius_m=float(scene_cfg.get("walker_spawn_keepout_radius_m", 0.0)),
    )

    static_vehicle_actors = list(result.static_vehicles)
    for actor in result.static_props:
        try:
            if str(actor.type_id).startswith("vehicle."):
                static_vehicle_actors.append(actor)
        except Exception:
            pass
    _reconcile_static_vehicles(
        world, static_vehicle_actors, margin_m=vehicle_snap_margin_m,
    )

    LOGGER.info(
        "Scene dressing summary: static_vehicles=%d, static_props=%d, "
        "moving_npcs=%d, walkers=%d",
        len(result.static_vehicles), len(result.static_props),
        len(result.npc_vehicles), len(result.walkers),
    )
    return result


def finalize_dressing_after_warmup(
    world: "carla.World",
    scene_cfg: Dict[str, Any],
    result: DressingResult,
    registry: ActorRegistry,
) -> int:
    """Re-check moving NPC Z after warmup ticks; cull any floaters."""
    vehicle_lift_m = float(scene_cfg.get("npc_vehicle_lift_m", 0.35))
    max_z_err_m = float(scene_cfg.get("npc_spawn_max_z_err_m", 1.0))
    return _cull_ungrounded_npcs(
        world, result, registry,
        vehicle_lift_m=vehicle_lift_m,
        max_z_err_m=max_z_err_m,
    )


def stop_walker_controllers(result: DressingResult) -> None:
    """Stop all walker AI controllers (call before destroying actors)."""
    for ctrl in result.walker_controllers:
        try:
            ctrl.stop()
        except Exception:  # pragma: no cover
            pass
