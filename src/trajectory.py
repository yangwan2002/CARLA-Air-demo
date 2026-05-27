"""
trajectory.py
-------------
Cooperative UGV + UAV trajectory controllers for the SAGENet evaluation
sequence.

The single source of truth is the :class:`RelaySweepCoordinator`. Given the
config's ``trajectory.phases`` block, it returns the desired UGV waypoint
target and UAV pose at any simulation time. Both ``UgvController.step()``
and ``CarlaUavController.step()`` consult the coordinator each tick.

Phases (default config has 4):

  0–15 s   "independent"        UGV drives the inner ring; UAV sits high (~50 m)
                                a few tens of metres off-axis. Low UAV-relay covis.
  15–30 s  "covis_relay_01"     Both converge on relay_01's look_at point;
                                UAV drops to 25 m. Strong three-view covis.
  30–45 s  "scale_var"          UGV crosses to relay_03; UAV trails it but stays
                                at 30 m the whole time (fixed altitude per
                                user request).
  45–60 s  "covis_relay_03"     Final approach to relay_03; UAV at 25 m above.

The coordinator does NOT do real-time covisibility detection. The phase
boundaries are by design and stored on every saved frame so the evaluation
script can slice the dataset by phase.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

try:
    import carla  # type: ignore
except ImportError:  # pragma: no cover
    carla = None  # type: ignore


LOGGER = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Coordinator
# --------------------------------------------------------------------------- #
@dataclass
class PhaseSpec:
    name: str
    t_start: float
    t_end: float
    ugv_path: List[Tuple[float, float]]   # 2D waypoints in CARLA xy
    uav_path: List[Tuple[float, float, float]]  # (x, y, z) waypoints in CARLA
    uav_yaw_mode: str = "track_ugv"       # "track_ugv" | "fixed" | "follow_path"
    uav_yaw_fixed_deg: float = 0.0
    expected_relay: Optional[str] = None  # for the saved phase label
    uav_speed_mps: float = 6.0            # speed along uav_path; hover at endpoint
    uav_loop: bool = False                # wrap uav_path when traveled >= total length
    uav_follow_ugv: bool = False          # xy from live UGV; z from uav_path / altitude


def make_uav_path_from_ugv(
    ugv_path: List[Tuple[float, float]],
    altitude_m: float,
    start_xy: Tuple[float, float],
) -> List[Tuple[float, float, float]]:
    """Build a single-lap UAV polyline matching the UGV ring (constant altitude)."""
    z = float(altitude_m)
    sx, sy = float(start_xy[0]), float(start_xy[1])
    pts: List[Tuple[float, float, float]] = [(sx, sy, z)]
    for x, y in ugv_path:
        pts.append((float(x), float(y), z))
    return pts


class RelaySweepCoordinator:
    """Time-driven UGV+UAV cooperative trajectory.

    The coordinator interprets phase definitions from config. For each phase
    we linearly interpolate between waypoints proportional to the elapsed
    time within the phase. This keeps the math simple, deterministic, and
    easy to debug — and the visual fidelity does not require physics-grade
    smoothing for our evaluation.
    """

    def __init__(self, traj_cfg: Dict[str, Any]) -> None:
        self.cfg = traj_cfg or {}
        self.duration_s = float(self.cfg.get("duration_seconds", 60.0))
        # Default UAV cruising speed; phases can override per-phase by setting
        # uav_speed_mps in their dict. The UAV walks the uav_path at this
        # speed and hovers once it reaches the path endpoint.
        self.uav_default_speed = float(self.cfg.get("uav_default_speed_mps", 6.0))
        # When True, UAV wraps along uav_path instead of hovering at the endpoint.
        self.uav_loop_mode = bool(self.cfg.get("uav_loop_mode", False))
        self._prev_ugv_xy: Optional[Tuple[float, float]] = None
        self._active_uav_phase: Optional[str] = None

        phases_cfg = self.cfg.get("phases", []) or []
        if not phases_cfg:
            raise ValueError("trajectory.phases is empty; cannot run cooperative trajectory.")

        self.phases: List[PhaseSpec] = []
        for p in phases_cfg:
            self.phases.append(PhaseSpec(
                name=str(p["name"]),
                t_start=float(p["t_start"]),
                t_end=float(p["t_end"]),
                ugv_path=[(float(x), float(y)) for (x, y) in p.get("ugv_path", [])],
                uav_path=[(float(x), float(y), float(z)) for (x, y, z) in p.get("uav_path", [])],
                uav_yaw_mode=str(p.get("uav_yaw_mode", "track_ugv")),
                uav_yaw_fixed_deg=float(p.get("uav_yaw_fixed_deg", 0.0)),
                expected_relay=p.get("expected_relay"),
                uav_speed_mps=float(p.get("uav_speed_mps", self.uav_default_speed)),
                uav_loop=bool(p.get("uav_loop", self.uav_loop_mode)),
                uav_follow_ugv=bool(p.get("uav_follow_ugv", False)),
            ))
        # sort by start time so lookups are simple
        self.phases.sort(key=lambda ph: ph.t_start)

    # ------------------------------------------------------------------ #
    def phase_at(self, t: float) -> PhaseSpec:
        """Return the phase active at simulation time ``t``."""
        for ph in self.phases:
            if ph.t_start <= t < ph.t_end:
                return ph
        # past the last phase: clamp to the last
        return self.phases[-1]

    @staticmethod
    def _interp_path(path: List[Tuple[float, ...]], alpha: float) -> Tuple[float, ...]:
        """Interpolate along a polyline. ``alpha`` in [0, 1] over the whole path."""
        if not path:
            raise ValueError("Empty path")
        if len(path) == 1:
            return path[0]
        alpha = max(0.0, min(1.0, alpha))
        # uniform parameterization across segments
        n_seg = len(path) - 1
        seg_pos = alpha * n_seg
        i = int(math.floor(seg_pos))
        if i >= n_seg:
            return path[-1]
        f = seg_pos - i
        a, b = path[i], path[i + 1]
        return tuple(a[k] + (b[k] - a[k]) * f for k in range(len(a)))

    # ------------------------------------------------------------------ #
    def ugv_target(self, t: float) -> Tuple[float, float]:
        ph = self.phase_at(t)
        if not ph.ugv_path:
            return 0.0, 0.0
        alpha = (t - ph.t_start) / max(ph.t_end - ph.t_start, 1e-6)
        return self._interp_path(ph.ugv_path, alpha)  # type: ignore[return-value]

    def uav_pose(
        self,
        t: float,
        ugv_xy: Optional[Tuple[float, float]] = None,
    ) -> Tuple[float, float, float, float]:
        """Return (x, y, z, yaw_deg) for the UAV at time t.

        When ``uav_follow_ugv`` is set, the UAV flies directly above the live
        UGV (same ground track as TrafficManager driving). Otherwise the UAV
        walks ``uav_path`` at ``uav_speed_mps`` (legacy / open-loop mode).
        """
        ph = self.phase_at(t)

        if ph.name != self._active_uav_phase:
            self._active_uav_phase = ph.name
            self._prev_ugv_xy = None

        # --- Companion mode: fly above live UGV (same ground track as TM) ------
        if ph.uav_follow_ugv and ugv_xy is not None:
            z = float(ph.uav_path[0][2]) if ph.uav_path else 35.0
            x, y = float(ugv_xy[0]), float(ugv_xy[1])
            yaw = 0.0
            if self._prev_ugv_xy is not None:
                dx = x - self._prev_ugv_xy[0]
                dy = y - self._prev_ugv_xy[1]
                if dx * dx + dy * dy > 1e-4:
                    yaw = math.degrees(math.atan2(dy, dx))
            self._prev_ugv_xy = (x, y)
            return x, y, z, yaw

        if not ph.uav_path:
            return 0.0, 0.0, 30.0, 0.0

        # Distance the UAV has traveled in this phase, capped at total path length.
        elapsed = max(0.0, t - ph.t_start)
        traveled = elapsed * ph.uav_speed_mps

        # Cumulative segment lengths
        seg_len = []
        for i in range(len(ph.uav_path) - 1):
            a, b = ph.uav_path[i], ph.uav_path[i + 1]
            seg_len.append(((b[0]-a[0])**2 + (b[1]-a[1])**2 + (b[2]-a[2])**2) ** 0.5)
        total_len = sum(seg_len)

        if ph.uav_loop and total_len > 1e-6:
            traveled = traveled % total_len

        if total_len < 1e-6:
            x, y, z = ph.uav_path[0]
        elif traveled >= total_len:
            x, y, z = ph.uav_path[-1]      # hover at endpoint
        else:
            # Find the segment we're currently on.
            d_remain = traveled
            x, y, z = ph.uav_path[0]
            for i, ln in enumerate(seg_len):
                if d_remain <= ln:
                    f = d_remain / ln if ln > 1e-6 else 0.0
                    a, b = ph.uav_path[i], ph.uav_path[i + 1]
                    x = a[0] + (b[0]-a[0]) * f
                    y = a[1] + (b[1]-a[1]) * f
                    z = a[2] + (b[2]-a[2]) * f
                    break
                d_remain -= ln

        # Yaw modes
        if ph.uav_yaw_mode == "fixed":
            yaw = ph.uav_yaw_fixed_deg
        elif ph.uav_yaw_mode == "track_ugv" and ugv_xy is not None:
            dx = ugv_xy[0] - x
            dy = ugv_xy[1] - y
            yaw = math.degrees(math.atan2(dy, dx)) if (dx*dx + dy*dy) > 1e-6 else 0.0
        else:
            # follow_path: yaw points along the local path direction
            n_seg = len(ph.uav_path) - 1
            if n_seg <= 0:
                yaw = 0.0
            else:
                # Find which segment we're on
                d_remain = traveled
                seg_idx = 0
                for i, ln in enumerate(seg_len):
                    if d_remain <= ln:
                        seg_idx = i
                        break
                    d_remain -= ln
                seg_idx = min(seg_idx, n_seg - 1)
                a = ph.uav_path[seg_idx]
                b = ph.uav_path[min(seg_idx + 1, len(ph.uav_path) - 1)]
                dx = b[0] - a[0]
                dy = b[1] - a[1]
                yaw = math.degrees(math.atan2(dy, dx)) if (dx or dy) else 0.0
        return float(x), float(y), float(z), float(yaw)

    def phase_label(self, t: float) -> Tuple[str, Optional[str]]:
        ph = self.phase_at(t)
        return ph.name, ph.expected_relay


# --------------------------------------------------------------------------- #
# UGV controller (waypoint follower driven by the coordinator)
# --------------------------------------------------------------------------- #
class UgvController:
    """Autopilot-driven UGV that follows a per-phase route on the real road graph.

    We hand the vehicle to CARLA's TrafficManager and use ``set_path`` to push
    each phase's target waypoints. The TM finds a legal driving path that
    actually stays on the road, which the previous "raw throttle/steer toward
    target xy" controller could not guarantee on Town10HD's tight street grid.

    Each phase declares ``ugv_path`` (a list of [x, y] target points). When
    the active phase changes, we rebuild the TM path. Within a phase, the TM
    is fully responsible for steering / speed.
    """

    def __init__(
        self,
        vehicle: "carla.Vehicle",
        ugv_cfg: Dict[str, Any],
        coordinator: RelaySweepCoordinator,
        traffic_manager: Optional["carla.TrafficManager"] = None,
    ) -> None:
        self.vehicle = vehicle
        self.cfg = ugv_cfg
        self.coord = coordinator
        self.tm = traffic_manager

        self.target_speed_mps = float(ugv_cfg.get("speed_mps", 5.0))
        self.speed_diff_pct = float(ugv_cfg.get("tm_speed_difference_pct", 30.0))
        self.ignore_vehicles_pct = float(ugv_cfg.get("tm_ignore_vehicles_pct", 0.0))
        self.ignore_walkers_pct = float(ugv_cfg.get("tm_ignore_walkers_pct", 0.0))
        # How close (metres) to the last waypoint before we brake to a stop.
        self._stop_radius = float(ugv_cfg.get("stop_radius_m", 8.0))
        self.loop_mode = bool(ugv_cfg.get("loop_mode", False))
        self._reloop_cooldown_s = float(ugv_cfg.get("reloop_cooldown_s", 8.0))

        self._current_phase_name: Optional[str] = None
        self._stopped = False          # True when we've braked at a phase endpoint
        self._last_reloop_time = -1e9
        # True once the UGV has left the depot / closure point (avoids instant stop).
        self._away_from_start = False
        self._min_depart_m = float(ugv_cfg.get("loop_depart_radius_m", 12.0))

    def start(self) -> None:
        if carla is None:
            return
        if self.tm is None:
            LOGGER.error("UGV controller requires a TrafficManager; aborting.")
            return
        port = self.tm.get_port()
        try:
            self.vehicle.set_autopilot(True, port)
            self.tm.ignore_lights_percentage(self.vehicle, 100.0)
            self.tm.ignore_signs_percentage(self.vehicle, 100.0)
            if self.ignore_vehicles_pct > 0.0 and hasattr(self.tm, "ignore_vehicles_percentage"):
                self.tm.ignore_vehicles_percentage(self.vehicle, self.ignore_vehicles_pct)
            if self.ignore_walkers_pct > 0.0 and hasattr(self.tm, "ignore_walkers_percentage"):
                self.tm.ignore_walkers_percentage(self.vehicle, self.ignore_walkers_pct)
            self.tm.vehicle_percentage_speed_difference(self.vehicle, self.speed_diff_pct)
            self.tm.auto_lane_change(self.vehicle, False)
        except Exception as e:  # pragma: no cover
            LOGGER.warning("UGV TM setup failed: %s", e)
        self.step(0.0)
        LOGGER.info(
            "UGV controller: autopilot ON, TM port=%d, slowdown=%.1f%%, "
            "ignore_vehicles=%.1f%%, ignore_walkers=%.1f%%.",
            port, self.speed_diff_pct, self.ignore_vehicles_pct, self.ignore_walkers_pct,
        )

    def step(self, sim_time: float) -> None:
        """Push the active phase's route; brake once we reach the phase endpoint."""
        if carla is None or self.tm is None:
            return
        ph = self.coord.phase_at(sim_time)

        # --- Phase transition: re-enable autopilot and push new path ----------
        if ph.name != self._current_phase_name:
            self._current_phase_name = ph.name
            self._stopped = False
            self._away_from_start = False
            self.coord._prev_ugv_xy = None
            # Re-enable autopilot in case we braked it during the previous phase.
            try:
                self.vehicle.set_autopilot(True, self.tm.get_port())
            except Exception:
                pass

            path: List["carla.Location"] = [
                carla.Location(x=float(x), y=float(y), z=0.5)
                for (x, y) in ph.ugv_path
            ]
            if path:
                try:
                    self.tm.set_path(self.vehicle, path)
                    LOGGER.info(
                        "UGV phase=%s: pushed %d-waypoint route "
                        "(first=(%.1f,%.1f), last=(%.1f,%.1f)).",
                        ph.name, len(path),
                        path[0].x, path[0].y, path[-1].x, path[-1].y,
                    )
                except Exception as e:  # pragma: no cover
                    LOGGER.warning("TM.set_path failed in phase %s: %s", ph.name, e)
            return

        # --- Already in this phase: check if we're near the endpoint ----------
        if self._stopped or not ph.ugv_path:
            return

        goal_x, goal_y = ph.ugv_path[-1]
        loc = self.vehicle.get_transform().location
        dist_goal = math.hypot(loc.x - goal_x, loc.y - goal_y)
        if dist_goal > self._min_depart_m:
            self._away_from_start = True

        if dist_goal < self._stop_radius:
            # loop_mode: never brake at the closure point; re-issue TM route instead.
            if self.loop_mode:
                if (sim_time - self._last_reloop_time) >= self._reloop_cooldown_s:
                    path = [
                        carla.Location(x=float(x), y=float(y), z=0.5)
                        for (x, y) in ph.ugv_path
                    ]
                    if path:
                        try:
                            self.vehicle.set_autopilot(True, self.tm.get_port())
                            self.tm.set_path(self.vehicle, path)
                            self._last_reloop_time = sim_time
                            self._away_from_start = False
                            LOGGER.info(
                                "UGV phase=%s: loop_mode re-issued %d-waypoint route "
                                "(dist=%.1fm to endpoint).",
                                ph.name, len(path), dist_goal,
                            )
                        except Exception as e:  # pragma: no cover
                            LOGGER.warning(
                                "UGV loop_mode set_path failed in phase %s: %s",
                                ph.name, e,
                            )
                return

            # At spawn before the first lap: do not treat as "finished".
            if not self._away_from_start:
                return

            # Disable autopilot and hold brake so TM cannot drive further.
            try:
                self.vehicle.set_autopilot(False)
                self.vehicle.apply_control(carla.VehicleControl(
                    throttle=0.0, brake=1.0, steer=0.0,
                ))
            except Exception as e:  # pragma: no cover
                LOGGER.warning("UGV brake at endpoint failed: %s", e)
            self._stopped = True
            LOGGER.info(
                "UGV phase=%s: reached endpoint (%.1f,%.1f) dist=%.1fm — braking.",
                ph.name, goal_x, goal_y, dist_goal,
            )


# --------------------------------------------------------------------------- #
# UAV controller (teleport-based, driven by the coordinator)
# --------------------------------------------------------------------------- #
class CarlaUavController:
    """Teleport-driven UAV controller.

    The coordinator gives us an absolute pose at every tick; we just push it
    onto the kinematic body via :func:`CarlaUav.set_pose`.
    """

    def __init__(
        self,
        uav,                                  # CarlaUav (avoid circular import)
        uav_cfg: Dict[str, Any],
        coordinator: RelaySweepCoordinator,
    ) -> None:
        self.uav = uav
        self.cfg = uav_cfg
        self.coord = coordinator

    def start(self) -> None:
        x, y, z, yaw = self.coord.uav_pose(0.0, ugv_xy=None)
        self.uav.set_pose(x, y, z, yaw)
        LOGGER.info("UAV initial pose set to (%.2f, %.2f, %.2f, yaw=%.1f).", x, y, z, yaw)

    def step(self, sim_time: float, ugv_xy: Optional[Tuple[float, float]] = None) -> None:
        x, y, z, yaw = self.coord.uav_pose(sim_time, ugv_xy=ugv_xy)
        self.uav.set_pose(x, y, z, yaw)
