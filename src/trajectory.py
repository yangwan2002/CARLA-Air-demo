"""
trajectory.py
-------------
Trajectory controllers for the UGV (CARLA) and UAV (AirSim).

UGV
~~~
* ``autopilot`` - hand the vehicle to the CARLA TrafficManager. We honour a
  few common knobs (ignore_lights_percentage etc.) so the UGV does sensible
  driving without us writing a controller.
* ``waypoint``  - a deliberately simple pure-pursuit-style follower that
  reads a list of ``[x, y, z]`` waypoints from config and drives toward
  them by issuing throttle / steer at every tick.

UAV (delegated to :mod:`airsim_client`)
~~~
* ``follow_ugv`` - we periodically read the UGV transform from CARLA,
  convert to AirSim NED via :func:`geometry.carla_to_airsim_ned`, apply an
  offset in the UGV's body frame, and re-issue ``moveToPositionAsync``.
* ``hover``      - go to ``altitude_m`` once and call ``hover``.
* ``waypoint``   - send the list ``uav.waypoints_ned`` to ``moveOnPathAsync``.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

try:
    import carla  # type: ignore
except ImportError:  # pragma: no cover
    carla = None  # type: ignore

from .airsim_client import AirsimClient
from .geometry import carla_to_airsim_ned


LOGGER = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# UGV
# ---------------------------------------------------------------------------
class UgvController:
    """Drives the UGV using either CARLA's TrafficManager or a waypoint follower."""

    def __init__(
        self,
        vehicle: "carla.Vehicle",
        ugv_cfg: Dict[str, Any],
        traffic_manager: Optional["carla.TrafficManager"] = None,
    ) -> None:
        self.vehicle = vehicle
        self.ugv_cfg = ugv_cfg
        self.tm = traffic_manager
        self.mode = ugv_cfg.get("control_mode", "autopilot")
        self.waypoints: List[List[float]] = list(ugv_cfg.get("waypoints", []) or [])
        self._wp_idx = 0
        self._wp_radius = 3.0  # meters

        self.target_speed_mps = float(ugv_cfg.get("speed_mps", 6.0))
        self.max_throttle = 0.6
        self.max_steer = 0.6

    def start(self) -> None:
        if self.mode == "autopilot" or (self.mode == "waypoint" and not self.waypoints):
            if self.mode == "waypoint":
                LOGGER.warning(
                    "control_mode=waypoint but no waypoints provided; falling back to autopilot.",
                )
                self.mode = "autopilot"
            ap_cfg = self.ugv_cfg.get("autopilot", {}) or {}
            try:
                if self.tm is not None:
                    self.vehicle.set_autopilot(True, self.tm.get_port())
                    self.tm.ignore_lights_percentage(
                        self.vehicle, float(ap_cfg.get("ignore_lights_percentage", 0.0)),
                    )
                    self.tm.ignore_signs_percentage(
                        self.vehicle, float(ap_cfg.get("ignore_signs_percentage", 0.0)),
                    )
                    self.tm.vehicle_percentage_speed_difference(
                        self.vehicle,
                        float(ap_cfg.get("vehicle_percentage_speed_difference", 0.0)),
                    )
                else:
                    self.vehicle.set_autopilot(True)
            except Exception as e:
                LOGGER.warning("Failed to enable autopilot: %s", e)
            LOGGER.info("UGV autopilot enabled.")
        else:
            LOGGER.info("UGV waypoint follower active (%d waypoints).", len(self.waypoints))

    def step(self) -> None:
        """Call once per CARLA tick (only meaningful in waypoint mode)."""
        if self.mode != "waypoint" or not self.waypoints:
            return

        if self._wp_idx >= len(self.waypoints):
            # Final waypoint reached: just brake.
            self._apply_control(throttle=0.0, brake=1.0, steer=0.0)
            return

        tf = self.vehicle.get_transform()
        loc = tf.location
        tx, ty, _tz = self.waypoints[self._wp_idx]
        dx, dy = tx - loc.x, ty - loc.y
        dist = math.hypot(dx, dy)

        if dist < self._wp_radius:
            self._wp_idx += 1
            LOGGER.info("UGV reached waypoint %d/%d", self._wp_idx, len(self.waypoints))
            return

        # Pure-pursuit-ish: steer toward the target heading.
        desired_yaw = math.degrees(math.atan2(dy, dx))
        yaw = tf.rotation.yaw
        # Normalise to [-180, 180]
        err = ((desired_yaw - yaw + 180.0) % 360.0) - 180.0
        steer = max(-self.max_steer, min(self.max_steer, err / 45.0))

        # Simple P controller on speed.
        v = self.vehicle.get_velocity()
        speed_now = math.sqrt(v.x * v.x + v.y * v.y + v.z * v.z)
        speed_err = self.target_speed_mps - speed_now
        throttle = max(0.0, min(self.max_throttle, 0.5 + speed_err * 0.1))
        brake = 0.0 if speed_err > -0.5 else min(1.0, abs(speed_err) * 0.1)

        self._apply_control(throttle=throttle, brake=brake, steer=steer)

    def _apply_control(self, throttle: float, brake: float, steer: float) -> None:
        if carla is None:
            return
        control = carla.VehicleControl(
            throttle=float(throttle),
            steer=float(steer),
            brake=float(brake),
            hand_brake=False,
            reverse=False,
            manual_gear_shift=False,
        )
        try:
            self.vehicle.apply_control(control)
        except Exception as e:  # pragma: no cover
            LOGGER.warning("apply_control failed: %s", e)


# ---------------------------------------------------------------------------
# UAV
# ---------------------------------------------------------------------------
@dataclass
class _FollowState:
    every_n: int = 1
    saves_since_last: int = 0


class UavController:
    """High-level UAV trajectory dispatcher, talks to :class:`AirsimClient`."""

    def __init__(self, airsim_client: AirsimClient, uav_cfg: Dict[str, Any], coords_cfg: Dict[str, Any]) -> None:
        self.client = airsim_client
        self.uav_cfg = uav_cfg
        self.coords_cfg = coords_cfg or {}
        self.mode = uav_cfg.get("mode", "follow_ugv")
        self._follow_state = _FollowState(
            every_n=int(uav_cfg.get("follow_offset", {}).get("update_every_n_saves", 1)),
        )

    def start(self) -> None:
        """Launch + initial maneuver depending on mode."""
        self.client.takeoff()
        self.client.go_to_initial_altitude()

        if self.mode == "hover":
            self.client.hover()
            LOGGER.info("UAV mode=hover: holding altitude.")
        elif self.mode == "waypoint":
            wps = self.uav_cfg.get("waypoints_ned", []) or []
            if not wps:
                LOGGER.warning("UAV mode=waypoint but no waypoints_ned configured; will hover.")
                self.client.hover()
            else:
                LOGGER.info("UAV mode=waypoint: dispatching %d waypoints.", len(wps))
                self.client.move_on_path(wps)
        elif self.mode == "follow_ugv":
            LOGGER.info("UAV mode=follow_ugv: will track UGV position each save tick.")
        else:
            LOGGER.warning("Unknown UAV mode %r; defaulting to hover.", self.mode)
            self.client.hover()

    def on_saved_frame(self, ugv_vehicle: Optional["carla.Vehicle"]) -> None:
        """Hook called once per *saved* frame from the main loop."""
        if self.mode != "follow_ugv" or ugv_vehicle is None:
            return
        self._follow_state.saves_since_last += 1
        if self._follow_state.saves_since_last < self._follow_state.every_n:
            return
        self._follow_state.saves_since_last = 0

        tf = ugv_vehicle.get_transform()
        offset = self.uav_cfg.get("follow_offset", {}) or {}
        behind = float(offset.get("behind_m", 8.0))
        right = float(offset.get("right_m", 0.0))
        alt_above = float(offset.get("altitude_m", 30.0))

        yaw_rad = math.radians(tf.rotation.yaw)
        # In CARLA body frame: +X forward, +Y right.
        body_dx = -behind
        body_dy = right
        world_dx = math.cos(yaw_rad) * body_dx - math.sin(yaw_rad) * body_dy
        world_dy = math.sin(yaw_rad) * body_dx + math.cos(yaw_rad) * body_dy

        target_x = tf.location.x + world_dx
        target_y = tf.location.y + world_dy
        target_z = tf.location.z + alt_above

        c2a = self.coords_cfg.get("carla_to_airsim_ned", {}) or {}
        n, e, d = carla_to_airsim_ned(
            (target_x, target_y, target_z),
            origin_offset_carla=c2a.get("origin_offset_carla", (0.0, 0.0, 0.0)),
            yaw_offset_deg=float(c2a.get("yaw_offset_deg", 0.0)),
        )
        # Yaw the UAV to face the UGV's heading.
        yaw_deg = tf.rotation.yaw + float(c2a.get("yaw_offset_deg", 0.0))
        self.client.move_to_ned(n, e, d, yaw_deg=yaw_deg)
