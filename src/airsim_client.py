"""
airsim_client.py
----------------
Thin wrapper around the AirSim multirotor client.

Responsibilities:
  * connect + enable API control + arm + takeoff
  * issue movement commands (hover / waypoints / follow)
  * grab synchronized RGB + depth frames via simGetImages
  * query camera info (intrinsics + pose) for the cameras we capture
  * cleanly disarm and release API control on shutdown

AirSim's coordinate system is NED (north-east-down). All linear quantities
are meters. Quaternion order is (w, x, y, z) for our serialized dicts,
matching the schema in the README.
"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np

try:
    import airsim  # type: ignore
except ImportError:  # pragma: no cover - airsim is required at runtime
    airsim = None  # type: ignore


LOGGER = logging.getLogger(__name__)


# Map our logical name ("front_rgb", ...) to (airsim_camera_key, airsim.ImageType)
def _build_image_request_plan(uav_cfg: Dict[str, Any]) -> List[Tuple[str, str, int]]:
    """Return list of (logical_name, airsim_camera_name, airsim.ImageType_value)."""
    if airsim is None:
        return []
    names = uav_cfg.get("airsim_camera_names", {}) or {}
    front_cam = names.get("front", "front_center")
    down_cam = names.get("down", "bottom_center")

    cams = uav_cfg.get("cameras", {}) or {}
    plan: List[Tuple[str, str, int]] = []
    if cams.get("front_rgb"):
        plan.append(("front_rgb", front_cam, airsim.ImageType.Scene))
    if cams.get("front_depth"):
        plan.append(("front_depth", front_cam, airsim.ImageType.DepthPlanar))
    if cams.get("down_rgb"):
        plan.append(("down_rgb", down_cam, airsim.ImageType.Scene))
    if cams.get("down_depth"):
        plan.append(("down_depth", down_cam, airsim.ImageType.DepthPlanar))
    return plan


@dataclass
class AirsimCameraInfo:
    """Snapshot of an AirSim camera's intrinsics + extrinsics at one frame."""

    name: str                          # logical name (e.g. "front_rgb")
    airsim_camera: str                 # AirSim camera key
    width: int
    height: int
    fov_deg: float
    pose: Optional["airsim.Pose"]      # in NED


class AirsimClient:
    """Manage a single AirSim multirotor and its cameras."""

    def __init__(self, sim_cfg: Dict[str, Any], uav_cfg: Dict[str, Any]) -> None:
        if airsim is None:
            raise ImportError(
                "The `airsim` python package is not importable. "
                "Install via `pip install airsim`."
            )
        self.sim_cfg = sim_cfg
        self.uav_cfg = uav_cfg

        self.host = sim_cfg.get("airsim_host", "localhost")
        self.port = int(sim_cfg.get("airsim_port", 41451))
        self.vehicle_name = uav_cfg.get("vehicle_name", "") or ""

        self.client: Optional["airsim.MultirotorClient"] = None
        self._api_control_enabled = False
        self._armed = False
        self._image_plan = _build_image_request_plan(uav_cfg)

    # ------------------------------------------------------------------
    # connection / lifecycle
    # ------------------------------------------------------------------
    def connect(self) -> None:
        LOGGER.info("Connecting to AirSim at %s:%d", self.host, self.port)
        client = airsim.MultirotorClient(ip=self.host, port=self.port)
        try:
            client.confirmConnection()
        except Exception as e:
            raise ConnectionError(
                f"Could not connect to AirSim at {self.host}:{self.port}. "
                "Is the CARLA-Air AirSim plugin running and listening?"
            ) from e

        self.client = client
        LOGGER.info("AirSim connection confirmed.")

    def enable_and_arm(self) -> None:
        assert self.client is not None
        self.client.enableApiControl(True, self.vehicle_name)
        self._api_control_enabled = True
        self.client.armDisarm(True, self.vehicle_name)
        self._armed = True
        LOGGER.info("AirSim: enableApiControl=True, armed=True")

    def takeoff(self) -> None:
        assert self.client is not None
        timeout = float(self.uav_cfg.get("takeoff_timeout_s", 8.0))
        LOGGER.info("AirSim: taking off (timeout=%.1fs)...", timeout)
        try:
            self.client.takeoffAsync(timeout_sec=timeout, vehicle_name=self.vehicle_name).join()
        except Exception as e:  # pragma: no cover
            LOGGER.warning("takeoffAsync raised %s; continuing anyway.", e)
        LOGGER.info("AirSim: takeoff done.")

    def go_to_initial_altitude(self) -> None:
        """Ascend to the configured cruise altitude before the main loop starts."""
        alt = -abs(float(self.uav_cfg.get("altitude_m", 30.0)))
        speed = float(self.uav_cfg.get("speed_mps", 6.0))
        assert self.client is not None
        LOGGER.info("AirSim: moveToZ to %.2f m (NED, negative = up) at %.2f m/s", alt, speed)
        try:
            self.client.moveToZAsync(alt, speed, vehicle_name=self.vehicle_name).join()
        except Exception as e:  # pragma: no cover
            LOGGER.warning("moveToZAsync raised %s.", e)

    def shutdown(self) -> None:
        """Safely release control of the multirotor. Idempotent."""
        if self.client is None:
            return
        try:
            if self._armed:
                LOGGER.info("AirSim: disarming.")
                self.client.armDisarm(False, self.vehicle_name)
                self._armed = False
        except Exception as e:
            LOGGER.warning("AirSim armDisarm(False) failed: %s", e)

        try:
            if self._api_control_enabled:
                LOGGER.info("AirSim: releasing API control.")
                self.client.enableApiControl(False, self.vehicle_name)
                self._api_control_enabled = False
        except Exception as e:
            LOGGER.warning("AirSim enableApiControl(False) failed: %s", e)

    # ------------------------------------------------------------------
    # motion commands
    # ------------------------------------------------------------------
    def hover(self) -> None:
        assert self.client is not None
        try:
            self.client.hoverAsync(vehicle_name=self.vehicle_name)
        except Exception as e:  # pragma: no cover
            LOGGER.warning("hoverAsync failed: %s", e)

    def move_to_ned(
        self,
        n: float,
        e: float,
        d: float,
        speed_mps: Optional[float] = None,
        yaw_deg: Optional[float] = None,
    ) -> None:
        """Non-blocking moveToPosition."""
        assert self.client is not None
        speed = float(speed_mps if speed_mps is not None else self.uav_cfg.get("speed_mps", 6.0))
        try:
            if yaw_deg is None:
                self.client.moveToPositionAsync(
                    n, e, d, speed, vehicle_name=self.vehicle_name,
                )
            else:
                self.client.moveToPositionAsync(
                    n, e, d, speed,
                    yaw_mode=airsim.YawMode(is_rate=False, yaw_or_rate=float(yaw_deg)),
                    vehicle_name=self.vehicle_name,
                )
        except Exception as e:  # pragma: no cover
            LOGGER.warning("moveToPositionAsync(%.2f,%.2f,%.2f) failed: %s", n, e, d, e)

    def move_on_path(self, waypoints_ned: Iterable[Iterable[float]], speed_mps: Optional[float] = None):
        """Issue a non-blocking moveOnPath. Returns the AirSim future (or None)."""
        assert self.client is not None
        speed = float(speed_mps if speed_mps is not None else self.uav_cfg.get("speed_mps", 6.0))
        path = [airsim.Vector3r(float(p[0]), float(p[1]), float(p[2])) for p in waypoints_ned]
        try:
            return self.client.moveOnPathAsync(
                path, speed,
                vehicle_name=self.vehicle_name,
            )
        except Exception as e:  # pragma: no cover
            LOGGER.warning("moveOnPathAsync failed: %s", e)
            return None

    # ------------------------------------------------------------------
    # state queries
    # ------------------------------------------------------------------
    def get_state(self) -> Optional["airsim.MultirotorState"]:
        assert self.client is not None
        try:
            return self.client.getMultirotorState(vehicle_name=self.vehicle_name)
        except Exception as e:
            LOGGER.warning("getMultirotorState failed: %s", e)
            return None

    def get_camera_infos(self) -> Dict[str, AirsimCameraInfo]:
        """Query simGetCameraInfo for every camera we are capturing."""
        assert self.client is not None
        out: Dict[str, AirsimCameraInfo] = {}
        seen: Dict[str, "airsim.CameraInfo"] = {}
        for logical, cam_name, _itype in self._image_plan:
            if cam_name not in seen:
                try:
                    seen[cam_name] = self.client.simGetCameraInfo(
                        cam_name, vehicle_name=self.vehicle_name,
                    )
                except Exception as e:
                    LOGGER.warning("simGetCameraInfo(%s) failed: %s", cam_name, e)
                    seen[cam_name] = None  # type: ignore[assignment]
            info = seen.get(cam_name)
            width = height = 0
            fov = float("nan")
            pose = None
            if info is not None:
                # AirSim CameraInfo exposes (fov, pose, proj_mat); resolution
                # is set in settings.json and is not returned here, so we fall
                # back to the request plan dimensions.
                try:
                    fov = float(info.fov)
                except Exception:
                    pass
                pose = getattr(info, "pose", None)
            out[logical] = AirsimCameraInfo(
                name=logical, airsim_camera=cam_name,
                width=width, height=height, fov_deg=fov, pose=pose,
            )
        return out

    # ------------------------------------------------------------------
    # image capture
    # ------------------------------------------------------------------
    def capture_images(self) -> Dict[str, "airsim.ImageResponse"]:
        """Snapshot every requested camera in a single simGetImages call."""
        assert self.client is not None
        if not self._image_plan:
            return {}

        requests = []
        for _logical, cam_name, itype in self._image_plan:
            if itype == airsim.ImageType.Scene:
                requests.append(
                    airsim.ImageRequest(cam_name, itype, False, False)  # uncompressed BGRA
                )
            else:
                # Depth - must be requested as float
                requests.append(
                    airsim.ImageRequest(cam_name, itype, True, False)
                )

        try:
            responses = self.client.simGetImages(requests, vehicle_name=self.vehicle_name)
        except Exception as e:
            LOGGER.warning("simGetImages failed: %s", e)
            return {}

        if len(responses) != len(self._image_plan):
            LOGGER.warning(
                "simGetImages returned %d responses, expected %d",
                len(responses), len(self._image_plan),
            )

        out: Dict[str, "airsim.ImageResponse"] = {}
        for (logical, cam_name, _itype), resp in zip(self._image_plan, responses):
            out[logical] = resp
        return out


# ---------------------------------------------------------------------------
# decoding helpers
# ---------------------------------------------------------------------------
def airsim_scene_to_rgb(resp) -> Optional[np.ndarray]:
    """Decode an AirSim Scene ImageResponse to HxWx3 uint8 RGB."""
    if resp is None or resp.width == 0 or resp.height == 0:
        return None
    # uncompressed: image_data_uint8 is BGRA in row-major
    buf = np.frombuffer(resp.image_data_uint8, dtype=np.uint8)
    expected = resp.height * resp.width * 4
    if buf.size != expected:
        # Some AirSim builds emit BGR (3 channels). Detect and handle.
        if buf.size == resp.height * resp.width * 3:
            bgr = buf.reshape((resp.height, resp.width, 3))
            return np.ascontiguousarray(bgr[:, :, ::-1])
        LOGGER.warning(
            "AirSim Scene response size %d != expected %d (HxWx4)",
            buf.size, expected,
        )
        return None
    bgra = buf.reshape((resp.height, resp.width, 4))
    rgb = bgra[:, :, :3][:, :, ::-1]
    return np.ascontiguousarray(rgb)


def airsim_depth_to_meters(resp) -> Optional[np.ndarray]:
    """Decode an AirSim DepthPlanar response (float) to HxW float32 meters."""
    if resp is None or resp.width == 0 or resp.height == 0:
        return None
    if not resp.image_data_float:
        return None
    arr = np.array(resp.image_data_float, dtype=np.float32)
    expected = resp.height * resp.width
    if arr.size != expected:
        LOGGER.warning(
            "AirSim depth response size %d != expected %d", arr.size, expected,
        )
        return None
    return arr.reshape((resp.height, resp.width))


def airsim_state_to_dict(state) -> Dict[str, Any]:
    """Flatten a MultirotorState into a small JSON-friendly dict."""
    if state is None:
        return {}
    kin = getattr(state, "kinematics_estimated", None)
    if kin is None:
        return {}
    p = kin.position
    o = kin.orientation
    lv = kin.linear_velocity
    av = kin.angular_velocity
    return {
        "position_ned": {"x": float(p.x_val), "y": float(p.y_val), "z": float(p.z_val)},
        "orientation_quat_wxyz": {
            "w": float(o.w_val), "x": float(o.x_val), "y": float(o.y_val), "z": float(o.z_val),
        },
        "linear_velocity_ned": {"x": float(lv.x_val), "y": float(lv.y_val), "z": float(lv.z_val)},
        "angular_velocity_body": {"x": float(av.x_val), "y": float(av.y_val), "z": float(av.z_val)},
        "landed_state": int(getattr(state, "landed_state", 0)),
        "timestamp": float(getattr(state, "timestamp", 0.0)),
    }
