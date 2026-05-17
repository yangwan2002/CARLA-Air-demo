#!/usr/bin/env python3
"""
main_collect.py
---------------
End-to-end air / ground / relay synchronous data-collection driver for
CARLA-Air.

Usage:
    python scripts/main_collect.py --config configs/default.yaml

Run from the repo root (``air_ground_relay_collect``).
"""

from __future__ import annotations

import argparse
import logging
import os
import signal
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

# Make sure `src/` is importable when running this script directly.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import numpy as np

from src.config import dump_config, load_config
from src.logging_utils import setup_logging
from src.save_utils import (
    JsonlWriter,
    PoseCsvWriter,
    SequencePaths,
    frame_filename,
    save_depth_npy,
    save_depth_png_viz,
    save_rgb_png,
    write_json,
)
from src.geometry import (
    airsim_pose_to_dict,
    carla_transform_to_dict,
    compute_camera_intrinsic,
)
from src.sync import FrameCounter, make_tick_save_planner

# CARLA / AirSim side (imported lazily inside main so import errors are
# reported through the logger rather than as a bare ImportError).


# --------------------------------------------------------------------------- #
LOGGER = logging.getLogger("collect")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default=str(_REPO_ROOT / "configs" / "default.yaml"),
        help="Path to the YAML config file.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    return parser.parse_args()


# --------------------------------------------------------------------------- #
# Calibration writers
# --------------------------------------------------------------------------- #
def write_ugv_calibration(paths: SequencePaths, ugv_cfg: Dict[str, Any]) -> None:
    cam_cfg = ugv_cfg["camera"]
    w, h = cam_cfg["resolution"]
    intrin = compute_camera_intrinsic(w, h, cam_cfg["fov"])
    payload = {
        "sensor_mode": ugv_cfg["sensor_mode"],
        "intrinsic": intrin,
        "extrinsic_in_vehicle_frame": {
            "location_xyz_m": list(cam_cfg["location"]),
            "rotation_rpy_deg": list(cam_cfg["rotation"]),
        },
        "stereo_baseline_m": float(cam_cfg.get("stereo_baseline_m", 0.0))
            if ugv_cfg["sensor_mode"] == "stereo" else 0.0,
        "coordinate": "carla",
    }
    write_json(paths.calibration_dir / "ugv_intrinsics.json", payload)


def write_uav_calibration(paths: SequencePaths, uav_cfg: Dict[str, Any], airsim_infos: Dict[str, Any]) -> None:
    """Best-effort UAV intrinsics. We cannot get resolution from AirSim's
    simGetCameraInfo, so we fall back to whatever was in settings.json (and
    add a note about that)."""
    payload: Dict[str, Any] = {
        "coordinate": "airsim_ned",
        "note": (
            "AirSim camera resolution comes from settings.json; only fov can be "
            "queried at runtime. Update width/height below to match your "
            "settings.json if you need precise intrinsics."
        ),
        "cameras": {},
    }
    for logical, info in airsim_infos.items():
        # AirSim's `info.fov` is the horizontal FOV in degrees.
        width = int(getattr(info, "width", 0) or 640)
        height = int(getattr(info, "height", 0) or 480)
        fov = float(getattr(info, "fov_deg", 90.0) or 90.0)
        intrin = compute_camera_intrinsic(width, height, fov)
        payload["cameras"][logical] = {
            "airsim_camera": info.airsim_camera,
            "intrinsic": intrin,
            "pose_in_body_frame_ned": (
                airsim_pose_to_dict(info.pose) if info.pose is not None else None
            ),
        }
    write_json(paths.calibration_dir / "uav_intrinsics.json", payload)


def write_relay_calibration(paths: SequencePaths, relay_cfg: Dict[str, Any], relay_cams: List[Any]) -> None:
    intrinsics: Dict[str, Any] = {"coordinate": "carla", "cameras": {}}
    extrinsics: Dict[str, Any] = {"coordinate": "carla", "cameras": {}}
    default_res = relay_cfg.get("default_resolution", [1280, 720])
    default_fov = relay_cfg.get("default_fov", 90.0)
    spec_by_name = {s["name"]: s for s in relay_cfg["cameras"]}

    for cam in relay_cams:
        spec = spec_by_name[cam.name]
        res = spec.get("resolution", default_res)
        fov = spec.get("fov", default_fov)
        intrinsics["cameras"][cam.name] = {
            "intrinsic": compute_camera_intrinsic(res[0], res[1], fov),
            "use_depth": bool(spec.get("use_depth", True)),
        }
        extrinsics["cameras"][cam.name] = carla_transform_to_dict(cam.transform)

    write_json(paths.calibration_dir / "relay_intrinsics.json", intrinsics)
    write_json(paths.calibration_dir / "relay_extrinsics.json", extrinsics)


# --------------------------------------------------------------------------- #
# Per-frame save helpers
# --------------------------------------------------------------------------- #
def _save_carla_rgb(image, dest: Path) -> None:
    from src.sensors import carla_rgb_image_to_array
    rgb = carla_rgb_image_to_array(image)
    save_rgb_png(dest, rgb)


def _save_carla_depth(image, dest_npy: Optional[Path], dest_png: Optional[Path], max_m: float) -> None:
    from src.geometry import decode_carla_depth
    from src.sensors import carla_depth_image_to_array
    bgra = carla_depth_image_to_array(image)
    depth_m = decode_carla_depth(bgra)
    if dest_npy is not None:
        save_depth_npy(dest_npy, depth_m)
    if dest_png is not None:
        save_depth_png_viz(dest_png, depth_m, max_m=max_m)


def _save_airsim_rgb(resp, dest: Path) -> None:
    from src.airsim_client import airsim_scene_to_rgb
    rgb = airsim_scene_to_rgb(resp)
    if rgb is None:
        LOGGER.warning("AirSim RGB response empty; skipping %s", dest.name)
        return
    save_rgb_png(dest, rgb)


def _save_airsim_depth(resp, dest_npy: Optional[Path], dest_png: Optional[Path], max_m: float) -> None:
    from src.airsim_client import airsim_depth_to_meters
    depth = airsim_depth_to_meters(resp)
    if depth is None:
        LOGGER.warning("AirSim depth response empty; skipping %s", dest_npy.name if dest_npy else "(png)")
        return
    if dest_npy is not None:
        save_depth_npy(dest_npy, depth)
    if dest_png is not None:
        save_depth_png_viz(dest_png, depth, max_m=max_m)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main() -> int:
    args = parse_args()
    cfg = load_config(args.config)

    sim_cfg = cfg["simulation"]
    ugv_cfg = cfg["ugv"]
    uav_cfg = cfg["uav"]
    relay_cfg = cfg["relay_cameras"]
    save_cfg = cfg["save"]

    # ---- paths -----------------------------------------------------------
    paths = SequencePaths.build(
        output_dir=sim_cfg["output_dir"],
        sequence_name=sim_cfg.get("sequence_name", "seq_001"),
        ugv_sensor_mode=ugv_cfg["sensor_mode"],
        ugv_capture_depth_stereo=bool(ugv_cfg["camera"].get("stereo_capture_depth", False)),
        uav_camera_flags=uav_cfg.get("cameras", {}) or {},
        relay_camera_specs=relay_cfg["cameras"],
        jsonl_filename=save_cfg.get("jsonl_filename", "frames.jsonl"),
    )
    paths.ensure()

    log_file = paths.root / "collect.log"
    setup_logging(getattr(logging, args.log_level.upper()), log_file=log_file)
    LOGGER.info("Loaded config from %s", args.config)
    LOGGER.info("Sequence root: %s", paths.root)

    # Dump a snapshot of the config so the dataset is self-describing.
    dump_config(cfg, paths.dataset_root / "dataset_config.yaml")
    dump_config(cfg, paths.root / "config_snapshot.yaml")

    # Seed numpy for reproducibility.
    seed = sim_cfg.get("seed")
    if seed is not None:
        np.random.seed(int(seed))

    # ---- lazy imports of CARLA / AirSim ---------------------------------
    try:
        from src.carla_client import CarlaWorld
        from src.actors import ActorRegistry, spawn_ugv
        from src.sensors import (
            build_ugv_camera_rig,
            build_relay_cameras,
            pull_image_for_frame,
        )
        from src.trajectory import UavController, UgvController
        from src.airsim_client import (
            AirsimClient,
            airsim_state_to_dict,
        )
    except ImportError as e:
        LOGGER.error("Required python packages missing: %s", e)
        return 2

    # Cooperative shutdown: catch Ctrl+C even on Windows.
    stop_requested = {"flag": False}

    def _on_sigint(signum, frame):  # noqa: ARG001
        LOGGER.warning("Signal %d received; will stop after current iteration.", signum)
        stop_requested["flag"] = True

    signal.signal(signal.SIGINT, _on_sigint)
    if hasattr(signal, "SIGTERM"):
        try:
            signal.signal(signal.SIGTERM, _on_sigint)
        except Exception:
            pass

    # ---------------------------------------------------------------------
    carla_world: Optional[CarlaWorld] = None
    registry: Optional[ActorRegistry] = None
    airsim_client: Optional[AirsimClient] = None

    jsonl: Optional[JsonlWriter] = None
    ugv_pose_csv: Optional[PoseCsvWriter] = None
    uav_pose_csv: Optional[PoseCsvWriter] = None

    counter = FrameCounter()
    exit_code = 0
    start_wall = time.time()

    try:
        # ----- CARLA -----------------------------------------------------
        carla_world = CarlaWorld(sim_cfg)
        carla_world.connect()
        carla_world.apply_map_and_weather()
        carla_world.enable_synchronous_mode()

        world = carla_world.world
        assert world is not None

        registry = ActorRegistry(world)

        LOGGER.info("Spawning UGV...")
        ugv_vehicle = spawn_ugv(world, ugv_cfg, registry, seed=seed)

        LOGGER.info("Building UGV camera rig (mode=%s)...", ugv_cfg["sensor_mode"])
        ugv_rig = build_ugv_camera_rig(world, ugv_vehicle, ugv_cfg, registry)
        for s in ugv_rig.sensors.values():
            s.start()

        LOGGER.info("Building %d relay cameras...", len(relay_cfg["cameras"]))
        relay_cams = build_relay_cameras(world, relay_cfg, registry)
        for cam in relay_cams:
            cam.rgb.start()
            if cam.depth is not None:
                cam.depth.start()

        # Per-relay extrinsic files (useful when consumers grab a single cam).
        for cam in relay_cams:
            cam_dir = paths.relay_dir / cam.name
            cam_dir.mkdir(parents=True, exist_ok=True)
            write_json(cam_dir / "extrinsic.json", {
                "coordinate": "carla",
                "transform": carla_transform_to_dict(cam.transform),
            })

        # ----- UGV controller -------------------------------------------
        ugv_ctrl = UgvController(ugv_vehicle, ugv_cfg, traffic_manager=carla_world.tm)
        ugv_ctrl.start()

        # ----- AirSim ----------------------------------------------------
        airsim_client = AirsimClient(sim_cfg, uav_cfg)
        airsim_client.connect()
        airsim_client.enable_and_arm()

        uav_ctrl = UavController(airsim_client, uav_cfg, cfg.get("coords", {}))
        uav_ctrl.start()

        # ----- Calibration ----------------------------------------------
        if save_cfg.get("save_calibration", True):
            LOGGER.info("Writing calibration files...")
            write_ugv_calibration(paths, ugv_cfg)
            write_relay_calibration(paths, relay_cfg, relay_cams)
            airsim_infos = airsim_client.get_camera_infos()
            write_uav_calibration(paths, uav_cfg, airsim_infos)

        # ----- Writers ---------------------------------------------------
        if save_cfg.get("save_frames_jsonl", True):
            jsonl = JsonlWriter(paths.frames_jsonl)
        if save_cfg.get("save_pose_csv", True):
            ugv_pose_csv = PoseCsvWriter(paths.ugv_dir / "pose.csv", mode="euler")
            uav_pose_csv = PoseCsvWriter(paths.uav_dir / "pose.csv", mode="quat")

        # ----- Tick loop --------------------------------------------------
        planner = make_tick_save_planner(
            simulation_hz=float(sim_cfg.get("simulation_hz", 20.0)),
            save_hz=float(sim_cfg.get("save_hz", 10.0)),
        )
        duration_s = float(sim_cfg.get("duration_seconds", 60.0))
        delta = float(sim_cfg.get("fixed_delta_seconds", 0.05))
        sensor_timeout = max(2.0, 5.0 * delta)

        sim_time = 0.0
        save_frame_id = 0

        LOGGER.info(
            "Entering tick loop: duration=%.2fs, save_hz=%.2f, simulation_hz=%.2f, sensor_timeout=%.2fs",
            duration_s, planner.save_hz, planner.simulation_hz, sensor_timeout,
        )

        while not stop_requested["flag"] and sim_time < duration_s:
            world_frame = world.tick()
            counter.tick_count += 1
            sim_time = counter.tick_count * delta

            ugv_ctrl.step()  # no-op for autopilot

            counter.log_progress(every_n_ticks=int(planner.simulation_hz), duration_s=duration_s, current_t=sim_time)

            if not planner.should_save():
                continue

            # ---- Pull every CARLA sensor for THIS tick frame -----------
            ugv_images: Dict[str, Any] = {}
            for logical, sensor in ugv_rig.sensors.items():
                img = pull_image_for_frame(sensor, world_frame, timeout_s=sensor_timeout)
                if img is None:
                    counter.dropped_sensor_frames += 1
                ugv_images[logical] = img

            relay_payload: Dict[str, Dict[str, Any]] = {}
            for cam in relay_cams:
                entry: Dict[str, Any] = {}
                rgb_img = pull_image_for_frame(cam.rgb, world_frame, timeout_s=sensor_timeout)
                if rgb_img is None:
                    counter.dropped_sensor_frames += 1
                entry["rgb"] = rgb_img
                if cam.depth is not None:
                    depth_img = pull_image_for_frame(cam.depth, world_frame, timeout_s=sensor_timeout)
                    if depth_img is None:
                        counter.dropped_sensor_frames += 1
                    entry["depth"] = depth_img
                relay_payload[cam.name] = entry

            # ---- AirSim -------------------------------------------------
            airsim_responses = airsim_client.capture_images()
            airsim_state = airsim_client.get_state()
            airsim_infos_now = airsim_client.get_camera_infos()

            # ---- Persist images ----------------------------------------
            file_ext = save_cfg.get("image_format", "png")
            png_name = frame_filename(save_frame_id, file_ext)
            npy_name = frame_filename(save_frame_id, "npy")
            depth_png_name = frame_filename(save_frame_id, "png")
            depth_max_m = float(save_cfg.get("depth_png_max_m", 80.0))
            save_depth_npy_flag = bool(save_cfg.get("save_depth_npy", True))
            save_depth_png_flag = bool(save_cfg.get("save_depth_png", True))

            images_index: Dict[str, str] = {}

            # UGV RGB / stereo
            for logical, img in ugv_images.items():
                if img is None:
                    continue
                if logical in ("front_rgb", "stereo_left", "stereo_right"):
                    dest = paths.ugv_subdirs[logical] / png_name
                    _save_carla_rgb(img, dest)
                    images_index[f"ugv_{logical}"] = paths.rel(dest)
                elif logical == "front_depth":
                    dest_npy = paths.ugv_subdirs[logical] / npy_name if save_depth_npy_flag else None
                    dest_png = paths.ugv_subdirs[logical] / depth_png_name if save_depth_png_flag else None
                    _save_carla_depth(img, dest_npy, dest_png, depth_max_m)
                    if dest_npy is not None:
                        images_index["ugv_front_depth"] = paths.rel(dest_npy)
                    elif dest_png is not None:
                        images_index["ugv_front_depth"] = paths.rel(dest_png)

            # UAV
            uav_cam_flags = uav_cfg.get("cameras", {}) or {}
            if uav_cam_flags.get("front_rgb") and "front_rgb" in airsim_responses:
                dest = paths.uav_subdirs["front_rgb"] / png_name
                _save_airsim_rgb(airsim_responses["front_rgb"], dest)
                images_index["uav_front_rgb"] = paths.rel(dest)
            if uav_cam_flags.get("front_depth") and "front_depth" in airsim_responses:
                dest_npy = paths.uav_subdirs["front_depth"] / npy_name if save_depth_npy_flag else None
                dest_png = paths.uav_subdirs["front_depth"] / depth_png_name if save_depth_png_flag else None
                _save_airsim_depth(airsim_responses["front_depth"], dest_npy, dest_png, depth_max_m)
                if dest_npy is not None:
                    images_index["uav_front_depth"] = paths.rel(dest_npy)
            if uav_cam_flags.get("down_rgb") and "down_rgb" in airsim_responses:
                dest = paths.uav_subdirs["down_rgb"] / png_name
                _save_airsim_rgb(airsim_responses["down_rgb"], dest)
                images_index["uav_down_rgb"] = paths.rel(dest)
            if uav_cam_flags.get("down_depth") and "down_depth" in airsim_responses:
                dest_npy = paths.uav_subdirs["down_depth"] / npy_name if save_depth_npy_flag else None
                dest_png = paths.uav_subdirs["down_depth"] / depth_png_name if save_depth_png_flag else None
                _save_airsim_depth(airsim_responses["down_depth"], dest_npy, dest_png, depth_max_m)
                if dest_npy is not None:
                    images_index["uav_down_depth"] = paths.rel(dest_npy)

            # Relay
            for cam in relay_cams:
                entry = relay_payload[cam.name]
                rgb_img = entry.get("rgb")
                if rgb_img is not None:
                    dest = paths.relay_subdirs[cam.name]["rgb"] / png_name
                    _save_carla_rgb(rgb_img, dest)
                    images_index[f"{cam.name}_rgb"] = paths.rel(dest)
                depth_img = entry.get("depth")
                if depth_img is not None:
                    dest_npy = paths.relay_subdirs[cam.name]["depth"] / npy_name if save_depth_npy_flag else None
                    dest_png = paths.relay_subdirs[cam.name]["depth"] / depth_png_name if save_depth_png_flag else None
                    _save_carla_depth(depth_img, dest_npy, dest_png, depth_max_m)
                    if dest_npy is not None:
                        images_index[f"{cam.name}_depth"] = paths.rel(dest_npy)

            # ---- Poses --------------------------------------------------
            ugv_tf = ugv_vehicle.get_transform()
            ugv_pose_dict = carla_transform_to_dict(ugv_tf)

            # UGV camera world pose: use the first attached sensor.
            ugv_cam_pose_dict: Dict[str, float] = {}
            for s in ugv_rig.sensors.values():
                try:
                    ugv_cam_pose_dict = carla_transform_to_dict(s.actor.get_transform())
                    break
                except Exception:
                    continue

            uav_state_dict = airsim_state_to_dict(airsim_state)
            uav_pose_dict: Dict[str, float] = {}
            if airsim_state is not None:
                try:
                    uav_pose_dict = airsim_pose_to_dict(airsim_state.kinematics_estimated)  # type: ignore[arg-type]
                except Exception:
                    # Fall back to flattening manually.
                    if "position_ned" in uav_state_dict:
                        pos = uav_state_dict["position_ned"]
                        ori = uav_state_dict.get("orientation_quat_wxyz", {})
                        uav_pose_dict = {
                            "x": pos["x"], "y": pos["y"], "z": pos["z"],
                            "qw": ori.get("w", 1.0), "qx": ori.get("x", 0.0),
                            "qy": ori.get("y", 0.0), "qz": ori.get("z", 0.0),
                        }

            uav_cam_poses: Dict[str, Dict[str, float]] = {}
            for logical, info in airsim_infos_now.items():
                if info.pose is not None:
                    try:
                        uav_cam_poses[logical] = airsim_pose_to_dict(info.pose)
                    except Exception:
                        pass

            relay_cam_poses: Dict[str, Dict[str, float]] = {}
            for cam in relay_cams:
                try:
                    relay_cam_poses[cam.name] = carla_transform_to_dict(cam.rgb.actor.get_transform())
                except Exception:
                    relay_cam_poses[cam.name] = carla_transform_to_dict(cam.transform)

            # ---- Writers ------------------------------------------------
            frame_record = {
                "frame_id": save_frame_id,
                "carla_frame": int(world_frame),
                "sim_time": float(sim_time),
                "sequence": sim_cfg.get("sequence_name", "seq_001"),
                "map": sim_cfg.get("map", ""),
                "images": images_index,
                "ugv_pose_carla": ugv_pose_dict,
                "ugv_camera_pose_carla": ugv_cam_pose_dict,
                "uav_pose_airsim_ned": uav_pose_dict,
                "uav_state_airsim_ned": uav_state_dict,
                "uav_camera_pose_airsim_ned": uav_cam_poses,
                "relay_camera_poses_carla": relay_cam_poses,
            }

            if jsonl is not None:
                jsonl.write(frame_record)
            if ugv_pose_csv is not None:
                ugv_pose_csv.write(save_frame_id, sim_time, ugv_pose_dict)
            if uav_pose_csv is not None and uav_pose_dict:
                uav_pose_csv.write(save_frame_id, sim_time, uav_pose_dict)

            # ---- Per-save callbacks ------------------------------------
            uav_ctrl.on_saved_frame(ugv_vehicle)

            counter.saved += 1
            save_frame_id += 1

        # ----- trajectory.yaml ------------------------------------------
        if save_cfg.get("save_trajectory_yaml", True):
            import yaml
            traj = {
                "sequence": sim_cfg.get("sequence_name", "seq_001"),
                "map": sim_cfg.get("map", ""),
                "duration_s": float(sim_time),
                "frame_count": counter.saved,
                "tick_count": counter.tick_count,
                "fixed_delta_seconds": delta,
                "simulation_hz": float(sim_cfg.get("simulation_hz", 20.0)),
                "save_hz": float(sim_cfg.get("save_hz", 10.0)),
                "ugv_mode": ugv_cfg.get("control_mode", "autopilot"),
                "uav_mode": uav_cfg.get("mode", "follow_ugv"),
            }
            with paths.trajectory_yaml.open("w", encoding="utf-8") as fp:
                yaml.safe_dump(traj, fp, sort_keys=False)

    except KeyboardInterrupt:
        LOGGER.warning("KeyboardInterrupt - stopping.")
    except Exception as e:
        LOGGER.exception("Fatal error during data collection: %s", e)
        exit_code = 1
    finally:
        LOGGER.info("Cleaning up...")

        # Close writers first so partial CSV/JSONL still flush.
        for w in (jsonl, ugv_pose_csv, uav_pose_csv):
            if w is not None:
                w.close()

        # Stop CARLA sensors before destroying their actors.
        if registry is not None:
            registry.destroy_all()

        if carla_world is not None:
            carla_world.shutdown()

        if airsim_client is not None:
            airsim_client.shutdown()

        wall_dt = time.time() - start_wall
        LOGGER.info(
            "Done. wall_time=%.2fs, ticks=%d, saved_frames=%d, dropped_sensor_frames=%d, output=%s",
            wall_dt, counter.tick_count, counter.saved, counter.dropped_sensor_frames, paths.root,
        )

    return exit_code


if __name__ == "__main__":
    sys.exit(main())
