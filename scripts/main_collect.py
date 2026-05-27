#!/usr/bin/env python3
"""
main_collect.py
---------------
End-to-end CARLA-only synchronous data collection driver for the
SAGENet evaluation sequence.

Replaces the older CARLA + AirSim driver. The UAV is now a CARLA-native
kinematic actor (see :mod:`src.carla_uav`) and trajectory is driven by
:class:`src.trajectory.RelaySweepCoordinator` in cooperative mode.

Usage::
    python scripts/main_collect.py --config configs/default.yaml
"""

from __future__ import annotations

import argparse
import logging
import signal
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

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
    carla_transform_to_dict,
    compute_camera_intrinsic,
)
from src.sync import FrameCounter, make_tick_save_planner


LOGGER = logging.getLogger("collect")


# --------------------------------------------------------------------------- #
# Args
# --------------------------------------------------------------------------- #
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
    payload = {
        "sensor_mode": ugv_cfg["sensor_mode"],
        "intrinsic": compute_camera_intrinsic(w, h, cam_cfg["fov"]),
        "extrinsic_in_vehicle_frame": {
            "location_xyz_m": list(cam_cfg["location"]),
            "rotation_rpy_deg": list(cam_cfg["rotation"]),
        },
        "stereo_baseline_m": float(cam_cfg.get("stereo_baseline_m", 0.0))
            if ugv_cfg["sensor_mode"] == "stereo" else 0.0,
        "coordinate": "carla",
    }
    write_json(paths.calibration_dir / "ugv_intrinsics.json", payload)


def write_uav_calibration(paths: SequencePaths, uav_cfg: Dict[str, Any]) -> None:
    cam_cfg = uav_cfg.get("camera", {}) or {}
    res = cam_cfg.get("resolution", [1280, 720])
    fov = float(cam_cfg.get("fov", 90.0))
    payload = {
        "coordinate": "carla",
        "intrinsic": compute_camera_intrinsic(int(res[0]), int(res[1]), fov),
        "extrinsic_in_body_frame": {
            "location_xyz_m": list(cam_cfg.get("location", [0.0, 0.0, -0.3])),
            "rotation_rpy_deg": [0.0, float(uav_cfg.get("camera_pitch_deg", -90.0)), 0.0],
        },
        "body_blueprint": uav_cfg.get("body_blueprint", "vehicle.diamondback.century"),
    }
    write_json(paths.calibration_dir / "uav_intrinsics.json", payload)


def write_relay_calibration(paths: SequencePaths, relay_cfg: Dict[str, Any], relay_cams: List[Any]) -> None:
    intrinsics: Dict[str, Any] = {"coordinate": "carla", "cameras": {}}
    extrinsics: Dict[str, Any] = {"coordinate": "carla", "cameras": {}}
    default_res = relay_cfg.get("default_resolution", [1280, 720])
    default_fov = relay_cfg.get("default_fov", 75.0)
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
# Image-saving helpers
# --------------------------------------------------------------------------- #
def _save_carla_rgb(image, dest: Path) -> None:
    from src.sensors import carla_rgb_image_to_array
    save_rgb_png(dest, carla_rgb_image_to_array(image))


def _save_carla_depth(image, dest_npy: Optional[Path], dest_png: Optional[Path], max_m: float) -> None:
    from src.geometry import decode_carla_depth
    from src.sensors import carla_depth_image_to_array
    bgra = carla_depth_image_to_array(image)
    depth_m = decode_carla_depth(bgra)
    if dest_npy is not None:
        save_depth_npy(dest_npy, depth_m)
    if dest_png is not None:
        save_depth_png_viz(dest_png, depth_m, max_m=max_m)


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
    scene_cfg = cfg.get("scene", {}).get("dressing", {}) or {}
    traj_cfg = cfg.get("trajectory", {}) or {}

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
    overwrite_existing = bool(sim_cfg.get("overwrite_existing_sequence", False))
    if paths.root.exists():
        existing_entries = list(paths.root.iterdir())
        if existing_entries:
            if overwrite_existing:
                shutil.rmtree(paths.root)
            else:
                print(
                    "Refusing to reuse existing non-empty sequence directory: "
                    f"{paths.root}. Change simulation.sequence_name or set "
                    "simulation.overwrite_existing_sequence=true to replace it.",
                    file=sys.stderr,
                )
                return 3
    paths.ensure()

    log_file = paths.root / "collect.log"
    setup_logging(getattr(logging, args.log_level.upper()), log_file=log_file)
    LOGGER.info("Loaded config from %s", args.config)
    LOGGER.info("Sequence root: %s", paths.root)

    dump_config(cfg, paths.dataset_root / "dataset_config.yaml")
    dump_config(cfg, paths.root / "config_snapshot.yaml")

    seed = sim_cfg.get("seed")
    if seed is not None:
        np.random.seed(int(seed))

    # ---- lazy imports ----------------------------------------------------
    try:
        from src.carla_client import CarlaWorld
        from src.actors import ActorRegistry, spawn_ugv
        from src.sensors import (
            build_ugv_camera_rig,
            build_relay_cameras,
            pull_image_for_frame,
            pull_latest_image,
        )
        from src.carla_uav import CarlaUav
        from src.scene_dressing import dress_scene, stop_walker_controllers
        from src.trajectory import (
            RelaySweepCoordinator,
            UgvController,
            CarlaUavController,
        )
    except ImportError as e:
        LOGGER.error("Required python packages missing: %s", e)
        return 2

    # ---- signal handling -------------------------------------------------
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

    # ---- state -----------------------------------------------------------
    carla_world: Optional[CarlaWorld] = None
    registry: Optional[ActorRegistry] = None
    dressing = None

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
        sync_mode = bool(sim_cfg.get("synchronous_mode", True))

        registry = ActorRegistry(world)

        # ----- coordinator (single source of truth for trajectories) -----
        # Pass duration_seconds down so a config can omit it on the trajectory block.
        traj_cfg.setdefault("duration_seconds", sim_cfg.get("duration_seconds", 60.0))
        coordinator = RelaySweepCoordinator(traj_cfg)
        dressing_path: List[tuple[float, float]] = []
        for phase in traj_cfg.get("phases", []) or []:
            for x, y in phase.get("ugv_path", []) or []:
                xy = (float(x), float(y))
                if not dressing_path or dressing_path[-1] != xy:
                    dressing_path.append(xy)

        # ----- spawn UGV ------------------------------------------------
        LOGGER.info("Spawning UGV...")
        ugv_vehicle = spawn_ugv(world, ugv_cfg, registry, seed=seed)
        # In sync mode, freshly-spawned actors report transform=(0,0,0) until
        # the next world tick. Tick once so ugv_init_xy below is correct,
        # otherwise NPC keepout and prop placement use a bogus origin.
        world.tick()

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
        for cam in relay_cams:
            cam_dir = paths.relay_dir / cam.name
            cam_dir.mkdir(parents=True, exist_ok=True)
            write_json(cam_dir / "extrinsic.json", {
                "coordinate": "carla",
                "transform": carla_transform_to_dict(cam.transform),
            })

        # ----- spawn UAV ------------------------------------------------
        LOGGER.info("Spawning CARLA-native UAV...")
        ugv_init_xy = (ugv_vehicle.get_transform().location.x,
                       ugv_vehicle.get_transform().location.y)
        x0, y0, z0, yaw0 = coordinator.uav_pose(0.0, ugv_xy=ugv_init_xy)
        uav = CarlaUav.spawn(world, uav_cfg, registry,
                             spawn_xy=[x0, y0], spawn_altitude_m=z0)

        # ----- scene dressing (NPCs, static cars, props) ---------------
        if scene_cfg:
            LOGGER.info("Dressing the scene with NPCs and static props...")
            dressing = dress_scene(
                world, scene_cfg, relay_cfg["cameras"],
                tm=carla_world.tm, registry=registry, seed=seed,
                ugv_xy=ugv_init_xy,
                ugv_path=dressing_path,
            )

        # ----- controllers ---------------------------------------------
        ugv_ctrl = UgvController(ugv_vehicle, ugv_cfg, coordinator,
                                 traffic_manager=carla_world.tm)
        ugv_ctrl.start()
        uav_ctrl = CarlaUavController(uav, uav_cfg, coordinator)
        uav_ctrl.start()

        # ----- calibration ---------------------------------------------
        if save_cfg.get("save_calibration", True):
            LOGGER.info("Writing calibration files...")
            write_ugv_calibration(paths, ugv_cfg)
            write_uav_calibration(paths, uav_cfg)
            write_relay_calibration(paths, relay_cfg, relay_cams)

        # ----- writers --------------------------------------------------
        if save_cfg.get("save_frames_jsonl", True):
            jsonl = JsonlWriter(paths.frames_jsonl)
        if save_cfg.get("save_pose_csv", True):
            ugv_pose_csv = PoseCsvWriter(paths.ugv_dir / "pose.csv", mode="euler")
            uav_pose_csv = PoseCsvWriter(paths.uav_dir / "pose.csv", mode="euler")

        # ----- tick loop ----------------------------------------------
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
            "Entering tick loop: duration=%.2fs, save_hz=%.2f, simulation_hz=%.2f, "
            "phases=%d, sync=%s",
            duration_s, planner.save_hz, planner.simulation_hz,
            len(coordinator.phases), sync_mode,
        )

        while not stop_requested["flag"] and sim_time < duration_s:
            # -- step controllers BEFORE world.tick() so the new poses
            # -- take effect this frame.
            ugv_ctrl.step(sim_time)
            ugv_loc = ugv_vehicle.get_transform().location
            uav_ctrl.step(sim_time, ugv_xy=(ugv_loc.x, ugv_loc.y))

            if sync_mode:
                world_frame = world.tick()
            else:
                snap = world.wait_for_tick(seconds=max(2.0, 10.0 * delta))
                world_frame = int(snap.frame)
            counter.tick_count += 1
            sim_time = counter.tick_count * delta

            counter.log_progress(
                every_n_ticks=int(planner.simulation_hz),
                duration_s=duration_s, current_t=sim_time,
            )

            if not planner.should_save():
                continue

            # ---- pull every sensor for THIS world frame ----------
            def _pull(sensor):
                if sync_mode:
                    return pull_image_for_frame(sensor, world_frame, timeout_s=sensor_timeout)
                return pull_latest_image(sensor, timeout_s=sensor_timeout)

            ugv_images: Dict[str, Any] = {}
            for logical, sensor in ugv_rig.sensors.items():
                img = _pull(sensor)
                if img is None:
                    counter.dropped_sensor_frames += 1
                ugv_images[logical] = img

            uav_images: Dict[str, Any] = {}
            for logical, sensor in uav.cameras.items():
                img = _pull(sensor)
                if img is None:
                    counter.dropped_sensor_frames += 1
                uav_images[logical] = img

            relay_payload: Dict[str, Dict[str, Any]] = {}
            for cam in relay_cams:
                entry: Dict[str, Any] = {"rgb": _pull(cam.rgb)}
                if entry["rgb"] is None:
                    counter.dropped_sensor_frames += 1
                if cam.depth is not None:
                    entry["depth"] = _pull(cam.depth)
                    if entry["depth"] is None:
                        counter.dropped_sensor_frames += 1
                relay_payload[cam.name] = entry

            # ---- persist images ---------------------------------------
            file_ext = save_cfg.get("image_format", "png")
            png_name = frame_filename(save_frame_id, file_ext)
            npy_name = frame_filename(save_frame_id, "npy")
            depth_max_m = float(save_cfg.get("depth_png_max_m", 80.0))
            save_depth_npy_flag = bool(save_cfg.get("save_depth_npy", True))
            save_depth_png_flag = bool(save_cfg.get("save_depth_png", True))

            images_index: Dict[str, str] = {}

            # UGV
            for logical, img in ugv_images.items():
                if img is None:
                    continue
                if logical in ("front_rgb", "stereo_left", "stereo_right"):
                    dest = paths.ugv_subdirs[logical] / png_name
                    _save_carla_rgb(img, dest)
                    images_index[f"ugv_{logical}"] = paths.rel(dest)
                elif logical == "front_depth":
                    dest_npy = paths.ugv_subdirs[logical] / npy_name if save_depth_npy_flag else None
                    dest_png = paths.ugv_subdirs[logical] / png_name if save_depth_png_flag else None
                    _save_carla_depth(img, dest_npy, dest_png, depth_max_m)
                    if dest_npy is not None:
                        images_index["ugv_front_depth"] = paths.rel(dest_npy)
                    elif dest_png is not None:
                        images_index["ugv_front_depth"] = paths.rel(dest_png)

            # UAV (CARLA-native, same code path as relay/ugv)
            for logical, img in uav_images.items():
                if img is None:
                    continue
                if logical == "front_rgb":
                    dest = paths.uav_subdirs["front_rgb"] / png_name
                    _save_carla_rgb(img, dest)
                    images_index["uav_front_rgb"] = paths.rel(dest)
                elif logical == "front_depth":
                    dest_npy = paths.uav_subdirs["front_depth"] / npy_name if save_depth_npy_flag else None
                    dest_png = paths.uav_subdirs["front_depth"] / png_name if save_depth_png_flag else None
                    _save_carla_depth(img, dest_npy, dest_png, depth_max_m)
                    if dest_npy is not None:
                        images_index["uav_front_depth"] = paths.rel(dest_npy)
                    elif dest_png is not None:
                        images_index["uav_front_depth"] = paths.rel(dest_png)

            # Relay
            for cam in relay_cams:
                entry = relay_payload[cam.name]
                if entry.get("rgb") is not None:
                    dest = paths.relay_subdirs[cam.name]["rgb"] / png_name
                    _save_carla_rgb(entry["rgb"], dest)
                    images_index[f"{cam.name}_rgb"] = paths.rel(dest)
                if entry.get("depth") is not None:
                    dest_npy = paths.relay_subdirs[cam.name]["depth"] / npy_name if save_depth_npy_flag else None
                    dest_png = paths.relay_subdirs[cam.name]["depth"] / png_name if save_depth_png_flag else None
                    _save_carla_depth(entry["depth"], dest_npy, dest_png, depth_max_m)
                    if dest_npy is not None:
                        images_index[f"{cam.name}_depth"] = paths.rel(dest_npy)

            # ---- poses + phase label -------------------------------
            ugv_pose_dict = carla_transform_to_dict(ugv_vehicle.get_transform())
            ugv_cam_pose_dict: Dict[str, float] = {}
            for s in ugv_rig.sensors.values():
                try:
                    ugv_cam_pose_dict = carla_transform_to_dict(s.actor.get_transform())
                    break
                except Exception:
                    continue

            uav_pose_dict = carla_transform_to_dict(uav.get_transform())
            uav_cam_poses: Dict[str, Dict[str, float]] = {}
            for logical in uav.cameras:
                tf = uav.get_camera_transform(logical)
                if tf is not None:
                    uav_cam_poses[logical] = carla_transform_to_dict(tf)

            relay_cam_poses: Dict[str, Dict[str, float]] = {}
            for cam in relay_cams:
                try:
                    relay_cam_poses[cam.name] = carla_transform_to_dict(cam.rgb.actor.get_transform())
                except Exception:
                    relay_cam_poses[cam.name] = carla_transform_to_dict(cam.transform)

            phase_name, phase_relay = coordinator.phase_label(sim_time)

            frame_record = {
                "frame_id": save_frame_id,
                "carla_frame": int(world_frame),
                "sim_time": float(sim_time),
                "sequence": sim_cfg.get("sequence_name", "seq_001"),
                "map": sim_cfg.get("map", ""),
                "phase": phase_name,
                "expected_covis_relay": phase_relay,
                "images": images_index,
                "ugv_pose_carla": ugv_pose_dict,
                "ugv_camera_pose_carla": ugv_cam_pose_dict,
                "uav_pose_carla": uav_pose_dict,
                "uav_camera_poses_carla": uav_cam_poses,
                "relay_camera_poses_carla": relay_cam_poses,
            }
            if jsonl is not None:
                jsonl.write(frame_record)
            if ugv_pose_csv is not None:
                ugv_pose_csv.write(save_frame_id, sim_time, ugv_pose_dict)
            if uav_pose_csv is not None:
                uav_pose_csv.write(save_frame_id, sim_time, uav_pose_dict)

            counter.saved += 1
            save_frame_id += 1

        # ----- trajectory.yaml summary -----------------------------
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
                "phases": [
                    {"name": p.name, "t_start": p.t_start, "t_end": p.t_end,
                     "expected_relay": p.expected_relay}
                    for p in coordinator.phases
                ],
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

        if dressing is not None:
            stop_walker_controllers(dressing)

        for w in (jsonl, ugv_pose_csv, uav_pose_csv):
            if w is not None:
                w.close()

        if registry is not None:
            registry.destroy_all()

        if carla_world is not None:
            carla_world.shutdown()

        wall_dt = time.time() - start_wall
        LOGGER.info(
            "Done. wall_time=%.2fs, ticks=%d, saved_frames=%d, dropped_sensor_frames=%d, output=%s",
            wall_dt, counter.tick_count, counter.saved, counter.dropped_sensor_frames, paths.root,
        )

    return exit_code


if __name__ == "__main__":
    sys.exit(main())
