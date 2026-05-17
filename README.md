# air_ground_relay_collect

A reproducible **air-ground-relay** data collection pipeline for
[CARLA-Air](https://github.com/MIT-AVT/CARLA-Air) — the hybrid simulator that
runs a CARLA world and an AirSim multirotor side-by-side in the same Unreal
process.

The pipeline drives:

1. one **ground vehicle (UGV)** in CARLA with an attached RGB-D *or* stereo
   camera rig,
2. one **aerial vehicle (UAV)** in AirSim with front + downward RGB / depth
   cameras,
3. a configurable set of **fixed relay cameras** (CARLA sensor actors that
   look at the action like a city surveillance camera would),

and writes a time-synchronized, self-describing dataset suitable for
multi-view / large-baseline SLAM, NeRF-style reconstruction, or image
matching benchmarks.

There is **no learning code, no SLAM**, and **no automatic overlap labelling**
in this repo — it is purely a data-collection harness.

---

## 1. What the pipeline produces

```
AirGroundRelay-Sim/
├── dataset_config.yaml            # snapshot of the YAML used at runtime
└── sequences/
    └── seq_001/
        ├── trajectory.yaml         # short summary (duration, frame count, ...)
        ├── frames.jsonl            # 1 JSON per saved frame; see Section 4
        ├── config_snapshot.yaml
        ├── collect.log
        ├── calibration/
        │   ├── ugv_intrinsics.json
        │   ├── uav_intrinsics.json
        │   ├── relay_intrinsics.json
        │   └── relay_extrinsics.json
        ├── ugv/
        │   ├── front_rgb/000000.png ...
        │   ├── front_depth/000000.npy   # float32, meters
        │   ├── stereo_left/   (stereo mode)
        │   ├── stereo_right/  (stereo mode)
        │   └── pose.csv             # frame_id, sim_time, x, y, z, roll, pitch, yaw
        ├── uav/
        │   ├── front_rgb/
        │   ├── front_depth/
        │   ├── down_rgb/
        │   ├── down_depth/
        │   └── pose.csv             # frame_id, sim_time, x, y, z, qw, qx, qy, qz
        └── relay/
            ├── relay_01/
            │   ├── rgb/
            │   ├── depth/
            │   └── extrinsic.json
            ├── relay_02/
            └── ...
```

Coordinate systems are **not** unified between CARLA and AirSim. Each pose is
labelled with `_carla` or `_airsim_ned` so the consumer can pick a convention
and apply the appropriate transform (see Section 8).

---

## 2. Environment requirements

| Component         | Recommended version                   | Notes |
| ----------------- | ------------------------------------- | ----- |
| Python            | 3.8 / 3.9 / 3.10 (match CARLA-Air)    | A single venv must satisfy *both* CARLA and AirSim. |
| CARLA-Air         | the version shipping with CARLA 0.9.x | Provides `carla` python egg + AirSim plugin. |
| `numpy`           | >= 1.22                               | |
| `opencv-python`   | >= 4.5                                | Used for PNG I/O and color conversion. |
| `PyYAML`          | >= 6.0                                | |
| `airsim`          | >= 1.8.1                              | `pip install airsim`. |
| `carla`           | bundled `.egg` in CARLA-Air           | Put `PythonAPI/carla/dist/carla-*.egg` on `PYTHONPATH`. |

Quick install of the pip-installable deps:

```bash
pip install -r requirements.txt
```

If `import carla` fails, you most likely need:

```bash
# Linux
export PYTHONPATH="$PYTHONPATH:/path/to/CARLA-Air/PythonAPI/carla/dist/carla-0.9.15-py3.8-linux-x86_64.egg"

# Windows PowerShell
$env:PYTHONPATH = "$env:PYTHONPATH;D:\path\to\CARLA-Air\PythonAPI\carla\dist\carla-0.9.15-py3.8-win-amd64.egg"
```

---

## 3. Starting CARLA-Air

In one shell (Linux example):

```bash
cd /path/to/CARLA-Air
./CarlaUE4.sh -RenderOffScreen -nosound -quality-level=Epic
```

On Windows:

```powershell
cd D:\path\to\CARLA-Air
.\CarlaUE4.exe -RenderOffScreen -nosound -quality-level=Epic
```

CARLA-Air launches both servers from a single executable:

- **CARLA** RPC on `localhost:2000`
- **AirSim** RPC on `localhost:41451`

You can change these in `configs/default.yaml` if you forwarded different
ports.

---

## 4. Running the collection script

From the `air_ground_relay_collect/` directory:

```bash
python scripts/main_collect.py --config configs/default.yaml
```

The script will:

1. connect to CARLA and load the configured map,
2. switch CARLA into synchronous mode (`fixed_delta_seconds = 0.05`),
3. spawn the UGV + cameras + relay cameras,
4. connect to AirSim, arm, takeoff, and start the UAV controller,
5. write `calibration/*.json` and the snapshot config,
6. run a `world.tick()` loop for `duration_seconds`, decimating to
   `save_hz` frames per second of saved data,
7. on exit (graceful, Ctrl+C, or exception) restore the original CARLA
   world settings, disarm the multirotor, release API control, and destroy
   every actor/sensor it spawned.

You can also use the convenience wrapper:

```bash
bash examples/example_default_run.sh
```

---

## 5. Frame record schema (`frames.jsonl`)

Each line is a single JSON object:

```json
{
  "frame_id": 120,
  "carla_frame": 4321,
  "sim_time": 6.0,
  "sequence": "seq_001",
  "map": "Town10HD",
  "images": {
    "ugv_front_rgb": "ugv/front_rgb/000120.png",
    "ugv_front_depth": "ugv/front_depth/000120.npy",
    "uav_front_rgb": "uav/front_rgb/000120.png",
    "uav_front_depth": "uav/front_depth/000120.npy",
    "uav_down_rgb": "uav/down_rgb/000120.png",
    "relay_01_rgb": "relay/relay_01/rgb/000120.png",
    "relay_01_depth": "relay/relay_01/depth/000120.npy"
  },
  "ugv_pose_carla":          { "x":..., "y":..., "z":..., "roll":..., "pitch":..., "yaw":... },
  "ugv_camera_pose_carla":   { ... same schema ... },
  "uav_pose_airsim_ned":     { "x":..., "y":..., "z":..., "qw":..., "qx":..., "qy":..., "qz":... },
  "uav_state_airsim_ned":    { "position_ned": {...}, "orientation_quat_wxyz": {...}, ... },
  "uav_camera_pose_airsim_ned": { "front_rgb": {...}, "down_rgb": {...} },
  "relay_camera_poses_carla":   { "relay_01": {...}, ... }
}
```

All paths are **relative to the sequence root** (`sequences/seq_001/`).

---

## 6. RGB-D vs Stereo mode for the UGV

`configs/default.yaml`:

```yaml
ugv:
  sensor_mode: rgbd      # or "stereo"
  camera:
    resolution: [1280, 720]
    fov: 90.0
    stereo_baseline_m: 0.2
    stereo_capture_depth: false   # only relevant in stereo mode
```

| Mode    | Sensors created                                | Folders                                              |
| ------- | ---------------------------------------------- | ---------------------------------------------------- |
| `rgbd`  | `ugv_front_rgb` + `ugv_front_depth`            | `ugv/front_rgb/`, `ugv/front_depth/`                 |
| `stereo`| `ugv_stereo_left` + `ugv_stereo_right` (+ opt. `ugv_stereo_depth`) | `ugv/stereo_left/`, `ugv/stereo_right/`, `(ugv/front_depth/)` |

Either mode is sufficient to avoid the scale ambiguity of a single monocular
camera.

---

## 7. Tuning the config

Everything is driven by `configs/default.yaml`:

| Key                                       | Meaning |
| ----------------------------------------- | ------- |
| `simulation.map`                          | CARLA map (must exist in CARLA-Air assets). |
| `simulation.fixed_delta_seconds`          | World tick in seconds (sync mode). |
| `simulation.simulation_hz` / `save_hz`    | Tick rate vs. on-disk sample rate. |
| `simulation.duration_seconds`             | How long the run lasts (sim time). |
| `simulation.weather`                      | Optional CARLA weather override; set to `null` to keep map default. |
| `ugv.control_mode`                        | `autopilot` or `waypoint`. |
| `ugv.waypoints`                           | List of `[x,y,z]` in CARLA world coords. |
| `ugv.camera.location` / `rotation`        | Mount of the front camera in the vehicle frame. |
| `uav.mode`                                | `follow_ugv` / `hover` / `waypoint`. |
| `uav.follow_offset`                       | UGV-relative offset (behind, right, altitude). |
| `uav.waypoints_ned`                       | NED waypoints for `mode=waypoint`. |
| `uav.airsim_camera_names`                 | Override if your AirSim `settings.json` uses non-default camera names. |
| `relay_cameras.cameras`                   | List of fixed cameras with `location`+`look_at`. |
| `coords.carla_to_airsim_ned`              | Approximate CARLA <-> AirSim origin alignment. |

---

## 8. Known limitations

- **CARLA <-> AirSim coordinates are not strictly equal.** CARLA-Air does not
  guarantee that the CARLA world origin coincides with the AirSim
  PlayerStart, so we keep both poses separate in the dataset and only
  approximate the conversion inside the `follow_ugv` controller via
  `src/geometry.py:carla_to_airsim_ned`. If you care about absolute
  alignment, *measure* the offset on your build and set
  `coords.carla_to_airsim_ned.origin_offset_carla` + `yaw_offset_deg`.
- **Waypoint follower for the UGV is a baseline.** It implements a simple
  pure-pursuit-style steer/throttle controller with a P speed loop. For
  long routes or tight corners use `control_mode: autopilot` and let the
  Traffic Manager drive.
- **CARLA depth decoding** uses the standard
  `(R + G·256 + B·256²) / (256³ − 1) · far_plane` formula with
  `far_plane = 1000 m`. Verify on your specific CARLA-Air build — some
  forks use a different far plane (see TODO in `geometry.py`).
- **AirSim image resolution** is set in `settings.json` and **cannot** be
  read back via `simGetCameraInfo`. The `uav_intrinsics.json` file falls
  back to `640 × 480` if it cannot infer the size; edit it once to match
  your AirSim config.
- **AirSim runs asynchronously** — its image timestamps are not tied to
  CARLA's `world.tick()`. We sample AirSim once per saved frame and accept
  small (~ a few ms) skew. For tighter sync, switch AirSim to its
  ClockSpeed=0 setting and step it manually.
- **No automatic overlap labels.** As stated in the requirements, we only
  collect raw data here; downstream tools can build overlap masks /
  matching pairs by projecting depth between camera poses.

---

## 9. Project layout

```
air_ground_relay_collect/
├── README.md
├── requirements.txt
├── configs/
│   └── default.yaml
├── scripts/
│   └── main_collect.py
├── src/
│   ├── __init__.py
│   ├── config.py            # YAML loader + validator
│   ├── carla_client.py      # CARLA connection + sync mode lifecycle
│   ├── airsim_client.py     # AirSim multirotor wrapper
│   ├── actors.py            # ActorRegistry + spawn_ugv
│   ├── sensors.py           # CARLA sensor wrappers + queue helpers
│   ├── trajectory.py        # UGV & UAV controllers
│   ├── sync.py              # tick-vs-save scheduling
│   ├── save_utils.py        # paths, image / json / csv writers
│   ├── geometry.py          # intrinsics, look_at, depth decode, coord helpers
│   └── logging_utils.py
└── examples/
    └── example_default_run.sh
```

---

## 10. Troubleshooting checklist

- `RuntimeError: time-out of 10000ms while waiting for the simulator` →
  CARLA server isn't reachable on the configured port.
- `ImportError: No module named carla` → CARLA egg not on `PYTHONPATH`.
- `ImportError: No module named airsim` → `pip install airsim`.
- AirSim `simGetImages` returns empty responses → the camera name in
  `uav.airsim_camera_names` doesn't match your `settings.json`.
- All depth values are zero → the depth sensor was created but the world
  tick was reached before the sensor produced its first frame. Increase
  `duration_seconds` or wait one extra tick before saving.
- The UAV does not move → it didn't take off (check `enableApiControl` and
  `armDisarm` log lines).
