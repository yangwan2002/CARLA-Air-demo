"""Capture a top-down satellite view of Town10HD's central area for planning.

Spawns a single high-altitude camera at (0, 0, 250 m) facing straight down
(pitch=-90), takes one frame, saves it to /tmp/town10hd_topdown.png.
"""
import time
import carla

client = carla.Client("localhost", 2000)
client.set_timeout(10.0)
world = client.get_world()
amap = world.get_map()
print(f"Map: {amap.name}")

# Settings: capture a high-resolution shot. Town10HD's playable area is
# roughly 250m x 250m around the origin, so altitude=250 with FOV=90 covers
# 500m x 500m on the ground — more than enough.
ALT = 250.0
RES_W, RES_H = 2048, 2048
FOV = 90.0

bp_lib = world.get_blueprint_library()
bp = bp_lib.find("sensor.camera.rgb")
bp.set_attribute("image_size_x", str(RES_W))
bp.set_attribute("image_size_y", str(RES_H))
bp.set_attribute("fov", str(FOV))
bp.set_attribute("sensor_tick", "0.0")
for k, v in (
    ("motion_blur_intensity", "0.0"),
    ("lens_flare_intensity", "0.0"),
    ("bloom_intensity", "0.0"),
    ("chromatic_aberration_intensity", "0.0"),
):
    try:
        bp.set_attribute(k, v)
    except Exception:
        pass

tf = carla.Transform(
    carla.Location(x=0.0, y=0.0, z=ALT),
    carla.Rotation(roll=0.0, pitch=-90.0, yaw=0.0),  # nadir, north up
)

# We need at least 1 world.tick for the camera to render in sync mode.
# Switch the world into sync mode briefly so the capture is reliable.
settings = world.get_settings()
restore_settings = None
if not settings.synchronous_mode:
    restore_settings = settings
    new_settings = world.get_settings()
    new_settings.synchronous_mode = True
    new_settings.fixed_delta_seconds = 0.05
    world.apply_settings(new_settings)
    print("[switched into sync mode for capture]")

cam = world.spawn_actor(bp, tf)
print(f"camera spawned at z={ALT}, fov={FOV}, res={RES_W}x{RES_H}")

import queue
q: "queue.Queue" = queue.Queue()
cam.listen(lambda data: q.put(data))

# Wait a couple of ticks so the engine flushes shaders.
for _ in range(5):
    world.tick()
    time.sleep(0.05)

img = q.get(timeout=10.0)
print(f"got frame {img.frame}, {img.width}x{img.height}")
img.save_to_disk("/tmp/town10hd_topdown.png")
print("saved /tmp/town10hd_topdown.png")

cam.stop()
cam.destroy()

# Also dump a smaller "annotated reference" with markings for the 155 spawn
# points so we can correlate the photo with the road graph.
import json
sp = amap.get_spawn_points()
sp_data = []
for i, p in enumerate(sp):
    sp_data.append({"i": i, "x": float(p.location.x), "y": float(p.location.y),
                    "yaw": float(p.rotation.yaw)})
with open("/tmp/town10hd_spawnpoints.json", "w") as f:
    json.dump(sp_data, f)
print(f"saved {len(sp_data)} spawn points to /tmp/town10hd_spawnpoints.json")

# Also dump central drivable waypoints so we can overlay road centerlines.
wps = amap.generate_waypoints(2.0)
central = [w for w in wps if abs(w.transform.location.x) < 100 and abs(w.transform.location.y) < 100]
wp_data = [{"x": float(w.transform.location.x), "y": float(w.transform.location.y),
            "yaw": float(w.transform.rotation.yaw),
            "road_id": int(w.road_id), "lane_id": int(w.lane_id)} for w in central]
with open("/tmp/town10hd_waypoints.json", "w") as f:
    json.dump(wp_data, f)
print(f"saved {len(wp_data)} central waypoints to /tmp/town10hd_waypoints.json")

# Restore async mode if we changed it.
if restore_settings is not None:
    world.apply_settings(restore_settings)
    print("[restored original world settings]")
