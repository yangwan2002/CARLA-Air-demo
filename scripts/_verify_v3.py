"""Verify v3 (10m, 15m horizontal) relay candidates + plan UAV speed-based path."""
import carla


def los_check(world, name, src, dst, tol_m=0.6):
    hits = world.cast_ray(carla.Location(*src), carla.Location(*dst))
    if not hits:
        return True, "no hits (open sky)"
    first = hits[0]
    label = str(first.label)
    d = ((first.location.x - dst[0]) ** 2 +
         (first.location.y - dst[1]) ** 2 +
         (first.location.z - dst[2]) ** 2) ** 0.5
    if "Roads" in label or "RoadLine" in label or "Sidewalk" in label or d <= tol_m:
        return True, f"OK {label} d={d:.2f}m"
    return False, f"BLOCKED {label} @({first.location.x:.1f},{first.location.y:.1f},{first.location.z:.1f}) d={d:.2f}m"


client = carla.Client("localhost", 2000); client.set_timeout(10.0)
world = client.get_world()

# === v3 relays: 10m height, 15m horizontal distance, FOV will be 90 in yaml ===
# North sidewalk relays look south at lanes y=25/28 (lookat y=25.5)
# South sidewalk relays look north at lanes y=13/17 (lookat y=15)
relays = [
    ("relay_01", (-30.0, 40.0, 10.0), (-30.0, 25.5, 0.4)),   # 14.5m h-dist
    ("relay_02", (  0.0, 40.0, 10.0), (  0.0, 25.5, 0.4)),
    ("relay_03", ( 30.0, 40.0, 10.0), ( 30.0, 25.5, 0.4)),
    ("relay_04", (-15.0,  0.0, 10.0), (-15.0, 15.0, 0.4)),   # 15m h-dist
    ("relay_05", ( 15.0,  0.0, 10.0), ( 15.0, 15.0, 0.4)),
]
print("=== RELAY LOS (v3: 10m high, ~15m horizontal) ===")
all_pass = True
for name, src, dst in relays:
    ok, msg = los_check(world, name, src, dst)
    status = "PASS" if ok else "FAIL"
    print(f"  {status}  {name}: cam={src} -> {dst}  {msg}")
    if not ok:
        all_pass = False

# Compute the FOV footprint each relay covers at the lane plane.
# pitch = atan2(dz, h_dist), camera FOV = 90 deg horizontal
import math
print("\n=== Relay ground-plane footprint (FOV=90°, cam pose -> lane) ===")
for name, src, dst in relays:
    dz = dst[2] - src[2]    # negative
    dxy = ((dst[0]-src[0])**2 + (dst[1]-src[1])**2) ** 0.5
    pitch_deg = math.degrees(math.atan2(dz, dxy))
    # Distance from camera to lookat:
    cam_to_lookat = (dxy*dxy + dz*dz) ** 0.5
    # Half-width on lookat plane (FOV horizontal = 90, so half_fov = 45)
    half_w = math.tan(math.radians(45)) * cam_to_lookat
    # Vertical extent: depends on aspect. For 1280x720, vertical fov ~58.7°
    half_h = math.tan(math.radians(58.7/2)) * cam_to_lookat
    print(f"  {name}: pitch={pitch_deg:.1f}°  cam->lookat dist={cam_to_lookat:.1f}m  "
          f"footprint at lane ≈ {2*half_w:.1f}m × {2*half_h:.1f}m")

# === UAV waypoint LOS (30 m altitude over the lanes & plaza) ===
print("\n=== UAV NADIR LOS at 30m ===")
uav_pts = [
    (-30.0, 17.0), (-15.0, 17.0), (0.0, 17.0), (15.0, 17.0), (30.0, 17.0),
    (-30.0, 25.0), (-15.0, 25.0), (0.0, 25.0), (15.0, 25.0), (30.0, 25.0),
    (-30.0, 21.0), (  0.0, 21.0), (30.0, 21.0),
]
for x, y in uav_pts:
    ok, msg = los_check(world, "uav", (x, y, 30.0), (x, y, 0.0), tol_m=0.5)
    print(f"  {'OK ' if ok else 'BAD'}  uav ({x:5.1f}, {y:5.1f}, 30) -> {msg}")

# UAV nadir camera ground footprint at 30m height (FOV=90 horizontal)
half_w = 30 * math.tan(math.radians(45))
half_h = 30 * math.tan(math.radians(58.7/2))
print(f"\n  UAV at 30m, FOV=90°:  footprint ≈ {2*half_w:.1f}m × {2*half_h:.1f}m")
print(f"\nFINAL: relays {'ALL PASS' if all_pass else 'SOME FAIL'}")
