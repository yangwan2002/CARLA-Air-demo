"""Verify v2 candidate positions: ALL relays + UGV + UAV confined to the
y in [10, 32] strip where Town10HD has clean drivable corridors.

The y=13/17 lane and y=25/28 lane run parallel about 12 m apart, with a
plaza between them. We mount relays on the OUTER sidewalks (y<13 and y>28)
looking inward at the lanes, so no building blocks the ray.
"""
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

# === Candidate v2 relays ===
relays = [
    # y=25/28 lane, mount relays on north sidewalk (y=32) looking south at lane
    ("relay_01", (-30.0, 32.0, 8.0), (-30.0, 28.0, 0.4)),
    ("relay_02", (  0.0, 32.0, 9.0), (  0.0, 28.0, 0.4)),
    ("relay_03", ( 30.0, 32.0, 8.0), ( 30.0, 28.0, 0.4)),
    # y=13/17 lane, mount relays on south sidewalk (y=10) looking north at lane
    ("relay_04", (-15.0, 10.0, 8.0), (-15.0, 17.0, 0.4)),
    ("relay_05", ( 15.0, 10.0, 8.0), ( 15.0, 17.0, 0.4)),
]
print("=== RELAY LOS (v2) ===")
all_pass = True
for name, src, dst in relays:
    ok, msg = los_check(world, name, src, dst)
    status = "PASS" if ok else "FAIL"
    print(f"  {status}  {name}: cam={src} -> {dst}  {msg}")
    if not ok:
        all_pass = False

# === UAV nadir LOS at 30 m, on the lanes ===
print("\n=== UAV NADIR LOS at 30m ===")
uav_pts = [
    (-30.0, 25.0), (-15.0, 25.0), (0.0, 25.0), (15.0, 25.0), (30.0, 25.0),
    (-30.0, 17.0), (-15.0, 17.0), (0.0, 17.0), (15.0, 17.0), (30.0, 17.0),
    (-30.0, 21.0), (  0.0, 21.0), (30.0, 21.0),  # over plaza between two lanes
]
for x, y in uav_pts:
    ok, msg = los_check(world, "uav", (x, y, 30.0), (x, y, 0.0), tol_m=0.5)
    print(f"  {'OK ' if ok else 'BAD'}  uav ({x:5.1f}, {y:5.1f}, 30) -> {msg}")

# === UGV waypoints — must snap to a real driving lane within 2m ===
amap = world.get_map()
print("\n=== UGV WAYPOINT VALIDITY (must snap < 2m) ===")
ugv_pts = [
    ("spawn",       (-0.76, 24.61)),  # spawn point #49
    ("p1_east",     ( 30.0, 25.0)),   # drive east on y=25
    ("p2_corner",   ( 40.0, 25.0)),   # turn at intersection
    ("p2_south",    ( 40.0, 17.0)),   # come south to y=17
    ("p3_west",     (-30.0, 17.0)),   # drive west on y=17
    ("p4_loopback", (-40.0, 25.0)),   # close the loop back to y=25
]
ugv_ok = True
for name, (x, y) in ugv_pts:
    wp = amap.get_waypoint(carla.Location(x=x, y=y, z=0.5),
                            project_to_road=True,
                            lane_type=carla.LaneType.Driving)
    if wp is None:
        print(f"  FAIL  {name} ({x:5.1f}, {y:5.1f}): NO LANE")
        ugv_ok = False
        continue
    d = ((wp.transform.location.x - x) ** 2 + (wp.transform.location.y - y) ** 2) ** 0.5
    snapped = (wp.transform.location.x, wp.transform.location.y)
    status = "OK  " if d < 2.0 else "FAIL"
    print(f"  {status}  {name} ({x:5.1f}, {y:5.1f}) -> ({snapped[0]:5.1f}, {snapped[1]:5.1f})  "
          f"d={d:.2f}m road={wp.road_id} lane={wp.lane_id} yaw={wp.transform.rotation.yaw:.1f}")
    if d >= 2.0:
        ugv_ok = False

print(f"\nFINAL: relays {'ALL PASS' if all_pass else 'SOME FAIL'}; "
      f"UGV waypoints {'ALL VALID' if ugv_ok else 'SOME INVALID'}")
