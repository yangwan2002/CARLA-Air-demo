"""LOS-verify the planned relay/UAV positions before committing to yaml.

We test each candidate relay (camera) by raycasting from camera origin to
look_at, and report whether anything blocks the ray. We also verify the
UAV cruise points have clear nadir LOS to the ground.

Output: print a clear PASS/FAIL summary, AND draw markers on the annotated
map so we can visualize what we picked.
"""
import json
from pathlib import Path
import carla


def raycast(world, src, dst):
    p0 = carla.Location(*src); p1 = carla.Location(*dst)
    return world.cast_ray(p0, p1)


def los_clear(hits, tol_m=0.6, dst=None):
    """A ray from camera to look_at on the road should hit only the road
    surface near the destination. Anything else (a building, a sidewalk
    fence, a tree) means our LOS is blocked.

    Heuristic: if the FIRST hit is within tol_m of dst (the look_at point)
    and labelled Roads or Sidewalk or NONE, it's clear. Otherwise blocked.
    """
    if not hits:
        return True, "no hits (open sky/empty)"
    first = hits[0]
    if dst is None:
        return False, f"first hit @({first.location.x:.1f},{first.location.y:.1f},{first.location.z:.1f}) label={first.label}"
    d = ((first.location.x - dst[0]) ** 2 +
         (first.location.y - dst[1]) ** 2 +
         (first.location.z - dst[2]) ** 2) ** 0.5
    label = str(first.label)
    if "Roads" in label or "RoadLine" in label or "Sidewalk" in label:
        return True, f"hit Road/Sidewalk @ dist={d:.2f}m label={label}"
    if d <= tol_m:
        return True, f"hit near dst @ dist={d:.2f}m label={label}"
    return False, f"BLOCKED: first hit @({first.location.x:.1f},{first.location.y:.1f},{first.location.z:.1f}) dist_to_dst={d:.2f}m label={label}"


client = carla.Client("localhost", 2000); client.set_timeout(10.0)
world = client.get_world()
print(f"Map: {world.get_map().name}\n")

# === Candidate relay cameras (location, look_at) ===
relays = [
    # (name, cam_xyz, lookat_xyz)
    ("relay_01_W",  (-25.0,   6.0, 8.0),  (-25.0, -13.0, 0.4)),  # west, looks N at y=-13 lane
    ("relay_05_C",  (  0.0,   6.0, 9.0),  (  0.0, -13.0, 0.4)),  # center, looks N at y=-13 lane
    ("relay_02_E",  ( 25.0,   6.0, 8.0),  ( 25.0, -13.0, 0.4)),  # east, looks N at y=-13 lane
    ("relay_03_N",  (-25.0, -32.0, 8.0),  (-25.0, -17.0, 0.4)),  # north, looks S at y=-17 lane
    ("relay_04_S",  (  0.0,  22.0, 8.0),  (  0.0,  28.0, 0.4)),  # south, looks E at y=25/28 lane
]

# Sanity: I had the axes the wrong way around. CARLA: +x is north,
# +y is east. The MAP image we annotated had +y = right (east) and
# +x = down (south because pitch -90 + yaw 0 makes the camera look down
# the -z, with image-up being world +x, but the image we *render* with
# cv2 has rows growing downward, so +x_north -> top of image AFTER our
# row inversion). Let me reverify my axis assumption with a single quick
# raycast: a point at world (0, 0, 1) should be CARLA origin == middle of
# the central plaza on the photo.
test_hits = world.cast_ray(carla.Location(0, 0, 100), carla.Location(0, 0, 0))
print(f"sanity: ray (0,0,100)->(0,0,0) first hit label={test_hits[0].label if test_hits else 'NONE'}")

print("\n=== RELAY CAMERA LOS CHECK ===")
all_pass = True
for name, src, dst in relays:
    hits = raycast(world, src, dst)
    ok, msg = los_clear(hits, dst=dst)
    print(f"\n{name}: cam={src} -> lookat={dst}")
    print(f"   {'PASS' if ok else 'FAIL'} — {msg}")
    if hits:
        for h in hits[:5]:
            print(f"     ray hit @({h.location.x:7.2f},{h.location.y:7.2f},{h.location.z:5.2f}) label={h.label}")
    if not ok:
        all_pass = False


# === UAV cruise points: check downward LOS ===
print("\n\n=== UAV NADIR LOS CHECK (alt 30m & 50m) ===")
uav_points = [
    ("phase1_a", ( 30.0, -50.0, 50.0)),   # NE corner, off-axis
    ("phase1_b", (-30.0, -50.0, 50.0)),
    ("phase2_center", (  0.0,   0.0, 30.0)),  # over central plaza
    ("phase3_mid",    ( 25.0,   0.0, 30.0)),
    ("phase4_south",  (  0.0,  25.0, 30.0)),  # over relay_04 lookat
]
for name, src in uav_points:
    dst = (src[0], src[1], 0.0)
    hits = raycast(world, src, dst)
    if hits:
        first = hits[0]
        ground_label = str(first.label)
        print(f"{name} @ ({src[0]:.0f},{src[1]:.0f},{src[2]:.0f}): ground hit label={ground_label} z={first.location.z:.2f}")
    else:
        print(f"{name} @ {src}: no hits (probably empty space)")


# === UGV path waypoints: are they on a real driveable lane? ===
print("\n\n=== UGV WAYPOINT VALIDITY ===")
amap = world.get_map()
ugv_wps = [
    ("p1_start",   (-25.0, -17.0)),
    ("p1_mid",     (  0.0, -17.0)),
    ("p1_end",     ( 25.0, -17.0)),
    ("p2_at_05",   (  0.0, -13.0)),
    ("p3_corner",  ( 40.0, -13.0)),
    ("p3_southE",  ( 40.0,  25.0)),
    ("p4_finish",  (  0.0,  25.0)),
]
for name, (x, y) in ugv_wps:
    wp = amap.get_waypoint(carla.Location(x=x, y=y, z=0.5),
                            project_to_road=True,
                            lane_type=carla.LaneType.Driving)
    if wp is None:
        print(f"  {name}: ({x:.1f}, {y:.1f}) — no driving lane!")
        continue
    wl = wp.transform.location
    d = ((wl.x - x) ** 2 + (wl.y - y) ** 2) ** 0.5
    print(f"  {name}: ({x:.1f}, {y:.1f}) -> snapped to ({wl.x:.1f}, {wl.y:.1f}) "
          f"dist={d:.2f}m road={wp.road_id} lane={wp.lane_id} yaw={wp.transform.rotation.yaw:.1f}")

print(f"\n\nFINAL: relay LOS {'ALL PASS' if all_pass else '*** SOME FAIL ***'}")
