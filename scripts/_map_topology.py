"""Sweep more spawn points and waypoints to map out Town10HD's drivable streets."""
import carla, math, json
c = carla.Client("localhost", 2000); c.set_timeout(10.0)
amap = c.get_world().get_map()
sp = amap.get_spawn_points()

# Group spawn points by approximate cell so we can see street layout
print("=== ALL 155 spawn points (sorted by y then x) ===")
sorted_sp = sorted(enumerate(sp), key=lambda kv: (round(kv[1].location.y, 0), kv[1].location.x))
for i, p in sorted_sp[:50]:
    print(f"  #{i:3d} ({p.location.x:7.2f}, {p.location.y:7.2f}, z={p.location.z:5.2f}) yaw={p.rotation.yaw:6.1f}")
print("...")
for i, p in sorted_sp[50:100]:
    print(f"  #{i:3d} ({p.location.x:7.2f}, {p.location.y:7.2f}, z={p.location.z:5.2f}) yaw={p.rotation.yaw:6.1f}")
print("...")
for i, p in sorted_sp[100:]:
    print(f"  #{i:3d} ({p.location.x:7.2f}, {p.location.y:7.2f}, z={p.location.z:5.2f}) yaw={p.rotation.yaw:6.1f}")

# Get full topology of the road network: list all driveable waypoints in the
# central area (xy in [-80, 80]) at 5m spacing.
print("\n=== Driveable waypoints, central area, 5m grid ===")
waypoints = amap.generate_waypoints(5.0)
central = [w for w in waypoints
           if abs(w.transform.location.x) < 80 and abs(w.transform.location.y) < 80]
print(f"central driveable waypoints: {len(central)}")

# Cluster by quadrant
quadrants = {"NW": [], "NE": [], "SW": [], "SE": [], "C": []}
for w in central:
    x, y = w.transform.location.x, w.transform.location.y
    if abs(x) < 5 and abs(y) < 5:
        quadrants["C"].append(w)
    elif x < 0 and y > 0:
        quadrants["NW"].append(w)
    elif x > 0 and y > 0:
        quadrants["NE"].append(w)
    elif x < 0 and y < 0:
        quadrants["SW"].append(w)
    else:
        quadrants["SE"].append(w)
for k, ws in quadrants.items():
    print(f"  {k}: {len(ws)} wp")
    if ws:
        # sample 3
        for w in ws[:3]:
            l = w.transform.location
            print(f"    ({l.x:.1f}, {l.y:.1f}) road={w.road_id} lane={w.lane_id}")

# Find 4 distinct waypoints far from each other (forming a usable UGV loop)
import random
random.seed(0)
random.shuffle(central)
print("\n=== Sample 8 random central waypoints to use as UGV anchors ===")
for w in central[:8]:
    l = w.transform.location
    print(f"  ({l.x:7.2f}, {l.y:7.2f}, z={l.z:.2f}) yaw={w.transform.rotation.yaw:6.1f} road={w.road_id}")
