"""Find Town10HD spawn points + driveable points near each relay's look_at.

For each relay, we report the closest spawn point and the closest driveable
waypoint on the road network. This grounds our trajectory waypoints to the
real road graph instead of arbitrary (x, y) coordinates.
"""
import json
import carla

c = carla.Client("localhost", 2000); c.set_timeout(10.0)
world = c.get_world()
amap = world.get_map()
sp = amap.get_spawn_points()

# Relay look_ats from default.yaml
relays = {
    "relay_01": (-10.0,  10.0, 1.5),
    "relay_02": ( 10.0,  10.0, 1.5),
    "relay_03": ( 10.0, -10.0, 1.5),
    "relay_04": (-10.0, -10.0, 1.5),
    "relay_05": (  0.0,   0.0, 1.5),
}

print(f"Total spawn points: {len(sp)}")
print(f"Map: {amap.name}")
print()

# Closest spawn point to each relay's look_at
out = {"map": amap.name, "spawn_points_total": len(sp), "relays": {}}
for rname, (rx, ry, rz) in relays.items():
    best = None; best_d = 1e18; best_idx = -1
    for i, p in enumerate(sp):
        d = ((p.location.x - rx)**2 + (p.location.y - ry)**2) ** 0.5
        if d < best_d:
            best_d = d; best = p; best_idx = i
    # Also find nearest driveable waypoint to relay center
    wp = amap.get_waypoint(carla.Location(x=rx, y=ry, z=rz),
                            project_to_road=True,
                            lane_type=carla.LaneType.Driving)
    print(f"{rname} look_at=({rx:.1f}, {ry:.1f})")
    print(f"  closest spawn pt #{best_idx}: ({best.location.x:.2f}, {best.location.y:.2f}, yaw={best.rotation.yaw:.1f})  dist={best_d:.2f}m")
    if wp is not None:
        wl = wp.transform.location
        print(f"  nearest driving wp:    ({wl.x:.2f}, {wl.y:.2f})  dist_to_lookat={((wl.x-rx)**2+(wl.y-ry)**2)**0.5:.2f}m  road_id={wp.road_id} lane_id={wp.lane_id}")
    out["relays"][rname] = {
        "lookat": [rx, ry, rz],
        "closest_spawn": {"index": best_idx,
                          "x": best.location.x, "y": best.location.y,
                          "yaw": best.rotation.yaw, "dist": best_d},
        "nearest_driving_wp": (
            {"x": wp.transform.location.x, "y": wp.transform.location.y,
             "road_id": wp.road_id, "lane_id": wp.lane_id}
            if wp is not None else None
        ),
    }
    print()

# Pick a "near relay_01" spawn point and a "near relay_03" spawn point as
# distinct anchors so the UGV has a meaningful inter-relay drive.
print("=== UGV spawn candidates near relays (top 5 by distance) ===")
for r_target in ["relay_01", "relay_03"]:
    rx, ry, _ = relays[r_target]
    cand = sorted(((((p.location.x - rx)**2 + (p.location.y - ry)**2) ** 0.5), i, p)
                  for i, p in enumerate(sp))
    print(f"\nNear {r_target}:")
    for d, i, p in cand[:5]:
        print(f"  #{i}: ({p.location.x:.2f}, {p.location.y:.2f}) yaw={p.rotation.yaw:.1f} dist={d:.2f}m")

with open("/tmp/relay_anchors.json", "w") as f:
    json.dump(out, f, indent=2)
print("\nWrote /tmp/relay_anchors.json")
