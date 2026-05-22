"""Find a valid relay_02 location that has LOS to a lane near x=0."""
import carla
client = carla.Client("localhost", 2000); client.set_timeout(10.0)
world = client.get_world()

def los(src, dst):
    hits = world.cast_ray(carla.Location(*src), carla.Location(*dst))
    if not hits: return True, "open"
    h = hits[0]
    label = str(h.label)
    d = ((h.location.x-dst[0])**2+(h.location.y-dst[1])**2+(h.location.z-dst[2])**2)**0.5
    if "Roads" in label or "RoadLine" in label or "Sidewalk" in label or d <= 0.6:
        return True, f"{label} d={d:.2f}m"
    return False, f"BLK {label} @({h.location.x:.1f},{h.location.y:.1f},{h.location.z:.1f}) d={d:.2f}m"

# Try several anchor positions for relay_02 — different sidewalks, distances.
print("=== Searching valid relay_02 covering middle x=0 area ===")
candidates = [
    # (label, cam, lookat)
    ("S-side @ y=10 cam(0,10,10)->lane_y17", (0.0, 10.0, 10.0), (0.0, 15.0, 0.4)),
    ("S-side @ y=8  cam(0,8,10) ->lane_y17", (0.0,  8.0, 10.0), (0.0, 15.0, 0.4)),
    ("N-side @ y=42 cam(0,42,10)->lane_y28", (0.0, 42.0, 10.0), (0.0, 28.0, 0.4)),
    ("N-side @ y=44 cam(0,44,10)->lane_y28", (0.0, 44.0, 10.0), (0.0, 28.0, 0.4)),
    ("N-side @ y=38 cam(0,38,10)->lane_y28", (0.0, 38.0, 10.0), (0.0, 28.0, 0.4)),
    ("N-side @ y=36 cam(0,36,10)->lane_y28", (0.0, 36.0, 10.0), (0.0, 28.0, 0.4)),
    ("offset x=5 cam(5,40,10)->lane(5,28)",  (5.0, 40.0, 10.0), (5.0, 28.0, 0.4)),
    ("offset x=-5 cam(-5,40,10)->lane(-5,28)", (-5.0, 40.0, 10.0), (-5.0, 28.0, 0.4)),
    # higher
    ("higher cam(0,40,15)->lane(0,28)",      (0.0, 40.0, 15.0), (0.0, 28.0, 0.4)),
    ("higher cam(0,42,15)->lane(0,28)",      (0.0, 42.0, 15.0), (0.0, 28.0, 0.4)),
]
for label, src, dst in candidates:
    ok, msg = los(src, dst)
    print(f"  {'OK' if ok else 'NO'}  {label}: {msg}")
