"""Check whether candidate relay positions can actually see the street.

Strategy: spawn an invisible camera at each candidate (location, look_at)
pair, do a CARLA raycast from the camera origin to the look_at point, and
see how many waypoints are within ray distance — i.e. is the street really
in line of sight, or is there a building in the way?

We also visualize 4 candidate UAV cruise heights (over the y=13 lane at
x=-30, 0, 30) by running a downward raycast from (x, 13, 30) and reporting
how far it falls before hitting something. If it's >=29 m, the line of
sight to the street is clear.
"""
import carla
c = carla.Client("localhost", 2000); c.set_timeout(10.0)
world = c.get_world()

candidates = [
    # (label, camera location, look_at) — all positions designed to be on
    # the SIDEWALK at y ~ 11 (south side of the y=13 lane) or y ~ 19
    # (north side of the y=17 lane), and look DOWN at the lane.
    ("relay_01_v2",  ( -30.0,  19.0, 8.0),  (-30.0, 13.0, 0.5)),
    ("relay_02_v2",  (  30.0,  19.0, 8.0),  ( 30.0, 13.0, 0.5)),
    ("relay_03_v2",  ( -30.0,  31.0, 8.0),  (-30.0, 25.0, 0.5)),
    ("relay_04_v2",  (  30.0,  31.0, 8.0),  ( 30.0, 25.0, 0.5)),
    ("relay_05_v2",  (   0.0,  19.0, 9.0),  (  0.0, 13.0, 0.5)),
]

def raycast(world, src, dst):
    p0 = carla.Location(*src); p1 = carla.Location(*dst)
    hits = world.cast_ray(p0, p1)
    return hits

print("=== Camera -> look_at line of sight ===")
for label, src, dst in candidates:
    hits = raycast(world, src, dst)
    nonself = [h for h in hits if h.label != carla.CityObjectLabel.NONE]
    dist_full = ((src[0]-dst[0])**2 + (src[1]-dst[1])**2 + (src[2]-dst[2])**2)**0.5
    print(f"\n{label}: src={src} dst={dst} dist={dist_full:.2f}m")
    print(f"  hits along ray: {len(hits)} total, {len(nonself)} non-empty")
    for h in hits[:6]:
        print(f"    @({h.location.x:.1f},{h.location.y:.1f},{h.location.z:.1f}) label={h.label}")

# UAV cruise heights — downward raycast at (x, 13, 30)
print("\n\n=== UAV downward LOS at altitude 30 m ===")
for x in [-30.0, -15.0, 0.0, 15.0, 30.0]:
    src = (x, 13.0, 30.0)
    dst = (x, 13.0, 0.0)
    hits = raycast(world, src, dst)
    print(f"\nUAV at ({x:.1f}, 13, 30):")
    for h in hits[:4]:
        print(f"    @({h.location.x:.1f},{h.location.y:.1f},{h.location.z:.1f}) label={h.label}")

# Check the y=24/28 streets too
print("\n\n=== UAV downward LOS at altitude 30 m over y=25 street ===")
for x in [-30.0, 0.0, 30.0]:
    src = (x, 25.0, 30.0)
    dst = (x, 25.0, 0.0)
    hits = raycast(world, src, dst)
    print(f"\nUAV at ({x:.1f}, 25, 30):")
    for h in hits[:4]:
        print(f"    @({h.location.x:.1f},{h.location.y:.1f},{h.location.z:.1f}) label={h.label}")
