"""Query CARLA blueprint library for static props and walker types."""
import sys
import carla

c = carla.Client("localhost", 2000)
c.set_timeout(10.0)
world = c.get_world()
bp_lib = world.get_blueprint_library()

print("=== static.prop.* ===")
props = sorted([b.id for b in bp_lib.filter("static.prop.*")])
for p in props:
    print(p)

print("\n=== walker.pedestrian.* (count) ===")
walkers = bp_lib.filter("walker.pedestrian.*")
print(f"{len(walkers)} walker blueprints")
print("first 5:", [w.id for w in walkers[:5]])

print("\n=== vehicle.* (count) ===")
veh = bp_lib.filter("vehicle.*")
print(f"{len(veh)} vehicle blueprints")
# bicycles / motorbikes
two_wheel = [v.id for v in veh if any(k in v.id for k in ("bike", "bicycle", "motor", "harley", "kawasaki", "yamaha", "vespa", "diamondback", "gazelle", "crossbike"))]
print("two-wheelers:", two_wheel)
