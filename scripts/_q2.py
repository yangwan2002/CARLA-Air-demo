import carla
c = carla.Client("localhost", 2000); c.set_timeout(10.0)
bp = c.get_world().get_blueprint_library()
ids = sorted(b.id for b in bp.filter("vehicle.*"))
print("total:", len(ids))
for i in ids:
    print(i)
print("--- walkers ---")
walkers = bp.filter("walker.pedestrian.*")
print("walker count:", len(walkers))
print(walkers[0].id, walkers[1].id, walkers[2].id)
