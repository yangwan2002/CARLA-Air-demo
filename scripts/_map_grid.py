"""Map the drivable street grid around the city center (±60 m) of Town10HD."""
import carla
c = carla.Client("localhost", 2000); c.set_timeout(10.0)
amap = c.get_world().get_map()

# Generate dense waypoints, filter to central 60x60
wps = amap.generate_waypoints(2.0)
central = [w for w in wps if abs(w.transform.location.x) < 60 and abs(w.transform.location.y) < 60]

# Bin to 4m grid for an ASCII map
import collections
grid = collections.defaultdict(int)
for w in central:
    gx = round(w.transform.location.x / 4) * 4
    gy = round(w.transform.location.y / 4) * 4
    grid[(gx, gy)] += 1

# Print grid: x range [-60, 60] step 4, y range [-60, 60] step 4
xs = list(range(-60, 64, 4))
ys = list(range(-60, 64, 4))
print(f"central drivable wps: {len(central)}")
print("\nDrivable street grid (.=no road, #=road, x in cols, y in rows; +x right, +y down):")
print("       " + "".join(f"{x:>3}" if x % 12 == 0 else "   " for x in xs))
for y in ys:
    row = f"y={y:>4}: "
    for x in xs:
        row += " # " if grid.get((x, y), 0) > 0 else " . "
    print(row)

# Now the key question: where can the UGV actually drive? Find continuous
# road segments crossing the center. Pick a few "main streets" by
# inspecting which y values have many drivable points.
print("\n=== Drivable y-bands (counts per y in [-60, 60]) ===")
y_counts = collections.Counter(round(w.transform.location.y, 0) for w in central)
for y in sorted(y_counts):
    if y_counts[y] >= 5:
        print(f"  y={int(y):>4}: {y_counts[y]} wps")
print("\n=== Drivable x-bands ===")
x_counts = collections.Counter(round(w.transform.location.x, 0) for w in central)
for x in sorted(x_counts):
    if x_counts[x] >= 5:
        print(f"  x={int(x):>4}: {x_counts[x]} wps")
