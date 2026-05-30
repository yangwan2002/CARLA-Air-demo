"""Query CARLA blueprint library — group static.prop.* by semantic category.

Run on the remote where CARLA is already up on Town10HD:
    python scripts/_query_props.py | tee /tmp/town10_props.txt

Outputs:
  1) full sorted list of static.prop.*
  2) grouped buckets (kiosk/vending/cone/barrel/...) — what's actually installable
  3) two-wheeler vehicle.* blueprints (already known but re-listed for completeness)
"""
import carla

c = carla.Client("localhost", 2000)
c.set_timeout(10.0)
world = c.get_world()
bp_lib = world.get_blueprint_library()

props = sorted(b.id for b in bp_lib.filter("static.prop.*"))

print(f"=== static.prop.* ({len(props)} blueprints) ===")
for p in props:
    print(p)

# Buckets we care about for L2 sem-rich theming.
buckets = {
    "tree":         ["tree", "cypress", "aporosa"],
    "bench/seat":   ["bench", "chair", "couch", "swing"],
    "trash/bin":    ["trash", "bin", "container"],
    "vending/atm":  ["vending", "atm"],
    "kiosk/booth":  ["kiosk", "busstop", "phonebooth"],
    "barrier/cone": ["barrier", "cone", "fence", "warning"],
    "advert/sign":  ["advert", "sign", "billboard", "mailbox"],
    "construction": ["construct", "barrel", "haybale", "pallet", "brick", "rock", "ironplank"],
    "market/cart":  ["cart", "table", "umbrella", "shoppingbag", "box", "case", "suitcase", "travelcase"],
    "park/garden":  ["lamp", "fountain", "statue", "doghouse", "trampoline", "gardenlamp"],
    "misc":         [],
}

print("\n=== Grouped buckets ===")
seen = set()
for name, kws in buckets.items():
    if not kws:
        continue
    hits = [p for p in props if any(k in p for k in kws)]
    seen.update(hits)
    print(f"\n[{name}]  {len(hits)}")
    for h in hits:
        print(f"  {h}")

leftover = [p for p in props if p not in seen]
print(f"\n[misc/uncategorized]  {len(leftover)}")
for p in leftover:
    print(f"  {p}")

print("\n=== two-wheelers (vehicle.*) ===")
veh = bp_lib.filter("vehicle.*")
two_wheel = sorted(
    v.id for v in veh
    if any(k in v.id for k in (
        "bike", "bicycle", "motor", "harley", "kawasaki", "yamaha",
        "vespa", "diamondback", "gazelle", "crossbike",
    ))
)
for tw in two_wheel:
    print(f"  {tw}")
