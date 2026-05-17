#!/usr/bin/env bash
# Quick example: collect 90 s of data with the default config.
#
# Run from the air_ground_relay_collect/ directory:
#   bash examples/example_default_run.sh
#
# Prerequisites:
#   1. CARLA-Air server is already running (CarlaUE4 + AirSim plugin).
#   2. `carla` and `airsim` python packages are importable.
#   3. CARLA port 2000 and AirSim port 41451 are reachable.

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${HERE}/.." && pwd)"

cd "${REPO_ROOT}"

python scripts/main_collect.py \
    --config configs/default.yaml \
    --log-level INFO
