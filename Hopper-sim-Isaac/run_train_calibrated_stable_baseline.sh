#!/bin/bash
# Fresh end-to-end stable baseline using the 2026-09-07 calibrated dynamics.
set -euo pipefail

NUM_ENVS="${1:-256}"
ITERATIONS="${2:-500}"
ISAAC_SIM_ROOT="${ISAAC_SIM_ROOT:-/home/terry/Documents/isaacsim/isaac-sim-standalone-5.1.0-linux-x86_64}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

"$ISAAC_SIM_ROOT/python.sh" "$SCRIPT_DIR/train_stable.py" \
  --headless \
  --num_envs "$NUM_ENVS" \
  --iterations "$ITERATIONS" \
  --experiment_name quadhopper_stable_calibrated_20260907
