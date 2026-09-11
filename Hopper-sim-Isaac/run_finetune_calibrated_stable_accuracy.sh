#!/bin/bash
set -euo pipefail

NUM_ENVS="${1:-256}"
ITERATIONS="${2:-100}"
ISAAC_SIM_ROOT="${ISAAC_SIM_ROOT:-/home/terry/Documents/isaacsim/isaac-sim-standalone-5.1.0-linux-x86_64}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BASELINE="${3:-$SCRIPT_DIR/logs/rsl_rl/quadhopper_stable_calibrated_20260907/2026-09-07_16-13-55/model_499.pt}"

"$ISAAC_SIM_ROOT/python.sh" "$SCRIPT_DIR/train_stable.py" \
  --headless \
  --num_envs "$NUM_ENVS" \
  --iterations "$ITERATIONS" \
  --checkpoint "$BASELINE" \
  --accuracy_finetune \
  --experiment_name quadhopper_stable_calibrated_accuracy_20260907
