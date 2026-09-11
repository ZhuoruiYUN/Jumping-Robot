#!/bin/bash
# Zero-update check: migrate the stable 37-D policy in memory and evaluate it
# in the planner environment while preserving the stable task's airborne reset.
set -euo pipefail

NUM_ENVS="${1:-256}"
MAX_STEPS="${2:-1500}"
ISAAC_SIM_ROOT="${ISAAC_SIM_ROOT:-/home/terry/Documents/isaacsim/isaac-sim-standalone-5.1.0-linux-x86_64}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BASELINE="${3:-$SCRIPT_DIR/logs/rsl_rl/quadhopper_stable_baseline/2026-08-05_01-28-58/model_498.pt}"
MAX_DISTANCE="${4:-0.00}"
TARGET_HEIGHT="${5:-1.00}"

echo "[BASELINE-CONTRACT] distance=0.00--${MAX_DISTANCE} m, height=${TARGET_HEIGHT} m"

"$ISAAC_SIM_ROOT/python.sh" "$SCRIPT_DIR/play_planner_circular.py" \
  --headless \
  --no_debug_vis \
  --num_envs "$NUM_ENVS" \
  --max_steps "$MAX_STEPS" \
  --checkpoint "$BASELINE" \
  --route random_two_hop \
  --distance_stage custom \
  --short_radius_min 0.00 \
  --short_radius_max "$MAX_DISTANCE" \
  --long_radius_min 0.00 \
  --long_radius_max "$MAX_DISTANCE" \
  --max_turn_angle 15 \
  --height_stage high \
  --height_high "$TARGET_HEIGHT" \
  --start_from_random_drop \
  --initial_drop_height_min 1.42 \
  --initial_drop_height_max 2.12 \
  --relative_next_hop \
  --baseline_compatible_stance_reference \
  --planner_reference_blend 0.0 \
  --lenient_apex_hit
