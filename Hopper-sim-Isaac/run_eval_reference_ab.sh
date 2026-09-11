#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

ISAAC_PYTHON="/home/terry/Documents/isaacsim/isaac-sim-standalone-5.1.0-linux-x86_64/python.sh"
CHECKPOINT="logs/rsl_rl/quadhopper_stable_calibrated_accuracy_20260907/2026-09-07_18-01-15/model_598.pt"

COMMON_ARGS=(
  play_planner_circular.py
  --headless
  --no_debug_vis
  --num_envs 256
  --max_steps 1500
  --seed 42
  --checkpoint "$CHECKPOINT"
  --route random_two_hop
  --distance_stage custom
  --short_radius_min 0.15
  --short_radius_max 0.15
  --long_radius_min 0.25
  --long_radius_max 0.25
  --max_turn_angle 30
  --height_stage high
  --height_high 1.0
  --target_tolerance 0.15
  --planner_reference_blend 0.0
  --baseline_compatible_stance_reference
  --relative_next_hop
  --pair_restart_queue
  --lenient_apex_hit
  --play_motor_time_constant 0.0674
  --compact_eval
)

echo "[REFERENCE-AB] A: stationary planner midpoint XY"
"$ISAAC_PYTHON" "${COMMON_ARGS[@]}"

echo "[REFERENCE-AB] B: stationary physical landing-target XY"
"$ISAAC_PYTHON" "${COMMON_ARGS[@]}" --direct_landing_reference
