#!/bin/bash
# Conservative rolling-target fine-tune from the calibrated stable baseline.
# Begin at the verified 0--0.02 m support and expand only to 0.05 m.
set -euo pipefail

NUM_ENVS="${1:-256}"
ISAAC_SIM_ROOT="${ISAAC_SIM_ROOT:-/home/terry/Documents/isaacsim/isaac-sim-standalone-5.1.0-linux-x86_64}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BASELINE="${2:-$SCRIPT_DIR/logs/rsl_rl/quadhopper_stable_calibrated_20260907/2026-09-07_16-13-55/model_499.pt}"

"$ISAAC_SIM_ROOT/python.sh" "$SCRIPT_DIR/train_planner_circular.py" \
  --headless \
  --num_envs "$NUM_ENVS" \
  --checkpoint "$BASELINE" \
  --route random_two_hop \
  --distance_stage custom \
  --short_radius_min 0.00 \
  --short_radius_max 0.05 \
  --long_radius_min 0.00 \
  --long_radius_max 0.05 \
  --max_turn_angle 15 \
  --height_stage high \
  --start_from_random_drop \
  --initial_drop_height_min 1.42 \
  --initial_drop_height_max 2.12 \
  --relative_next_hop \
  --baseline_warmstart_finetune \
  --smooth_distance_curriculum \
  --distance_curriculum_iterations 40 \
  --retry_target_on_miss \
  --freeze_baseline_actor \
  --baseline_compatible_stance_reference \
  --planner_reference_blend 0.0 \
  --fine_tune_lr 2e-6 \
  --fixed_action_std 0.01 \
  --iterations 80 \
  "${@:3}"
