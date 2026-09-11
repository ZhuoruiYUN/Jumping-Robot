#!/bin/bash
# Preserve the corrected-physics stable policy and learn only bounded motor
# corrections for the first 10 cm horizontal curriculum.
set -euo pipefail

NUM_ENVS="${1:-256}"
ITERATIONS="${2:-100}"
ISAAC_SIM_ROOT="${ISAAC_SIM_ROOT:-/home/terry/Documents/isaacsim/isaac-sim-standalone-5.1.0-linux-x86_64}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
TEACHER="${3:-$SCRIPT_DIR/outputs/planner_migrations/play_initial_policy_37d_to_43d.pt}"

"$ISAAC_SIM_ROOT/python.sh" "$SCRIPT_DIR/experiments/random_two_hop/train_two_hop_residual.py" \
  --headless \
  --num_envs "$NUM_ENVS" \
  --iterations "$ITERATIONS" \
  --teacher_checkpoint "$TEACHER" \
  --target_height 1.0 \
  --short_radius_max 0.05 \
  --long_radius_max 0.05 \
  --curriculum_radius_max 0.02 \
  --max_turn_angle 15 \
  --curriculum_iterations 40 \
  --retry_target_on_miss \
  --start_from_random_drop \
  --initial_drop_height_min 1.42 \
  --initial_drop_height_max 2.12 \
  --collective_scale 0.01 \
  --attitude_scale 0.025 \
  --residual_slew_rate 0.006 \
  --residual_penalty_scale 2 \
  --slew_penalty_scale 1 \
  --next_tilt_deg 0.0 \
  --init_noise_std 0.04 \
  --save_interval 10
