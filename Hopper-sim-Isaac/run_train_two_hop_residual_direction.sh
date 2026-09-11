#!/bin/bash
# Continue the stable bounded residual on commands above the measured ~0.10 m
# landing-noise floor, so the policy receives an observable direction signal.
set -euo pipefail

NUM_ENVS="${1:-256}"
ITERATIONS="${2:-100}"
ISAAC_SIM_ROOT="${ISAAC_SIM_ROOT:-/home/terry/Documents/isaacsim/isaac-sim-standalone-5.1.0-linux-x86_64}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
TEACHER="${3:-$SCRIPT_DIR/outputs/planner_migrations/play_initial_policy_37d_to_43d.pt}"
RESIDUAL="${4:-$SCRIPT_DIR/logs/rsl_rl/quadhopper_planner_random_two_hop_v55_residual_continuous_queue_ppo/2026-09-07_17-16-10/model_99.pt}"

"$ISAAC_SIM_ROOT/python.sh" "$SCRIPT_DIR/experiments/random_two_hop/train_two_hop_residual.py" \
  --headless \
  --num_envs "$NUM_ENVS" \
  --iterations "$ITERATIONS" \
  --teacher_checkpoint "$TEACHER" \
  --checkpoint "$RESIDUAL" \
  --target_height 1.0 \
  --target_tolerance 0.15 \
  --short_radius_min 0.12 \
  --short_radius_max 0.20 \
  --long_radius_min 0.12 \
  --long_radius_max 0.20 \
  --curriculum_radius_min 0.12 \
  --curriculum_radius_max 0.20 \
  --max_turn_angle 15 \
  --curriculum_iterations 40 \
  --curriculum_iteration_offset 40 \
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
