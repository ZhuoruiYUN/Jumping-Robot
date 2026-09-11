#!/bin/bash
# Contract test: a zero-output residual must exactly preserve the frozen
# baseline teacher before residual learning is trusted.
set -euo pipefail

NUM_ENVS="${1:-256}"
EVAL_STEPS="${2:-1500}"
ISAAC_SIM_ROOT="${ISAAC_SIM_ROOT:-/home/terry/Documents/isaacsim/isaac-sim-standalone-5.1.0-linux-x86_64}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
RUN_DIR="${3:-$SCRIPT_DIR/logs/rsl_rl/quadhopper_planner_random_two_hop_v55_residual_continuous_queue_ppo/2026-09-07_15-46-27}"
TEACHER="${4:-$SCRIPT_DIR/outputs/planner_migrations/play_initial_policy_37d_to_43d.pt}"

"$ISAAC_SIM_ROOT/python.sh" "$SCRIPT_DIR/experiments/random_two_hop/train_two_hop_residual.py" \
  --headless \
  --num_envs "$NUM_ENVS" \
  --teacher_checkpoint "$TEACHER" \
  --eval_checkpoint "$RUN_DIR/model_0_zero_residual.pt" \
  --eval_steps "$EVAL_STEPS" \
  --target_height 1.0 \
  --short_radius_max 0.10 \
  --long_radius_max 0.10 \
  --curriculum_radius_max 0.02 \
  --max_turn_angle 15 \
  --curriculum_iterations 40 \
  --start_from_random_drop \
  --initial_drop_height_min 1.42 \
  --initial_drop_height_max 2.12 \
  --collective_scale 0.02 \
  --attitude_scale 0.035 \
  --residual_slew_rate 0.008 \
  --residual_penalty_scale 35 \
  --slew_penalty_scale 10 \
  --next_tilt_deg 0.0 \
  --init_noise_std 0.03
