#!/bin/bash
# Stage 1: continuous robot dynamics, but restart the command pair around the
# measured touchdown after each short+long pair to prevent route-error drift.
set -euo pipefail

NUM_ENVS="${1:-256}"
ITERATIONS="${2:-120}"
ISAAC_SIM_ROOT="${ISAAC_SIM_ROOT:-/home/terry/Documents/isaacsim/isaac-sim-standalone-5.1.0-linux-x86_64}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
TEACHER="${3:-$SCRIPT_DIR/logs/rsl_rl/quadhopper_stable_calibrated_accuracy_20260907/2026-09-07_18-01-15/model_598.pt}"

"$ISAAC_SIM_ROOT/python.sh" "$SCRIPT_DIR/experiments/random_two_hop/train_two_hop_residual.py" \
  --headless \
  --num_envs "$NUM_ENVS" \
  --iterations "$ITERATIONS" \
  --teacher_checkpoint "$TEACHER" \
  --pair_restart_queue \
  --gentle_pair_curriculum \
  --target_height 1.0 \
  --target_tolerance 0.15 \
  --short_radius_min 0.12 \
  --short_radius_max 0.20 \
  --long_radius_min 0.18 \
  --long_radius_max 0.28 \
  --curriculum_radius_min 0.12 \
  --curriculum_radius_max 0.18 \
  --max_turn_angle 30 \
  --curriculum_iterations 80 \
  --start_from_random_drop \
  --initial_drop_height_min 1.42 \
  --initial_drop_height_max 2.12 \
  --collective_scale 0.015 \
  --attitude_scale 0.040 \
  --residual_slew_rate 0.008 \
  --residual_penalty_scale 3 \
  --slew_penalty_scale 1 \
  --next_tilt_deg 0 \
  --init_noise_std 0.04 \
  --save_interval 10
