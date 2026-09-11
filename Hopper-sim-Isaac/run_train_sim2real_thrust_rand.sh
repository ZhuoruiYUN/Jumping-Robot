#!/bin/bash
set -euo pipefail

NUM_ENVS="${1:-256}"
ISAAC_SIM_ROOT="${ISAAC_SIM_ROOT:-/home/terry/Documents/isaacsim/isaac-sim-standalone-5.1.0-linux-x86_64}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

AIR050_CHECKPOINT="$SCRIPT_DIR/logs/rsl_rl/quadhopper_planner_random_two_hop_v60_custom_relative_next_random_drop_safety_turn_180_tol_10_spd_030_tilt_06_arw_100_lcg_000_attw_115_angw_125_xyw_110_yaww_120_heightw_125_air_050_fixed_100/2026-08-30_14-16-23/model_79.pt"

"$ISAAC_SIM_ROOT/python.sh" "$SCRIPT_DIR/train_planner_circular.py" \
  --headless \
  --num_envs "$NUM_ENVS" \
  --checkpoint "$AIR050_CHECKPOINT" \
  --route random_two_hop \
  --distance_stage custom \
  --short_radius_min 0.00 \
  --short_radius_max 0.08 \
  --long_radius_min 0.00 \
  --long_radius_max 0.08 \
  --height_stage high \
  --relative_next_hop \
  --start_from_random_drop \
  --safety_finetune \
  --thrust_rand_min 0.85 \
  --thrust_rand_max 1.15 \
  --thrust_shape_rand_min 0.90 \
  --thrust_shape_rand_max 1.10 \
  --fixed_action_std 0.003 \
  --airborne_stability_scale 1.0 \
  --safety_yaw_scale 1.5 \
  --safety_attitude_scale 1.2 \
  --safety_angular_vel_scale 1.3 \
  "${@:2}"
