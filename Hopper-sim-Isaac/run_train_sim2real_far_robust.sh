#!/bin/bash
set -euo pipefail

NUM_ENVS="${1:-256}"
ISAAC_SIM_ROOT="${ISAAC_SIM_ROOT:-/home/terry/Documents/isaacsim/isaac-sim-standalone-5.1.0-linux-x86_64}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

SOURCE_CHECKPOINT="$SCRIPT_DIR/logs/rsl_rl/quadhopper_planner_random_two_hop_v60_custom_relative_next_random_drop_safety_turn_180_tol_10_spd_030_tilt_06_arw_100_lcg_000_attw_120_angw_130_yaww_150_air_100_thr_090_110_tshape_095_105_fixed_100/2026-09-02_15-05-45/model_79.pt"

"$ISAAC_SIM_ROOT/python.sh" "$SCRIPT_DIR/train_planner_circular.py" \
  --headless \
  --num_envs "$NUM_ENVS" \
  --checkpoint "$SOURCE_CHECKPOINT" \
  --route random_two_hop \
  --distance_stage custom \
  --short_radius_min 0.00 \
  --short_radius_max 0.30 \
  --long_radius_min 0.30 \
  --long_radius_max 0.60 \
  --height_stage high \
  --relative_next_hop \
  --start_from_random_drop \
  --safety_finetune \
  --thrust_rand_min 0.90 \
  --thrust_rand_max 1.10 \
  --thrust_shape_rand_min 0.95 \
  --thrust_shape_rand_max 1.05 \
  --fixed_action_std 0.001 \
  --airborne_stability_scale 1.0 \
  --safety_yaw_scale 1.5 \
  --safety_attitude_scale 1.2 \
  --safety_angular_vel_scale 1.3 \
  --disturbance_resample_time 0.20 \
  "${@:2}"
