#!/bin/bash
set -euo pipefail

NUM_ENVS="${1:-256}"
ISAAC_SIM_ROOT="${ISAAC_SIM_ROOT:-/home/terry/Documents/isaacsim/isaac-sim-standalone-5.1.0-linux-x86_64}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

SOURCE_CHECKPOINT="$SCRIPT_DIR/logs/rsl_rl/quadhopper_planner_random_two_hop_v60_custom_relative_next_random_drop_safety_turn_180_tol_10_spd_030_tilt_06_arw_100__ptakeoff_030_080_setup_012_000lcg_000_attw_080_angw_080_yaww_120_heightw_120_blend_100_air_050_fixed_100/2026-09-03_11-19-47/model_79.pt"

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
  --attitude_damping_polish \
  --planner_reference_blend 1.0 \
  --fixed_action_std 0.0012 \
  --airborne_stability_scale 0.45 \
  --safety_yaw_scale 1.1 \
  --safety_attitude_scale 0.75 \
  --safety_angular_vel_scale 0.75 \
  --safety_height_scale 1.15 \
  --damping_airborne_angvel_start 1.05 \
  --damping_airborne_angvel_scale 2.2 \
  --damping_action_spread_scale 24.0 \
  --damping_action_rate_scale 1.8 \
  --damping_yaw_rate_start 0.80 \
  --damping_yaw_rate_scale 0.18 \
  --disturbance_resample_time 0.20 \
  "${@:2}"
