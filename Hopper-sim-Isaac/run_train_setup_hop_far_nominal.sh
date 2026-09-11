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
  --setup_hop_finetune \
  --airborne_setup_finetune \
  --planner_reference_blend 1.0 \
  --fixed_action_std 0.002 \
  --setup_velocity_target 0.12 \
  --setup_velocity_width 0.18 \
  --setup_velocity_reward_scale 0.0 \
  --setup_lateral_velocity_penalty 0.0 \
  --setup_touchdown_attitude_deadband 0.03 \
  --airborne_setup_start_phase 0.68 \
  --airborne_setup_tilt_deg 2.0 \
  --airborne_setup_reward_scale 0.3 \
  --airborne_setup_landing_gate_width 0.14 \
  --final_touchdown_attitude_penalty 60.0 \
  --final_touchdown_velocity_penalty 20.0 \
  --airborne_stability_scale 0.50 \
  --safety_yaw_scale 1.2 \
  --safety_attitude_scale 0.8 \
  --safety_angular_vel_scale 0.8 \
  --safety_height_scale 1.2 \
  --disturbance_resample_time 0.20 \
  "${@:2}"
