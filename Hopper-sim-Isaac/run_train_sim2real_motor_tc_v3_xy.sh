#!/bin/bash
# v3 落点精度微调：从 v2_240（v2 峰点）起，v2 配方全部不变，
#   加 --safety_xy_scale 1.3（触地落点奖励/惩罚）+ --safety_xy_dense_scale 1.3（空中稠密 XY）。
#   200 迭代。验收（三选一否决）：td_err @ tm0.03/0.06 下降为主指标；
#   护栏：逐跳 yaw 摆动 ≤8°、apex tilt ≤20°、hit@tm0.125 ≥0.25。
set -euo pipefail

NUM_ENVS="${1:-256}"
ISAAC_SIM_ROOT="${ISAAC_SIM_ROOT:-/home/terry/Documents/isaacsim/isaac-sim-standalone-5.1.0-linux-x86_64}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

SOURCE_CHECKPOINT="$SCRIPT_DIR/logs/rsl_rl/quadhopper_planner_random_two_hop_v60_custom_relative_next_random_drop_safety_turn_180_tol_10_spd_030_tilt_06_arw_100_lcg_000_attw_080_angw_080_yaww_120_heightw_120_blend_100_air_050_tm_030_060_fixed_100/2026-09-03_15-54-06/model_240.pt"

"$ISAAC_SIM_ROOT/python.sh" "$SCRIPT_DIR/train_planner_circular.py" \
  --headless \
  --num_envs "$NUM_ENVS" \
  --checkpoint "$SOURCE_CHECKPOINT" \
  --route random_two_hop \
  --distance_stage custom \
  --short_radius_min 0.00 \
  --short_radius_max 0.00 \
  --long_radius_min 0.00 \
  --long_radius_max 0.00 \
  --height_stage high \
  --relative_next_hop \
  --start_from_random_drop \
  --safety_finetune \
  --planner_reference_blend 1.0 \
  --motor_time_constant_min 0.030 \
  --motor_time_constant_max 0.060 \
  --fixed_action_std 0.001 \
  --fine_tune_lr 5.0e-5 \
  --airborne_stability_scale 0.50 \
  --safety_yaw_scale 1.2 \
  --safety_attitude_scale 0.8 \
  --safety_angular_vel_scale 0.8 \
  --safety_height_scale 1.2 \
  --disturbance_resample_time 0.20 \
  --safety_xy_scale 1.3 \
  --safety_xy_dense_scale 1.3 \
  --iterations 200 \
  "${@:2}"
