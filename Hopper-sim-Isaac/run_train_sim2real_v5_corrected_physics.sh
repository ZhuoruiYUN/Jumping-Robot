#!/bin/bash
# v5 修正物理参数后的重训：
#   已修正（2026-09-05，对照实机辨识/LQR 部署工程）：
#     - I_ZZ: 2.306e-3 → 7.932e-4（Quadhopper_Stable/model/OriginJumpHopperAsset.usda 覆盖层）
#     - K_tau: 5.4e-2 → 1.72e-2（quadhopper_env.py）
#     - 动作延迟: 固定 3 步 → 0 步为主（play_action_delay=0，随机化分支 89.3%/7.5%/3.2%）
#   训练配方：
#     - far 契约 0-0.3/0.3-0.6（追点闭环）
#     - 电机滞后随机化 U[0.08, 0.18]（实测值 0.13 附近，LQR 工程同款区间）
#     - 出生 roll/pitch ±10°、yaw ±180°（LQR 工程同款随机出生）
#     - 血统原始权重 + std 0.001 + lr 5e-5 + 400 迭代（v2 成功配方）
#   验收：direct eval 三点 tm 0.125（主）/0.06/0.03（鲁棒性），far 契约，scale 0.80。
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
  --planner_reference_blend 1.0 \
  --motor_time_constant_min 0.080 \
  --motor_time_constant_max 0.180 \
  --initial_roll_pitch_perturb_deg 10.0 \
  --initial_yaw_perturb_deg 180.0 \
  --fixed_action_std 0.001 \
  --fine_tune_lr 5.0e-5 \
  --airborne_stability_scale 0.50 \
  --safety_yaw_scale 1.2 \
  --safety_attitude_scale 0.8 \
  --safety_angular_vel_scale 0.8 \
  --safety_height_scale 1.2 \
  --iterations 400 \
  "${@:2}"
