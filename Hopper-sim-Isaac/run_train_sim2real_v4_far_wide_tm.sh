#!/bin/bash
# v4 稳健化重训（raw-PWM 血统内，不用 LQR）：
#   目标：同时解决"稳"（实机快环失配）和"偏"（XY 追点弱）。
#   做法（四维一起）：
#     1. far 契约 0-0.3/0.3-0.6：每跳主动闭 XY 误差（追点 = 日常任务，出生即偏移 0-0.3m）
#     2. tm 随机化加宽 U[0.03, 0.125]：覆盖旧标定 0.125 到实测等效 0.03 的整个不确定带
#     3. 逐电机推力不对称 U[0.90, 1.10]：覆盖电机个体差异，不假设具体偏置方向
#     4. 出生 yaw 扰动 ±30°：练 yaw 偏移恢复
#   其余保持 v2 成功配方：血统原始权重 + std 0.001 + lr 5e-5 + 400 迭代。
#   验收：direct eval 在 tm 0.03/0.06/0.125 三点（far 契约），主指标 hit/td_err，
#   护栏：yaw 摆动、apex tilt、death。
set -euo pipefail

NUM_ENVS="${1:-256}"
ISAAC_SIM_ROOT="${ISAAC_SIM_ROOT:-/home/terry/Documents/isaacsim/isaac-sim-standalone-5.1.0-linux-x86_64}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# 从 v2_240 起（当前最优：快电机下最稳，far 追点能力为残留水平，本轮到 far 契约补回）
SOURCE_CHECKPOINT="$SCRIPT_DIR/logs/rsl_rl/quadhopper_planner_random_two_hop_v60_custom_relative_next_random_drop_safety_turn_180_tol_10_spd_030_tilt_06_arw_100_lcg_000_attw_080_angw_080_yaww_120_heightw_120_blend_100_air_050_tm_030_060_fixed_100/2026-09-03_15-54-06/model_240.pt"

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
  --motor_time_constant_min 0.030 \
  --motor_time_constant_max 0.125 \
  --thrust_rand_min 0.90 \
  --thrust_rand_max 1.10 \
  --initial_yaw_perturb_deg 30.0 \
  --fixed_action_std 0.001 \
  --fine_tune_lr 5.0e-5 \
  --airborne_stability_scale 0.50 \
  --safety_yaw_scale 1.2 \
  --safety_attitude_scale 0.8 \
  --safety_angular_vel_scale 0.8 \
  --safety_height_scale 1.2 \
  --iterations 400 \
  "${@:2}"
