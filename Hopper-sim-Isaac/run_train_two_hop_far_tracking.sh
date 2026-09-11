#!/bin/bash
# 两跳追点（向前跳）训练：第一跳 0-0.3 m、第二跳 0.3-0.6 m，随机方向。
#
# 这就是"追点能力"的训练：每次触地后目标点前移，策略必须精确落到 P_t 再规划 P_(t+1)。
# 短跳半径下限为 0（0-0.3 含原地附近），训练分布覆盖近原地起跳；
# 因此该血统标称电机下原地跳（0/0 部署契约 hit 0.90）和向前跳（far 契约 hit 0.73）都能做。
#
# 默认配方 = m79 血统原始权重 + tm 随机化 0.03-0.06（v2 成功配方，实机快环增益匹配）。
# 若只要 m79 原始追点训练（不做 sim2real 鲁棒化）：删除 --motor_time_constant_min/max 两行。
set -euo pipefail

NUM_ENVS="${1:-256}"
ISAAC_SIM_ROOT="${ISAAC_SIM_ROOT:-/home/terry/Documents/isaacsim/isaac-sim-standalone-5.1.0-linux-x86_64}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

SOURCE_CHECKPOINT="$SCRIPT_DIR/checkpoints/m79_far_base/model_79.pt"

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
  --motor_time_constant_max 0.060 \
  --fixed_action_std 0.001 \
  --fine_tune_lr 5.0e-5 \
  --airborne_stability_scale 0.50 \
  --safety_yaw_scale 1.2 \
  --safety_attitude_scale 0.8 \
  --safety_angular_vel_scale 0.8 \
  --safety_height_scale 1.2 \
  --iterations 400 \
  "${@:2}"
