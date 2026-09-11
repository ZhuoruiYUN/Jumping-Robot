#!/bin/bash
# 从零训练 v2（真实工作点版）：
#   2026-09-06 实机验证飞行（11:04/11:08）证明：修正物理 + tm 0.03 + 推力 1.3x
#   三个条件同时成立时，sim 逐跳复现实机（apex tilt 20-24° vs 实机 25-28°、
#   yaw 摆动 14-24°/跳、td_err 0.25-0.30 vs 实机 0.2-0.8、apex 过冲 1.34-1.47）。
#   故真实工作点 = 电机 tau ≈ 0.03 + 推力 ≈ 1.3x sim 曲线。
#
#   本脚本：从零训练，随机化覆盖真实工作点全不确定带：
#     - 电机滞后 U[0.03, 0.125]（实机行为等效 ~0.03-0.06 vs 标定 0.125，两头下注）
#     - 推力 U[1.15, 1.45]（实机 ~1.3）
#     - 其余基础随机化（质量/惯量/延迟）
#     - far 契约 0-0.3/0.3-0.6 + 距离课程
#   lr 3e-4、2000 迭代、每 25 迭代存点。
#   验收：direct eval 在 (tm 0.03, thr 1.3) 主点 + (tm 0.06, thr 1.3) + (tm 0.125, thr 1.0)
#   三点对照，逐跳指标与实机日志同口径对比。
set -euo pipefail

NUM_ENVS="${1:-256}"
ISAAC_SIM_ROOT="${ISAAC_SIM_ROOT:-/home/terry/Documents/isaacsim/isaac-sim-standalone-5.1.0-linux-x86_64}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

"$ISAAC_SIM_ROOT/python.sh" "$SCRIPT_DIR/train_planner_circular.py" \
  --headless \
  --num_envs "$NUM_ENVS" \
  --route random_two_hop \
  --distance_stage custom \
  --short_radius_min 0.00 \
  --short_radius_max 0.30 \
  --long_radius_min 0.30 \
  --long_radius_max 0.60 \
  --height_stage high \
  --relative_next_hop \
  --start_from_random_drop \
  --planner_reference_blend 1.0 \
  --motor_time_constant_min 0.030 \
  --motor_time_constant_max 0.125 \
  --thrust_rand_min 1.15 \
  --thrust_rand_max 1.45 \
  --fine_tune_lr 3.0e-4 \
  --iterations 2000 \
  "${@:2}"
