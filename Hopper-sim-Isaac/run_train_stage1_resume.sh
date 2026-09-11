#!/bin/bash
# Stage 1 续训：从 10:42 从零训练（默认温和随机化）的 model_350 精确恢复。
#   背景（2026-09-06 评估结论）：
#   - 10:42 那轮从零训练（tau[0.1,0.14]、质量±5%、无推力随机化）
#     在被提前终止前正在正常学习：it 300 时 apex 已 1.04m、reward -24→+76，
#     只是 hit 还没冒头（it 359 时仍 0）。
#   - 11:39 realop 宽随机化（tau[0.03,0.125]+推力[1.15,1.45]）从零 2000 迭代
#     完全学不会（apex≤0.38、hit 全 0）→ 宽随机化必须从"会跳的"策略起步。
#   - 14:52 stage1（--safety_finetune）失败是脚本 bug：safety 分支在
#     --fine_tune_lr 之后覆盖 lr=2e-6/std=0.01，从零训练被冻住。勿再用
#     --safety_finetune 当"关随机化"开关。
#
#   本脚本：与 10:42 完全相同的 flags（EXPERIMENT 一致）→ --resume_optimizer
#   精确恢复 optimizer + iter=350，续到 it 2000。
#   验收：target_hit_rate 明显爬升（>0.3）且 apex 稳定 ~1.0 → 进 Stage 2
#   （从本 checkpoint 续训，随机化扩宽到 tau U[0.03,0.125] + 推力 U[1.15,1.45]）。
set -euo pipefail

NUM_ENVS="${1:-256}"
ISAAC_SIM_ROOT="${ISAAC_SIM_ROOT:-/home/terry/Documents/isaacsim/isaac-sim-standalone-5.1.0-linux-x86_64}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

CKPT="${2:-$SCRIPT_DIR/logs/rsl_rl/quadhopper_planner_random_two_hop_v59_custom_relative_next_random_drop_turn_180_tol_10_spd_030_tilt_06_arw_100_lcg_000_blend_100_fixed_100/2026-09-06_10-42-27/model_350.pt}"

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
  --fine_tune_lr 3.0e-4 \
  --iterations 2000 \
  --checkpoint "$CKPT" \
  --resume_optimizer \
  "${@:3}"
