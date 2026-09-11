#!/bin/bash
# Stage 1b：从 model_990 续训，打开 low-apex 落地惩罚，打破平跳局部最优。
#   背景（it 1011 诊断）：overall hit 0.432 但 long hit 精确为 0——
#   长跳落点误差 0.069m（< 0.10 判定线）却永不 hit，因为策略学会了
#   "低平抛物线"（apex < 0.58 最低有效高度），valid_apex 判定永远失败；
#   而 env 默认 cfg 里 low_apex 惩罚 scale = 0，没有任何梯度把长跳拉高。
#   mean apex 已从 0.90 漂到 0.54，短跳也在向 0.58 地板靠拢。
#
#   本脚本：--low_apex_event_penalty 120 给每个低 apex 落地 -120 惩罚
#   （与 trainer 里 takeoff_impulse 分支已验证的值一致），
#   + --lenient_apex_hit 关掉 apex 容差命中门（第一层锁：trainer 通用设置里
#   require_apex_tolerance_for_hit=True，long 跳 apex 天然在 ±0.12 窗口外，
#   落点再准也永远不 hit），
#   + --tilt_death_sq_threshold 0.95 放宽空中翻车判死（第二层锁：long 跳
#   起飞后 ~0.3s 翻到 90°+ 被判死，策略从没经历过一次完整 long 跳，永远学不会）。
#   + --terminate_angvel_norm 40（从 18 放宽角速度判死，同上目的）。
#   其余 flags 不变。EXPERIMENT 带 _lax_120_lenapex_tds_095 token → transfer 路径。
#   验收：long_target_hit_rate 破 0 并爬升 + long 落地次数占比回升到 ~40%。
set -euo pipefail

NUM_ENVS="${1:-1024}"
ISAAC_SIM_ROOT="${ISAAC_SIM_ROOT:-/home/terry/Documents/isaacsim/isaac-sim-standalone-5.1.0-linux-x86_64}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

CKPT="${2:-$SCRIPT_DIR/logs/rsl_rl/quadhopper_planner_random_two_hop_v59_custom_relative_next_random_drop_turn_180_tol_10_spd_030_tilt_06_arw_100_lcg_000_blend_100_lax_120_lenapex_fixed_100/2026-09-06_23-01-59/model_799.pt}"

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
  --low_apex_event_penalty 120 \
  --lenient_apex_hit \
  --tilt_death_sq_threshold 0.95 \
  --terminate_angvel_norm 40 \
  --iterations 1000 \
  --checkpoint "$CKPT" \
  "${@:3}"
