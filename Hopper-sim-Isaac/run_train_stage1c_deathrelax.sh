#!/bin/bash
# Stage 1c：从 lenapex 轮 model_790 续训，放宽死亡判定，让策略练会 long 跳。
#   背景（lenapex 轮 it 799 诊断）：
#   - overall hit 0.90、short hit 0.90 —— far 契约 short 跳基本解决；
#   - long 跳命中在最后几十迭代开始出现（EMA 尖峰 = 100% 命中窗口），
#     但 long 跳只占落地的 ~3%：策略在 long 跳中段就翻倒/漂移死亡
#     （tilt>45° / angvel>18 / 离目标>2m），没有 touchdown 就没有学习信号，
#     恶性循环。
#   本脚本：--terminate_far_xy 6.0 + --terminate_angvel_norm 30.0 放宽死亡，
#   让 long 跳能落地、吃到 miss/hit 信号。tilt 45° 门槛不动（真翻了就该死）。
#   EXPERIMENT 带 _fxy_006_av_030 token → transfer 路径（optimizer 重置）。
#   验收：long_target_hit_rate 累计值持续爬升（不再是 0.0000/0.0039 闪烁），
#   且 episode 平均长度明显变长（>300 步，说明活过了第二个跳）。
set -euo pipefail

NUM_ENVS="${1:-1024}"
ISAAC_SIM_ROOT="${ISAAC_SIM_ROOT:-/home/terry/Documents/isaacsim/isaac-sim-standalone-5.1.0-linux-x86_64}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

CKPT="${2:-$SCRIPT_DIR/logs/rsl_rl/quadhopper_planner_random_two_hop_v59_custom_relative_next_random_drop_turn_180_tol_10_spd_030_tilt_06_arw_100_lcg_000_blend_100_lax_120_lenapex_fixed_100/2026-09-06_23-01-59/model_790.pt}"

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
  --terminate_far_xy 6.0 \
  --terminate_angvel_norm 30.0 \
  --iterations 800 \
  --checkpoint "$CKPT" \
  "${@:3}"
