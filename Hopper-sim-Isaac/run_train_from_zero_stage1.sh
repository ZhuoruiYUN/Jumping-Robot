#!/bin/bash
# 从零训练 Stage 1（先学会跳）：标称动力学，无随机化。
#   - 修正物理已生效（I_ZZ/K_tau/延迟）
#   - --safety_finetune 分支把 randomize_dynamics/randomize_action_delay 关掉
#     → tau 固定 0.125、质量/惯量 1.0、零延迟 —— 最容易的动力学
#   - far 契约 0-0.3/0.3-0.6 + 距离课程 100 迭代（0.3-0.5 起步）
#   - lr 3e-4（从零探索 std 0.08 默认），600 迭代，每 10 迭代存点
#   验收：stage1 结束时 target_hit_rate 应明显 >0（训练曲线），
#   或者 direct eval (tm0.125, thr1.0) hit ≥ 0.4 再进 stage2。
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
  --safety_finetune \
  --fine_tune_lr 3.0e-4 \
  --iterations 600 \
  "${@:2}"
