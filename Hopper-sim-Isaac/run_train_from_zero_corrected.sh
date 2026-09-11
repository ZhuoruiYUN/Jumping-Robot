#!/bin/bash
# 从零训练（修正物理后，raw-PWM 血统，不用 LQR）：
#   - 修正后的物理（I_ZZ 7.932e-4 / K_tau 1.72e-2 / 延迟 0 主导）已生效
#   - random_two_hop far 契约 0-0.3 / 0.3-0.6（追点任务）
#   - 基础动力学随机化（randomize_dynamics 默认开：质量/惯量/tau[0.1,0.14]/延迟）
#   - 距离课程 100 迭代（0.3-0.5 起步 → 0-0.3/0.3-0.6）
#   - lr 3e-4（--fine_tune_lr 只是复用了 lr 覆盖口，非微调）
#   - 2000 迭代（256 env × ~2.8s ≈ 90 分钟），每 25 迭代存点
#   验收：direct eval 三点 tm 0.125/0.06/0.03（far 契约 + 0/0 契约都测），
#   目标超过 m79 修正物理基线（far: 0.578/0.450/0.295；0/0: 见 SUMMARY §9）。
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
  --fine_tune_lr 3.0e-4 \
  --iterations 2000 \
  "${@:2}"
