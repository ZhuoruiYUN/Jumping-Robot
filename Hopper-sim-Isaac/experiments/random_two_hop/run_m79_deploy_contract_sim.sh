#!/bin/bash
# 复现实机部署契约的 sim 对照实验（m79 far_030_060 model_79）
#
#   A: 部署契约    short=long=0.0（双跳同一落点，pair context 交替） action_scale 1.0
#   B: 标称契约    short 0.0-0.3 / long 0.3-0.6（V59 direct eval 配置）  action_scale 1.0
#   C: 部署契约 + 部署动作缩放 action_scale 0.85
#
# 每轮 1 env + CSV 日志（Time/位姿/四元数/PWM/接触），跑 6000 步(100Hz, 60s sim ≈ 45+ 跳)
# 输出: outputs/experiments/m79_deploy_contract/<name>.csv + stdout 日志
set -euo pipefail

ISAAC_SIM_ROOT="${ISAAC_SIM_ROOT:-/home/terry/Documents/isaacsim/isaac-sim-standalone-5.1.0-linux-x86_64}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
OUT_DIR="$PROJECT_DIR/outputs/experiments/m79_deploy_contract"
CKPT="$PROJECT_DIR/logs/rsl_rl/quadhopper_planner_random_two_hop_v60_custom_relative_next_random_drop_safety_turn_180_tol_10_spd_030_tilt_06_arw_100__ptakeoff_030_080_setup_012_000lcg_000_attw_080_angw_080_yaww_120_heightw_120_blend_100_air_050_fixed_100/2026-09-03_11-19-47/model_79.pt"

COMMON=(
  "$PROJECT_DIR/play_planner_circular.py"
  --checkpoint "$CKPT"
  --route random_two_hop
  --distance_stage custom
  --height_stage high
  --height_high 1.0
  --relative_next_hop
  --start_from_random_drop
  --planner_reference_blend 1.0
  --num_envs 1
  --max_steps 6000
  --no_debug_vis
  --headless
)

run_case() {
  local name="$1"; shift
  echo "==== [$(date +%T)] RUN $name ===="
  "$ISAAC_SIM_ROOT/python.sh" "$@" 2>&1 | tee "$OUT_DIR/$name.log"
  cp "$PROJECT_DIR/outputs/planner_random_two_hop/on_quadhopper_sim.csv" "$OUT_DIR/$name.csv"
  echo "==== [$(date +%T)] DONE $name -> $OUT_DIR/$name.csv ===="
}

run_case A_deploy_contract \
  "${COMMON[@]}" \
  --short_radius_min 0.00 --short_radius_max 0.00 \
  --long_radius_min 0.00 --long_radius_max 0.00

run_case B_nominal_contract \
  "${COMMON[@]}" \
  --short_radius_min 0.00 --short_radius_max 0.30 \
  --long_radius_min 0.30 --long_radius_max 0.60

run_case C_deploy_contract_scale085 \
  "${COMMON[@]}" \
  --short_radius_min 0.00 --short_radius_max 0.00 \
  --long_radius_min 0.00 --long_radius_max 0.00 \
  --action_scale 0.85

echo "ALL DONE"
