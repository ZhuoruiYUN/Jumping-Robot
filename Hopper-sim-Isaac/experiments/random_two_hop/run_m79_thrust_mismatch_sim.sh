#!/bin/bash
# 推力失配假设检验 v2（带 --allow_single_env_thrust_rand）：
# 部署契约 + action_scale 0.85 + 全局推力倍率 1.2 / 1.4 / 1.7
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
  --short_radius_min 0.00 --short_radius_max 0.00
  --long_radius_min 0.00 --long_radius_max 0.00
  --action_scale 0.85
  --allow_single_env_thrust_rand
)

run_case() {
  local name="$1"; shift
  echo "==== [$(date +%T)] RUN $name ===="
  "$ISAAC_SIM_ROOT/python.sh" "$@" 2>&1 | tee "$OUT_DIR/$name.log"
  cp "$PROJECT_DIR/outputs/planner_random_two_hop/on_quadhopper_sim.csv" "$OUT_DIR/$name.csv"
  echo "==== [$(date +%T)] DONE $name -> $OUT_DIR/$name.csv ===="
}

run_case F_thrust_120 \
  "${COMMON[@]}" \
  --thrust_rand_min 1.20 --thrust_rand_max 1.20

run_case D2_thrust_140 \
  "${COMMON[@]}" \
  --thrust_rand_min 1.40 --thrust_rand_max 1.40

run_case E2_thrust_170 \
  "${COMMON[@]}" \
  --thrust_rand_min 1.70 --thrust_rand_max 1.70

echo "ALL DONE"
