#!/bin/bash
# m79 sim2real 微调选型：候选 checkpoint x tm 三点（0.125 标称 / 0.06 / 0.03 实机近似）
# 部署契约 + action_scale 0.85 + 256 env + 1700 步（V59 direct eval 协议）
set -euo pipefail

ISAAC_SIM_ROOT="${ISAAC_SIM_ROOT:-/home/terry/Documents/isaacsim/isaac-sim-standalone-5.1.0-linux-x86_64}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
OUT_DIR="$PROJECT_DIR/outputs/experiments/m79_deploy_contract"
EXP="quadhopper_planner_random_two_hop_v60_custom_relative_next_random_drop_safety_turn_180_tol_10_spd_030_tilt_06_arw_100_lcg_000_attw_120_angw_130_yaww_150_air_100_thr_095_135_tm_030_060_fixed_100"
M79="$PROJECT_DIR/logs/rsl_rl/quadhopper_planner_random_two_hop_v60_custom_relative_next_random_drop_safety_turn_180_tol_10_spd_030_tilt_06_arw_100__ptakeoff_030_080_setup_012_000lcg_000_attw_080_angw_080_yaww_120_heightw_120_blend_100_air_050_fixed_100/2026-09-03_11-19-47/model_79.pt"

eval_case() {
  local name="$1"; local ckpt="$2"; local tm="$3"
  echo "==== [$(date +%T)] EVAL $name tm=$tm ===="
  "$ISAAC_SIM_ROOT/python.sh" "$PROJECT_DIR/play_planner_circular.py" \
    --checkpoint "$ckpt" \
    --route random_two_hop \
    --distance_stage custom \
    --short_radius_min 0.00 --short_radius_max 0.00 \
    --long_radius_min 0.00 --long_radius_max 0.00 \
    --height_stage high --height_high 1.0 \
    --relative_next_hop --start_from_random_drop \
    --planner_reference_blend 1.0 \
    --num_envs 256 --max_steps 1700 \
    --action_scale 0.85 \
    --play_motor_time_constant "$tm" \
    --no_debug_vis --headless \
    2>&1 | tee "$OUT_DIR/eval_${name}_tm${tm}.log" | \
    grep -E "EVAL-DIRECT\] (target_hit_rate|short_target_hit_rate|long_target_hit_rate|episode_touchdown_error_m|touchdown_attitude_error_rad|last_apex_height_m|max_consecutive_hits)|EVAL-DEPLOY\] (death_rate|deploy_success_rate|tilt_death_events)" | \
    sed "s/^/[${name}_tm${tm}] /"
}

for tm in 0.125 0.060 0.030; do
  eval_case "m79base" "$M79" "$tm"
done

FT_USER="$PROJECT_DIR/logs/rsl_rl/$EXP/2026-09-03_15-29-13"
for tm in 0.125 0.060 0.030; do
  eval_case "ft_u50" "$FT_USER/model_50.pt" "$tm"
  eval_case "ft_u99" "$FT_USER/model_99.pt" "$tm"
done

FT_LEFT="$PROJECT_DIR/logs/rsl_rl/$EXP/2026-09-03_15-20-53"
for tm in 0.125 0.060 0.030; do
  eval_case "ft_L95" "$FT_LEFT/model_95.pt" "$tm"
done

echo "==== ALL EVALS DONE ===="
