#!/bin/bash
# m79 sim2real 聚焦重训 v2（对照第一轮 15:29 run 的四项改动全部回滚/修正）：
#   v1 失败原因：yaww 1.5/attw 1.2/angw 1.3 权重放大 + 推力同时随机化 + lr 1e-5 太小。
#   v2：血统原始权重（attw 0.8 / angw 0.8 / yaww 1.2 / heightw 1.2 / air 0.5 / blend 1.0）
#       + 只开 tm 随机化 0.03-0.06（关推力）+ lr 5e-5 + 400 迭代。
#   契约：原地 0/0（部署契约，与 v1 一致）。验收：direct eval 在 tm 0.125/0.06/0.03 三点，
#   选型只看 direct eval（每 10 迭代存点）。
set -euo pipefail

NUM_ENVS="${1:-256}"
ISAAC_SIM_ROOT="${ISAAC_SIM_ROOT:-/home/terry/Documents/isaacsim/isaac-sim-standalone-5.1.0-linux-x86_64}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

SOURCE_CHECKPOINT="$SCRIPT_DIR/logs/rsl_rl/quadhopper_planner_random_two_hop_v60_custom_relative_next_random_drop_safety_turn_180_tol_10_spd_030_tilt_06_arw_100__ptakeoff_030_080_setup_012_000lcg_000_attw_080_angw_080_yaww_120_heightw_120_blend_100_air_050_fixed_100/2026-09-03_11-19-47/model_79.pt"

"$ISAAC_SIM_ROOT/python.sh" "$SCRIPT_DIR/train_planner_circular.py" \
  --headless \
  --num_envs "$NUM_ENVS" \
  --checkpoint "$SOURCE_CHECKPOINT" \
  --route random_two_hop \
  --distance_stage custom \
  --short_radius_min 0.00 \
  --short_radius_max 0.00 \
  --long_radius_min 0.00 \
  --long_radius_max 0.00 \
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
  --disturbance_resample_time 0.20 \
  --iterations 400 \
  "${@:2}"
