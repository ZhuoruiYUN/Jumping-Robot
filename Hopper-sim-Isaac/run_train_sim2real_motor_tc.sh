#!/bin/bash
# m79 sim2real 微调：电机时间常数随机化（实机疑似 ~0.03s，sim 假设 0.125s）
#
# 依据：2026-09-03 sim 对照实验（outputs/experiments/m79_deploy_contract/SUMMARY.md）
#   - 部署契约（双跳同点）本身无发散（54 跳 hit 0.981, death 0）
#   - tm=0.03 复现实机发散前幅度（apex tilt 16-29°、每跳 yaw 8.9-10.7°）
#   - 推力失配（均匀/逐电机）只降精度，不是快环失稳主因
#
# 本脚本：从 far_030_060 m79 起，原地落点分布（0/0）+ 1.0m，
# 覆盖 tm 0.03-0.06s + 推力 0.95-1.35，低 lr + 低 std 防塌。
# 验收：跑完用 play_planner_circular.py 做 direct eval，并在
# --play_motor_time_constant 0.125/0.06/0.03 三点对照（不能只认标称）。
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
  --motor_time_constant_min 0.030 \
  --motor_time_constant_max 0.060 \
  --thrust_rand_min 0.95 \
  --thrust_rand_max 1.35 \
  --fixed_action_std 0.001 \
  --fine_tune_lr 1.0e-5 \
  --airborne_stability_scale 1.0 \
  --safety_yaw_scale 1.5 \
  --safety_attitude_scale 1.2 \
  --safety_angular_vel_scale 1.3 \
  --iterations 100 \
  "${@:2}"
