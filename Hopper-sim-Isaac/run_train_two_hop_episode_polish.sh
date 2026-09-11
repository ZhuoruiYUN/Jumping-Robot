#!/usr/bin/env bash
# Polish direct landing control with exactly two touchdowns per episode.
set -euo pipefail

NUM_ENVS="${1:-256}"
ITERATIONS="${2:-60}"
ISAAC_SIM_ROOT="${ISAAC_SIM_ROOT:-/home/terry/Documents/isaacsim/isaac-sim-standalone-5.1.0-linux-x86_64}"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
SOURCE_RUN="$SCRIPT_DIR/logs/rsl_rl/quadhopper_planner_random_two_hop_v67_custom_relative_next_baseline_warmstart_smooth_distance_turn_030_tol_15_spd_030_tilt_06_arw_100_lcg_000_blend_000_landingxy_pairrestart_air_100_yawair_spring_fixed_100/2026-09-07_19-53-26"
CHECKPOINT="${3:-$SOURCE_RUN/model_100.pt}"

cd "$SCRIPT_DIR"

"$ISAAC_SIM_ROOT/python.sh" train_planner_circular.py \
  --headless \
  --num_envs "$NUM_ENVS" \
  --iterations "$ITERATIONS" \
  --checkpoint "$CHECKPOINT" \
  --route random_two_hop \
  --distance_stage custom \
  --short_radius_min 0.05 \
  --short_radius_max 0.15 \
  --long_radius_min 0.05 \
  --long_radius_max 0.15 \
  --max_turn_angle 30 \
  --height_stage high \
  --relative_next_hop \
  --two_hop_episode \
  --baseline_warmstart_finetune \
  --baseline_compatible_stance_reference \
  --planner_reference_blend 0.0 \
  --direct_landing_reference \
  --target_tolerance 0.15 \
  --fine_tune_lr 2e-6 \
  --fixed_action_std 0.02 \
  "${@:4}"
