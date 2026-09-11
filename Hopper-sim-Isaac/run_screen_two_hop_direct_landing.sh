#!/usr/bin/env bash
# Deterministically screen several checkpoints from the direct-landing run.
set -euo pipefail

ISAAC_SIM_ROOT="${ISAAC_SIM_ROOT:-/home/terry/Documents/isaacsim/isaac-sim-standalone-5.1.0-linux-x86_64}"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
RUN_DIR="${1:-$SCRIPT_DIR/logs/rsl_rl/quadhopper_planner_random_two_hop_v67_custom_relative_next_baseline_warmstart_turn_030_tol_15_spd_030_tilt_06_arw_100_lcg_000_blend_000_landingxy_pairrestart_air_100_yawair_spring_fixed_100/2026-09-07_21-56-53}"
if (( $# > 0 )); then
  shift
fi
CHECKPOINT_IDS=("$@")
if (( ${#CHECKPOINT_IDS[@]} == 0 )); then
  CHECKPOINT_IDS=(0 20 40 59)
fi
SHORT_MIN="${SHORT_MIN:-0.05}"
SHORT_MAX="${SHORT_MAX:-0.15}"
LONG_MIN="${LONG_MIN:-$SHORT_MIN}"
LONG_MAX="${LONG_MAX:-$SHORT_MAX}"

cd "$SCRIPT_DIR"

for ITERATION in "${CHECKPOINT_IDS[@]}"; do
  CHECKPOINT="$RUN_DIR/model_${ITERATION}.pt"
  if [[ ! -f "$CHECKPOINT" ]]; then
    echo "[SCREEN] missing checkpoint: $CHECKPOINT" >&2
    continue
  fi
  echo "[SCREEN] model_${ITERATION}.pt"
  "$ISAAC_SIM_ROOT/python.sh" play_planner_circular.py \
    --headless \
    --no_debug_vis \
    --num_envs 256 \
    --max_steps 1500 \
    --seed 42 \
    --checkpoint "$CHECKPOINT" \
    --route random_two_hop \
    --distance_stage custom \
    --short_radius_min "$SHORT_MIN" \
    --short_radius_max "$SHORT_MAX" \
    --long_radius_min "$LONG_MIN" \
    --long_radius_max "$LONG_MAX" \
    --max_turn_angle 30 \
    --height_stage high \
    --height_high 1.0 \
    --target_tolerance 0.15 \
    --planner_reference_blend 0.0 \
    --direct_landing_reference \
    --baseline_compatible_stance_reference \
    --relative_next_hop \
    --pair_restart_queue \
    --lenient_apex_hit \
    --play_motor_time_constant 0.0674 \
    --compact_eval
done
