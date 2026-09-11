#!/bin/bash
# V2 short -> long transition bridge. V1 jumped directly from a reset-state
# long hop to a 0.30--0.435 m second hop and produced zero pair hits. Start the
# second phase near stationary, then expand it slowly while every episode
# retains the deployment order short -> long.
set -euo pipefail

NUM_ENVS="${1:-1024}"
ISAAC_SIM_ROOT="${ISAAC_SIM_ROOT:-/home/terry/Documents/isaacsim/isaac-sim-standalone-5.1.0-linux-x86_64}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# Deliberately branch from the successful isolated-long model, never from the
# failed V1 transition checkpoints.
CKPT="${2:-$SCRIPT_DIR/logs/rsl_rl/quadhopper_planner_random_two_hop_v59_custom_relative_next_random_drop_turn_180_tol_10_spd_030_tilt_06_arw_100_lcg_000_blend_100_lax_120_lenapex_longcur_038_400_tds_075_av_030_fixed_100/2026-09-07_11-27-53/model_99.pt}"

"$ISAAC_SIM_ROOT/python.sh" "$SCRIPT_DIR/train_planner_circular.py" \
  --headless \
  --num_envs "$NUM_ENVS" \
  --route random_two_hop \
  --distance_stage custom \
  --short_radius_min 0.00 \
  --short_radius_max 0.30 \
  --long_radius_min 0.30 \
  --long_radius_max 0.435 \
  --max_turn_angle 67 \
  --long_hop_curriculum \
  --long_curriculum_radius_min 0.05 \
  --long_curriculum_radius_max 0.15 \
  --long_curriculum_turn_deg 15 \
  --distance_curriculum_iterations 400 \
  --two_hop_transition_curriculum \
  --height_stage high \
  --relative_next_hop \
  --start_from_random_drop \
  --planner_reference_blend 1.0 \
  --fine_tune_lr 3.0e-5 \
  --fixed_action_std 0.06 \
  --low_apex_event_penalty 120 \
  --lenient_apex_hit \
  --tilt_death_sq_threshold 0.75 \
  --terminate_angvel_norm 30 \
  --iterations 75 \
  --checkpoint "$CKPT" \
  "${@:3}"
