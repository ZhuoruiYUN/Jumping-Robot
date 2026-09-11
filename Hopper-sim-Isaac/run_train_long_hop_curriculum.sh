#!/bin/bash
# Long-hop recovery curriculum after corrected Izz/Ktau training exposed a
# roll/pitch flip local optimum. Half of resets start directly on the long
# phase. Long commands expand from 0.30--0.38 m to 0.30--0.60 m over 400 PPO
# iterations while short commands remain at 0.00--0.30 m. Extra stability
# shaping applies only during the long phase.
set -euo pipefail

NUM_ENVS="${1:-1024}"
ISAAC_SIM_ROOT="${ISAAC_SIM_ROOT:-/home/terry/Documents/isaacsim/isaac-sim-standalone-5.1.0-linux-x86_64}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

CKPT="${2:-$SCRIPT_DIR/logs/rsl_rl/quadhopper_planner_random_two_hop_v59_custom_relative_next_random_drop_turn_180_tol_10_spd_030_tilt_06_arw_100_lcg_000_blend_100_lax_120_lenapex_tds_095_av_040_fixed_100/2026-09-07_11-12-19/model_29.pt}"

"$ISAAC_SIM_ROOT/python.sh" "$SCRIPT_DIR/train_planner_circular.py" \
  --headless \
  --num_envs "$NUM_ENVS" \
  --route random_two_hop \
  --distance_stage custom \
  --short_radius_min 0.00 \
  --short_radius_max 0.30 \
  --long_radius_min 0.30 \
  --long_radius_max 0.60 \
  --long_hop_curriculum \
  --long_curriculum_radius_min 0.30 \
  --long_curriculum_radius_max 0.38 \
  --long_curriculum_turn_deg 30 \
  --distance_curriculum_iterations 400 \
  --height_stage high \
  --relative_next_hop \
  --start_from_random_drop \
  --planner_reference_blend 1.0 \
  --fine_tune_lr 1.0e-4 \
  --low_apex_event_penalty 120 \
  --lenient_apex_hit \
  --tilt_death_sq_threshold 0.75 \
  --terminate_angvel_norm 30 \
  --iterations 500 \
  --checkpoint "$CKPT" \
  "${@:3}"
