#!/bin/bash
# Compare the same isolated-long checkpoint with normal recurrent continuity
# and with recurrent state cleared at every touchdown. This is an evaluation
# ablation only; robot physics, actuator history, and observations are intact.
set -euo pipefail

MODE="${1:-baseline}"
NUM_ENVS="${2:-512}"
MAX_STEPS="${3:-1500}"
ISAAC_SIM_ROOT="${ISAAC_SIM_ROOT:-/home/terry/Documents/isaacsim/isaac-sim-standalone-5.1.0-linux-x86_64}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
CKPT="${4:-$SCRIPT_DIR/logs/rsl_rl/quadhopper_planner_random_two_hop_v59_custom_relative_next_random_drop_turn_180_tol_10_spd_030_tilt_06_arw_100_lcg_000_blend_100_lax_120_lenapex_longcur_038_400_tds_075_av_030_fixed_100/2026-09-07_11-27-53/model_99.pt}"

case "$MODE" in
  baseline)
    RNN_ARGS=()
    ;;
  touchdown-reset)
    RNN_ARGS=(--reset_rnn_on_touchdown)
    ;;
  *)
    echo "usage: $0 [baseline|touchdown-reset] [num_envs] [max_steps] [checkpoint]" >&2
    exit 2
    ;;
esac

"$ISAAC_SIM_ROOT/python.sh" "$SCRIPT_DIR/play_planner_circular.py" \
  --headless \
  --no_debug_vis \
  --num_envs "$NUM_ENVS" \
  --max_steps "$MAX_STEPS" \
  --checkpoint "$CKPT" \
  --route random_two_hop \
  --distance_stage custom \
  --short_radius_min 0.00 \
  --short_radius_max 0.30 \
  --long_radius_min 0.05 \
  --long_radius_max 0.15 \
  --max_turn_angle 15 \
  --height_stage high \
  --relative_next_hop \
  --start_from_random_drop \
  --planner_reference_blend 1.0 \
  --target_tolerance 0.10 \
  --lenient_apex_hit \
  --tilt_death_sq_threshold 0.75 \
  --terminate_angvel_norm 30 \
  "${RNN_ARGS[@]}"
