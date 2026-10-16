#!/usr/bin/env bash
# Strict 1 m success refinement of the best zero-curve policy; no added energy cost.
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec env \
  CHECKPOINT="${CHECKPOINT:-$SCRIPT_DIR/outputs/tracking_recovery/two_circle_path_free_stage2_v2_20260922_192329/model_599.pt}" \
  OUTPUT="${OUTPUT:-$SCRIPT_DIR/outputs/tracking_recovery/two_circle_path_free_stage2_v3_$(date +%Y%m%d_%H%M%S)}" \
  ITERATIONS="${ITERATIONS:-600}" NUM_ENVS="${NUM_ENVS:-256}" \
  LEARNING_RATE="${LEARNING_RATE:-0.0000003}" \
  bash "$SCRIPT_DIR/run_train_two_circle_v19_history_foot.sh" \
  --path-free-stage2-v3 \
  --target-hit-reward-scale 140 --landing-precision-reward-scale 50 \
  --distance-min 0.12 --distance-max 0.50 \
  --short-distance-min 0.12 --short-distance-max 0.30 \
  --long-distance-min 0.30 --long-distance-max 0.50 \
  --hard-sample-probability 0 --distance-curriculum-iterations 0 \
  --max-turn-angle-deg 180 --zero-hop-probability 0.12 \
  --reverse-turn-probability 0.65 --reverse-turn-halfwidth-deg 30 \
  --direction-curriculum-iterations 0 \
  --hardware-randomization-probability 0.45 \
  --hardware-randomization-probability-start 0.45 \
  --hardware-randomization-curriculum-iterations 1 \
  "$@"
