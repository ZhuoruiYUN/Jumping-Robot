#!/usr/bin/env bash
# Same Stage-C task, expanded observations, weights-only initialization.
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec env \
  CHECKPOINT="${CHECKPOINT:-$SCRIPT_DIR/outputs/tracking_recovery/two_circle_v18_stage_c_20260921_145051/model_119.pt}" \
  OUTPUT="${OUTPUT:-$SCRIPT_DIR/outputs/tracking_recovery/two_circle_v19_history_foot_stage_c_$(date +%Y%m%d_%H%M%S)}" \
  RESUME_RUNNER_STATE=0 \
  bash "$SCRIPT_DIR/run_train_40v1_two_circle_v18_stage_b.sh" \
  --observation-layout history_foot_v19 \
  --distance-min 0.12 --distance-max 0.25 \
  --short-distance-min 0.12 --short-distance-max 0.25 \
  --long-distance-min 0.25 --long-distance-max 0.40 \
  --hard-distance-min 0.18 --max-turn-angle-deg 25 "$@"
