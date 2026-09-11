#!/usr/bin/env bash
# Fine-tune the accepted 8--12 cm precision policy with explicit stationary hops.
set -euo pipefail

NUM_ENVS="${1:-1024}"
ITERATIONS="${2:-120}"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
CHECKPOINT="${3:-$SCRIPT_DIR/outputs/tracking_recovery/precision_08_12_turn30_nominal_v1/model_119.pt}"
OUTPUT_ROOT="${4:-$SCRIPT_DIR/outputs/tracking_recovery/precision_zero40_08_12_turn30_nominal_v1}"

cd "$SCRIPT_DIR"

python run_tracking_recovery.py train \
  --checkpoint "$CHECKPOINT" \
  --root "$OUTPUT_ROOT" \
  --num-envs "$NUM_ENVS" \
  --seed 42 \
  --steps 1500 \
  --iterations "$ITERATIONS" \
  --distance-min 0.08 \
  --distance-max 0.12 \
  --hard-sample-probability 0.0 \
  --max-turn-angle-deg 30 \
  --zero-hop-probability 0.40 \
  --learning-rate 1e-6 \
  --target-tolerance 0.05 \
  --landing-precision-width 0.045 \
  --target-hit-reward-scale 180 \
  --landing-precision-reward-scale 100 \
  --landing-error-penalty-scale -150 \
  --landing-lateral-error-penalty-scale -100
