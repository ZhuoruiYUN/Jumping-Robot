#!/usr/bin/env bash
# Matched V41 continuation from the V38 chain anchor; one reward variable.
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
if (( $# != 0 )); then
  echo 'Use VARIANT, NUM_ENVS, ITERATIONS, OUTPUT and DRY_RUN; no positional options.' >&2
  exit 2
fi
VARIANT="${VARIANT:-bonus20}"
case "$VARIANT" in
  control) HIT10_ENV=(-u STAGE2_FOOT_HIT_10CM_SCALE); HIT10_SCALE=0 ;;
  bonus20) HIT10_ENV=(STAGE2_FOOT_HIT_10CM_SCALE=20); HIT10_SCALE=20 ;;
  *) echo 'VARIANT must be control or bonus20.' >&2; exit 2 ;;
esac
ANCHOR="$ROOT/outputs/tracking_recovery/two_circle_path_free_stage2_v38_chain_20261015_083646/model_199.pt"
if [[ ! -f "$ANCHOR" || ! -f "$(dirname -- "$ANCHOR")/manifest.json" || \
      ! -f "$(dirname -- "$ANCHOR")/train_complete.json" ]]; then
  echo "Missing completed V38 source checkpoint, manifest or marker: $ANCHOR" >&2
  exit 2
fi
OUTPUT="${OUTPUT:-$ROOT/outputs/tracking_recovery/two_circle_path_free_stage2_v41_${VARIANT}_$(date +%Y%m%d_%H%M%S)}"
echo "[FOOT-HIT10-AB] variant=$VARIANT hit10_scale=$HIT10_SCALE source=$ANCHOR output=$OUTPUT"
exec env "${HIT10_ENV[@]}" STAGE=chain CHECKPOINT="$ANCHOR" OUTPUT="$OUTPUT" \
  NUM_ENVS="${NUM_ENVS:-1024}" ITERATIONS="${ITERATIONS:-50}" \
  REVERSE_TURN_PROBABILITY=0.65 STAGE2_FOOT_PRECISION_WIDTH_M=0.06 \
  bash "$ROOT/run_train_two_circle_path_free_stage2_v38_two_hop.sh"
