#!/usr/bin/env bash
# V39 one-variable continuation: reverse-turn point mass only.
# Actual Isaac training runs only when the user executes this script.
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
if (( $# != 0 )); then
  echo 'Use VARIANT, NUM_ENVS, ITERATIONS, OUTPUT and DRY_RUN; no positional options.' >&2
  exit 2
fi
VARIANT="${VARIANT:-uniform_focus}"
case "$VARIANT" in
  control) REVERSE_TURN_PROBABILITY=0.65 ;;
  uniform_focus) REVERSE_TURN_PROBABILITY=0.45 ;;
  *) echo 'VARIANT must be control or uniform_focus.' >&2; exit 2 ;;
esac
ANCHOR="$ROOT/outputs/tracking_recovery/two_circle_path_free_stage2_v38_chain_20261015_083646/model_199.pt"
if [[ ! -f "$ANCHOR" || ! -f "$(dirname -- "$ANCHOR")/manifest.json" || \
      ! -f "$(dirname -- "$ANCHOR")/train_complete.json" ]]; then
  echo "Missing completed V38 source checkpoint, manifest or marker: $ANCHOR" >&2
  exit 2
fi
OUTPUT="${OUTPUT:-$ROOT/outputs/tracking_recovery/two_circle_path_free_stage2_v39_${VARIANT}_$(date +%Y%m%d_%H%M%S)}"
echo "[DIRECTION-MIX-AB] variant=$VARIANT reverse_probability=$REVERSE_TURN_PROBABILITY source=$ANCHOR output=$OUTPUT"
exec env STAGE=chain CHECKPOINT="$ANCHOR" OUTPUT="$OUTPUT" \
  NUM_ENVS="${NUM_ENVS:-1024}" ITERATIONS="${ITERATIONS:-50}" \
  REVERSE_TURN_PROBABILITY="$REVERSE_TURN_PROBABILITY" \
  bash "$ROOT/run_train_two_circle_path_free_stage2_v38_two_hop.sh"
