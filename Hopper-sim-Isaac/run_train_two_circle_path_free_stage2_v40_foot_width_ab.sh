#!/usr/bin/env bash
# Single-variable foot endpoint reward width A/B from the V38 continuous anchor.
# The user starts Isaac training manually; DRY_RUN=1 only prints the launch.
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
if (( $# != 0 )); then
  echo 'Use VARIANT, NUM_ENVS, ITERATIONS, OUTPUT and DRY_RUN; no positional options.' >&2
  exit 2
fi
VARIANT="${VARIANT:-wide09}"
case "$VARIANT" in
  control) FOOT_WIDTH_M=0.06 ;;
  wide09) FOOT_WIDTH_M=0.09 ;;
  *) echo 'VARIANT must be control or wide09.' >&2; exit 2 ;;
esac
ANCHOR="$ROOT/outputs/tracking_recovery/two_circle_path_free_stage2_v38_chain_20261015_083646/model_199.pt"
if [[ ! -f "$ANCHOR" || ! -f "$(dirname -- "$ANCHOR")/manifest.json" || \
      ! -f "$(dirname -- "$ANCHOR")/train_complete.json" ]]; then
  echo "Missing completed V38 source checkpoint, manifest or marker: $ANCHOR" >&2
  exit 2
fi
OUTPUT="${OUTPUT:-$ROOT/outputs/tracking_recovery/two_circle_path_free_stage2_v40_${VARIANT}_$(date +%Y%m%d_%H%M%S)}"
echo "[FOOT-WIDTH-AB] variant=$VARIANT foot_width_m=$FOOT_WIDTH_M source=$ANCHOR output=$OUTPUT"
exec env STAGE=chain CHECKPOINT="$ANCHOR" OUTPUT="$OUTPUT" \
  NUM_ENVS="${NUM_ENVS:-1024}" ITERATIONS="${ITERATIONS:-50}" \
  REVERSE_TURN_PROBABILITY=0.65 STAGE2_FOOT_PRECISION_WIDTH_M="$FOOT_WIDTH_M" \
  bash "$ROOT/run_train_two_circle_path_free_stage2_v38_two_hop.sh"
