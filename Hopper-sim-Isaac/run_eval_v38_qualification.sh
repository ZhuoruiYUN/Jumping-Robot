#!/usr/bin/env bash
# Frozen-policy V38 comparison. Run manually; each child uses Isaac python.sh.
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PHASE="${1:-screen}"
case "$PHASE" in
  screen) DEFAULT_SEED=483; ROBUST_PROFILE=checkpoint ;;
  confirm) DEFAULT_SEED=583; ROBUST_PROFILE=historical ;;
  *) echo "Usage: bash $0 [screen|confirm]" >&2; exit 2 ;;
esac
SEED="${SEED:-$DEFAULT_SEED}"
STEPS="${STEPS:-30000}"
if [[ ! "$SEED" =~ ^[0-9]+$ || ! "$STEPS" =~ ^[1-9][0-9]*$ ]]; then
  echo 'SEED must be nonnegative and STEPS must be positive integers' >&2
  exit 2
fi

ANCHOR="$ROOT/outputs/tracking_recovery/two_circle_path_free_stage2_v38_robust_20261014_112859/model_149.pt"
CANDIDATE="$ROOT/outputs/tracking_recovery/two_circle_path_free_stage2_v38_chain_20261015_083646/model_199.pt"
OUT_ROOT="${OUT_ROOT:-$ROOT/outputs/tracking_recovery/v38_qualification_${PHASE}_seed${SEED}_$(date +%Y%m%d_%H%M%S)}"
mkdir -p "$OUT_ROOT"

run_case() {
  local label="$1" checkpoint="$2" mode="$3" length="$4"
  local case_root="$OUT_ROOT/$label"
  if [[ -f "$case_root/summary.json" ]]; then
    echo "[V38-EVAL] already complete: $case_root/summary.json"
    return
  fi
  if [[ -e "$case_root" ]]; then
    echo "[V38-EVAL] incomplete case: $case_root (inspect it; set a new OUT_ROOT to retry)" >&2
    exit 1
  fi
  echo "[V38-EVAL] $label seed=$SEED steps=$STEPS"
  bash "$ROOT/run_directional_eval_direct.sh" "$checkpoint" \
    --root "$case_root" --episode-mode "$mode" \
    --episode-length-s "$length" --condition robust \
    --robust-profile "$ROBUST_PROFILE" \
    --robust-seed "$SEED" --steps "$STEPS"
}

# Paired two-hop task retains the fixed 50/25/25 direction mix and terminal
# settle check. Continuous tests cover every turn, stationary, and reverse.
run_case anchor_two_hop "$ANCHOR" two_hop 12
run_case candidate_two_hop "$CANDIDATE" two_hop 12
run_case anchor_chain61 "$ANCHOR" continuous 61
run_case candidate_chain61 "$CANDIDATE" continuous 61
if [[ "$PHASE" == confirm ]]; then
  # Five 61-second training horizons with no scheduled intermediate reset.
  run_case candidate_chain301 "$CANDIDATE" continuous 301
fi
echo "[V38-EVAL] complete: $OUT_ROOT"
