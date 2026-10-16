#!/usr/bin/env bash
# Matched frozen-policy A/B for the V40 touchdown reward width experiment.
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
if (( $# != 2 )); then
  echo "Usage: bash $0 CONTROL_MODEL_PT WIDE09_MODEL_PT" >&2
  exit 2
fi
CONTROL="$1"
WIDE09="$2"
for checkpoint in "$CONTROL" "$WIDE09"; do
  if [[ ! -f "$checkpoint" || ! -f "$(dirname -- "$checkpoint")/manifest.json" ]]; then
    echo "Missing checkpoint or manifest: $checkpoint" >&2
    exit 2
  fi
done
PROFILE="${PROFILE:-historical}"
case "$PROFILE" in
  historical) DEFAULT_SEED=583 ;;
  checkpoint) DEFAULT_SEED=483 ;;
  *) echo 'PROFILE must be historical or checkpoint' >&2; exit 2 ;;
esac
SEED="${SEED:-$DEFAULT_SEED}"
STEPS="${STEPS:-30000}"
if [[ ! "$SEED" =~ ^[0-9]+$ || ! "$STEPS" =~ ^[1-9][0-9]*$ ]]; then
  echo 'SEED must be nonnegative and STEPS must be positive integers' >&2
  exit 2
fi
OUT_ROOT="${OUT_ROOT:-$ROOT/outputs/tracking_recovery/v40_foot_width_ab_${PROFILE}_seed${SEED}_$(date +%Y%m%d_%H%M%S)}"
mkdir -p "$OUT_ROOT"

run_case() {
  local label="$1" checkpoint="$2" case_root="$OUT_ROOT/$1"
  if [[ -f "$case_root/summary.json" ]]; then
    echo "[V40-EVAL] already complete: $case_root/summary.json"
    return
  fi
  if [[ -e "$case_root" ]]; then
    echo "[V40-EVAL] incomplete case: $case_root (inspect it; set a new OUT_ROOT to retry)" >&2
    exit 1
  fi
  bash "$ROOT/run_directional_eval_direct.sh" "$checkpoint" \
    --root "$case_root" --condition robust --scenario all \
    --robust-profile "$PROFILE" --robust-seed "$SEED" \
    --episode-mode continuous --episode-length-s 61 --steps "$STEPS"
}

run_case control "$CONTROL"
run_case wide09 "$WIDE09"
echo "[V40-EVAL] complete: $OUT_ROOT"
