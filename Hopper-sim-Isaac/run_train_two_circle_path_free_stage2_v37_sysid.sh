#!/usr/bin/env bash
# Current Stage2 training configuration with the 2026-10-11 mounted SysID plant.
# Weights-only continuation from the completed v36 control; no eval is launched.
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
if (( $# != 0 )); then
  echo 'Use NUM_ENVS, ITERATIONS, OUTPUT and DRY_RUN; no positional options.' >&2
  exit 2
fi
ANCHOR="$ROOT/outputs/tracking_recovery/two_circle_path_free_stage2_v36_control_20261013_153740/model_99.pt"
if [[ ! -f "$ANCHOR" || ! -f "$(dirname -- "$ANCHOR")/manifest.json" ]]; then
  echo "Missing completed v36 control anchor or manifest: $ANCHOR" >&2
  exit 2
fi
OUTPUT="${OUTPUT:-$ROOT/outputs/tracking_recovery/two_circle_path_free_stage2_v37_sysid_$(date +%Y%m%d_%H%M%S)}"
echo "[SYSID-V37] source=$ANCHOR output=$OUTPUT"
unset http_proxy https_proxy all_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY
if [[ "${DRY_RUN:-0}" != 1 ]]; then
  GROUND_URL='https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/5.1/Isaac/Environments/Grid/default_environment.usd'
  echo '[ASSET-PREFLIGHT] Checking the original NVIDIA ground asset over direct HTTPS'
  if ! curl --noproxy '*' --fail --silent --show-error --location \
      --connect-timeout 10 --max-time 30 --retry 1 \
      --output /dev/null "$GROUND_URL"; then
    echo '[ASSET-PREFLIGHT] Direct download failed; Isaac training was not started.' >&2
    exit 1
  fi
  if [[ "${ASSET_PREFLIGHT_ONLY:-0}" == 1 ]]; then
    echo '[ASSET-PREFLIGHT] Preflight-only mode; Isaac training was not started'
    exit 0
  fi
fi
exec env CHECKPOINT="$ANCHOR" OUTPUT="$OUTPUT" \
  NUM_ENVS="${NUM_ENVS:-1024}" ITERATIONS="${ITERATIONS:-100}" \
  LEARNING_RATE=0.0000003 \
  bash "$ROOT/run_train_two_circle_path_free_stage2_v3.sh" \
  --hardware-randomization-probability 0.45 \
  --hardware-randomization-probability-start 0.45 \
  --hardware-randomization-curriculum-iterations 1 \
  --episode-length-s 61 --v3-accounting-fix --foot-pair-v33 \
  --height-progress-reward-scale 3 \
  --liftoff-cross-track-penalty-scale 0 \
  --stage2-foot-precision-scale 60 \
  --pair-xy-quality-weight-v36 0 \
  --comprehensive-global-thrust-scale-min 0.95 \
  --comprehensive-global-thrust-scale-max 1.05 \
  --comprehensive-yaw-effectiveness-min 1 \
  --comprehensive-yaw-effectiveness-max 1 \
  --comprehensive-yaw-bias-torque-max-mnm 0 \
  --sysid-20261011-plant
