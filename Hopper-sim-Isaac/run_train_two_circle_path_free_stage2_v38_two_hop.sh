#!/usr/bin/env bash
# Stage A: nominal two-hop adaptation. B: ramp hardware. C: long chains.
# Actual Isaac training is started only when the user runs this script.
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
if (( $# != 0 )); then
  echo 'Use STAGE, CHECKPOINT, NUM_ENVS, ITERATIONS, OUTPUT, DRY_RUN; no positional options.' >&2
  exit 2
fi
STAGE="${STAGE:-nominal}"
REVERSE_TURN_PROBABILITY="${REVERSE_TURN_PROBABILITY:-0.65}"
case "$STAGE" in
  nominal)
    DEFAULT_CHECKPOINT="$ROOT/outputs/tracking_recovery/two_circle_path_free_stage2_v37_sysid_20261013_164822/model_99.pt"
    DEFAULT_ITERATIONS=100
    FINAL_HARD_PROBABILITY=0
    START_HARD_PROBABILITY=0
    RAMP_ITERATIONS=1
    EPISODE_LENGTH_S=12
    EPISODE_ARGS=(--two-hop-episode-v38)
    ;;
  robust)
    if [[ -z "${CHECKPOINT:-}" ]]; then
      echo 'Robust stage requires CHECKPOINT from a completed nominal v38 run.' >&2
      exit 2
    fi
    DEFAULT_CHECKPOINT="$CHECKPOINT"
    DEFAULT_ITERATIONS=150
    FINAL_HARD_PROBABILITY=0.45
    START_HARD_PROBABILITY=0
    RAMP_ITERATIONS=100
    EPISODE_LENGTH_S=12
    EPISODE_ARGS=(--two-hop-episode-v38)
    ;;
  chain_short|chain_mid|chain_bridge)
    if [[ -z "${CHECKPOINT:-}" ]]; then
      echo 'Continuous-chain stage requires CHECKPOINT from a completed previous stage.' >&2
      exit 2
    fi
    DEFAULT_CHECKPOINT="$CHECKPOINT"
    DEFAULT_ITERATIONS=50
    FINAL_HARD_PROBABILITY=0.45
    START_HARD_PROBABILITY=0.45
    RAMP_ITERATIONS=1
    case "$STAGE" in
      chain_short) EPISODE_LENGTH_S=12 ;;
      chain_mid) EPISODE_LENGTH_S=24 ;;
      chain_bridge) EPISODE_LENGTH_S=40; DEFAULT_ITERATIONS=80 ;;
    esac
    EPISODE_ARGS=()
    ;;
  chain)
    if [[ -z "${CHECKPOINT:-}" ]]; then
      echo 'Chain stage requires CHECKPOINT from a completed robust v38 run.' >&2
      exit 2
    fi
    DEFAULT_CHECKPOINT="$CHECKPOINT"
    DEFAULT_ITERATIONS=100
    FINAL_HARD_PROBABILITY=0.45
    START_HARD_PROBABILITY=0.45
    RAMP_ITERATIONS=1
    EPISODE_LENGTH_S=61
    EPISODE_ARGS=()
    ;;
  *)
    echo 'STAGE must be nominal, robust, chain_short, chain_mid, chain_bridge or chain.' >&2
    exit 2
    ;;
esac
ANCHOR="${CHECKPOINT:-$DEFAULT_CHECKPOINT}"
if [[ ! -f "$ANCHOR" || ! -f "$(dirname -- "$ANCHOR")/manifest.json" || \
      ! -f "$(dirname -- "$ANCHOR")/train_complete.json" ]]; then
  echo "Missing completed source checkpoint, manifest or marker: $ANCHOR" >&2
  exit 2
fi
OUTPUT="${OUTPUT:-$ROOT/outputs/tracking_recovery/two_circle_path_free_stage2_v38_${STAGE}_$(date +%Y%m%d_%H%M%S)}"
echo "[TWO-HOP-V38] stage=$STAGE source=$ANCHOR output=$OUTPUT"
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
  NUM_ENVS="${NUM_ENVS:-1024}" ITERATIONS="${ITERATIONS:-$DEFAULT_ITERATIONS}" \
  LEARNING_RATE=0.0000003 \
  bash "$ROOT/run_train_two_circle_path_free_stage2_v3.sh" \
  --hardware-randomization-probability "$FINAL_HARD_PROBABILITY" \
  --hardware-randomization-probability-start "$START_HARD_PROBABILITY" \
  --hardware-randomization-curriculum-iterations "$RAMP_ITERATIONS" \
  --episode-length-s "$EPISODE_LENGTH_S" --successful-hops-per-episode 10000 \
  --v3-accounting-fix --foot-pair-v33 "${EPISODE_ARGS[@]}" \
  --height-progress-reward-scale 3 \
  --liftoff-cross-track-penalty-scale 0 \
  --stage2-foot-precision-scale 60 \
  --stage2-foot-precision-width-m "${STAGE2_FOOT_PRECISION_WIDTH_M:-0.06}" \
  --pair-xy-quality-weight-v36 0 \
  --reverse-turn-probability "$REVERSE_TURN_PROBABILITY" \
  --comprehensive-global-thrust-scale-min 0.95 \
  --comprehensive-global-thrust-scale-max 1.05 \
  --comprehensive-yaw-effectiveness-min 1 \
  --comprehensive-yaw-effectiveness-max 1 \
  --comprehensive-yaw-bias-torque-max-mnm 0 \
  --sysid-20261011-plant
