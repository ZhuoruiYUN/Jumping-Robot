#!/usr/bin/env bash
# Canonical native-Isaac evaluation for the retained hardware-proven 40_v1.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ISAAC_PYTHON="${QUADHOPPER_PYTHON:-/home/terry/Documents/isaacsim/isaac-sim-standalone-5.1.0-linux-x86_64/python.sh}"
CHECKPOINT="$SCRIPT_DIR/outputs/tracking_recovery/precision_zero40_08_12_turn30_nominal_v1/model_119.pt"
STAMP="$(date +%Y%m%d_%H%M%S)"
OUTPUT="$SCRIPT_DIR/outputs/tracking_recovery/isaac_40v1_stationary_1m_$STAMP"

"$ISAAC_PYTHON" "$SCRIPT_DIR/run_tracking_recovery.py" eval \
  --checkpoint "$CHECKPOINT" --root "$OUTPUT" --num-envs 1 --steps 30000 --seed 42 \
  --distance-min 0.001 --distance-max 0.001 --hard-sample-probability 0 \
  --max-turn-angle-deg 1 --zero-hop-probability 1 --target-height 1.0 \
  --episode-length-s 301 \
  --target-tolerance 0.05 --actuator-mode baseline --legacy-v11-contract \
  --allow-compatible-checkpoint

python "$SCRIPT_DIR/play_spatial_tracking_mujoco1.py" "$OUTPUT/single_env.csv" --target-height 1.0
echo "[ISAAC-40V1] output=$OUTPUT"
