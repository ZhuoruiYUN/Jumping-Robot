#!/usr/bin/env bash
# Stage B: introduce a continuous rolling two-target route from the deployed
# v15.1 controller. This script loads weights only; it never resumes PPO state.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
NUM_ENVS="${NUM_ENVS:-256}"
ITERATIONS="${ITERATIONS:-120}"
STEPS="${STEPS:-256}"
LEARNING_RATE="${LEARNING_RATE:-0.0000001}"
CHECKPOINT="${CHECKPOINT:-$SCRIPT_DIR/outputs/tracking_recovery/height_attitude_yaw_40v1_v15_1_cross_hop_recovery/model_359.pt}"
OUTPUT="${OUTPUT:-$SCRIPT_DIR/outputs/tracking_recovery/two_circle_v18_stage_b_$(date +%Y%m%d_%H%M%S)}"

if [[ "${RESUME_RUNNER_STATE:-0}" != "0" ]]; then
  echo "Stage B is a cross-protocol weights-only initialization; RESUME_RUNNER_STATE must be 0." >&2
  exit 2
fi
if [[ ! -f "$CHECKPOINT" ]]; then
  echo "Missing v15.1 source checkpoint: $CHECKPOINT" >&2
  exit 2
fi
if [[ -e "$OUTPUT" ]]; then
  echo "Refusing to overwrite existing output: $OUTPUT" >&2
  exit 2
fi

EXTRA_ARGS=()
if [[ "${DRY_RUN:-0}" == "1" ]]; then
  EXTRA_ARGS+=(--dry-run)
fi

ISAAC_PYTHON="${ISAAC_SIM_ROOT:-/home/terry/Documents/isaacsim/isaac-sim-standalone-5.1.0-linux-x86_64}/python.sh"
exec env -u CONDA_PREFIX -u CONDA_DEFAULT_ENV -u PYTHONHOME -u PYTHONPATH \
  "$ISAAC_PYTHON" "$SCRIPT_DIR/run_tracking_recovery.py" train \
  --root "$OUTPUT" --checkpoint "$CHECKPOINT" \
  --num-envs "$NUM_ENVS" --iterations "$ITERATIONS" --steps "$STEPS" --seed 181 \
  --distance-min 0.06 --distance-max 0.12 \
  --short-distance-min 0.06 --short-distance-max 0.12 \
  --long-distance-min 0.06 --long-distance-max 0.12 \
  --hard-distance-min 0.09 --hard-sample-probability 0.0 \
  --max-turn-angle-deg 20.0 --zero-hop-probability 0.0 \
  --episode-length-s 61 --successful-hops-per-episode 10000 \
  --target-height 1.0 --target-tolerance 0.10 \
  --landing-precision-width 0.12 \
  --target-hit-reward-scale 100.0 \
  --landing-precision-reward-scale 35.0 \
  --landing-error-penalty-scale -45.0 \
  --landing-lateral-error-penalty-scale -30.0 \
  --actuator-mode parity --action-parameterization direct_motor \
  --stability-profile two_circle_v18 --learning-rate "$LEARNING_RATE" \
  --target-relative-apex-gate --apex-tolerance 0.04 \
  --observation-noise-std 0.002 \
  --initial-yaw-perturb-deg 30.0 \
  --initial-angular-velocity-perturb-rad-s 0.15 \
  --airborne-attitude-penalty-scale -4000.0 \
  --airborne-angvel-xy-penalty-scale -8.0 \
  --airborne-yaw-penalty-scale -600.0 \
  --airborne-yaw-rate-penalty-scale -50.0 \
  --airborne-action-spread-penalty-scale -8.0 \
  --airborne-tilt-penalty-start 0.0012 \
  --airborne-angvel-xy-penalty-start 0.35 \
  --airborne-yaw-deadband-deg 3.0 \
  --airborne-yaw-rate-penalty-start 0.20 \
  --airborne-action-spread-penalty-start 0.40 \
  --stationary-hold-radius 0.15 \
  --stationary-tilt-penalty-scale -8.0 \
  --stationary-yaw-rate-penalty-scale -5.0 \
  --stationary-action-spread-penalty-scale -5.0 \
  --stationary-pitch-deadband-deg 2.0 \
  --stationary-pitch-penalty-scale -45.0 \
  --stationary-pitch-rate-deadband-rad-s 0.18 \
  --stationary-pitch-rate-penalty-scale -12.0 \
  --stationary-airborne-differential-action-rate-deadband 0.020 \
  --stationary-airborne-differential-action-rate-penalty-scale -2.0 \
  --stationary-apex-width 0.040 \
  --stationary-apex-reward-scale 100.0 \
  --stationary-apex-error-penalty-scale -160.0 \
  --stationary-apex-shortfall-penalty-scale -260.0 \
  --stationary-apex-progress-scale 30.0 \
  --stationary-apex-progress-width 0.20 \
  --stationary-predicted-apex-width 0.07 \
  --stationary-predicted-apex-reward-scale 24.0 \
  --comprehensive-hardware-randomization \
  --hardware-randomization-probability-start 0.20 \
  --hardware-randomization-probability 0.45 \
  --hardware-randomization-curriculum-iterations 60 \
  --comprehensive-global-thrust-scale-min 0.90 \
  --comprehensive-global-thrust-scale-max 1.04 \
  --comprehensive-com-offset-x-max-mm 2.0 \
  --comprehensive-com-offset-y-max-mm 2.0 \
  --comprehensive-com-offset-z-max-mm 1.0 \
  --comprehensive-yaw-effectiveness-min 0.80 \
  --comprehensive-yaw-effectiveness-max 1.20 \
  --comprehensive-yaw-bias-torque-max-mnm 4.0 \
  "${EXTRA_ARGS[@]}" "$@"
