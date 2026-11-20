#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
mode="${1:-}"
if [[ -z "$mode" ]]; then
  echo 'Usage: bash run_benchmark73_phase_motion.sh train|check|select-second-hop|eval|eval-stationary|eval-straight|eval-straight05|eval-step-hold|eval-cardinal-queue|eval-guarded-keys|eval-guarded-suite|eval-guarded-remaining|play [arguments]' >&2
  exit 2
fi
shift
course="benchmark37/configs/obs73_quaternion_straight_focus_course_v23.json"
case "$mode" in
  select-second-hop) exec python3 -m benchmark37.second_release_selection "$@" ;;
  train)
    # Resolve the final selected course before choosing its source. Explicit
    # initialization remains authoritative; this wrapper never launches until
    # the user invokes train.
    selected_course="$course"
    supplied_init=false
    train_args=("$@")
    for ((i=0; i<${#train_args[@]}; i++)); do
      case "${train_args[i]}" in
        --ring-config)
          if (( i+1 >= ${#train_args[@]} )); then
            echo '--ring-config requires a path' >&2
            exit 2
          fi
          selected_course="${train_args[i+1]}"
          ;;
        --ring-config=*) selected_course="${train_args[i]#*=}" ;;
        --initialize-*|--from-scratch) supplied_init=true ;;
      esac
    done
    initial_args=()
    if [[ "$supplied_init" == false ]]; then
      initial_checkpoint="$(python3 - "$selected_course" <<'PY'
import json,sys
from pathlib import Path
course=json.loads(Path(sys.argv[1]).read_text())
print(Path(course['omni']['initial_checkpoint']).resolve(strict=True))
PY
)"
      initial_args=(--initialize-from "$initial_checkpoint")
    fi
    exec bash run_benchmark73_quaternion_omni.sh train --ring-config "$course" "${initial_args[@]}" "$@"
    ;;
  check)
    exec bash run_benchmark73_quaternion_omni.sh check --ring-config "$course" "$@"
    ;;
  eval-guarded-suite|eval-guarded-remaining)
    if [[ $# != 1 || ! -f "$1" ]]; then echo "Usage: $mode PATH_TO_CHECKPOINT" >&2; exit 2; fi
    ckpt="$1"
    baseline=outputs/benchmark73_quaternion_omni_reversal_v37
    if [[ "$mode" == eval-guarded-suite ]]; then
      bash run_benchmark37_ring.sh eval-guarded-keys --checkpoint "$ckpt"
      bash run_benchmark73_quaternion_omni.sh eval --headless --checkpoint "$ckpt" --cases-file "$baseline/eval_seed7001_20261118_094602_663803Z/cases.json"
    fi
    # The runner resolves the original v23 cases from the saved checkpoint course.
    bash run_benchmark73_quaternion_omni.sh eval --headless --stationary-retention-eval --checkpoint "$ckpt"
    exec bash run_benchmark73_quaternion_omni.sh eval --headless --cardinal-queue-eval --checkpoint "$ckpt" --cases-file "$baseline/eval_seed7001_20261118_094400_585026Z/cases.json"
    ;;
  eval-guarded-keys) exec bash run_benchmark37_ring.sh eval-guarded-keys "$@" ;;
  eval-stationary)
    exec bash run_benchmark73_quaternion_omni.sh eval --stationary-retention-eval "$@"
    ;;
  eval-straight)
    exec bash run_benchmark73_quaternion_omni.sh eval --straight-x-step-m 0.4 --straight-jumps 10 "$@"
    ;;
  eval-straight05)
    exec bash run_benchmark73_quaternion_omni.sh eval --straight-x-step-m 0.5 --straight-jumps 10 "$@"
    ;;
  eval-step-hold)
    exec bash run_benchmark73_quaternion_omni.sh eval --step-hold-eval "$@"
    ;;
  eval-cardinal-queue)
    exec bash run_benchmark73_quaternion_omni.sh eval --cardinal-queue-eval "$@"
    ;;
  eval|play) exec bash run_benchmark73_quaternion_omni.sh "$mode" "$@" ;;
  *) echo 'Use train/check/eval/eval-stationary/eval-straight/eval-straight05/eval-step-hold/eval-cardinal-queue/play; training is manual.' >&2; exit 2 ;;
esac
