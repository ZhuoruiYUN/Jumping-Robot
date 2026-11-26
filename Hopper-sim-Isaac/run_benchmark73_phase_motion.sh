#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
mode="${1:-}"
if [[ -z "$mode" ]]; then
  echo 'Usage: bash run_benchmark73_phase_motion.sh train|train-and-eval|check|select-second-hop|eval|eval-stationary|eval-straight|eval-straight05|eval-step-hold|eval-cardinal-queue|eval-guarded-keys|eval-guarded-suite|eval-guarded-remaining|play [arguments]' >&2
  exit 2
fi
shift
course="training_config_v51/ab_only/course.json"
case "$mode" in
  select-second-hop) exec python3 -m benchmark37.second_release_selection "$@" ;;
  train-and-eval)
    if (( $# < 2 )) || [[ "$1" == -* || "$2" == -* ]]; then
      echo 'Usage: train-and-eval NEW_RUN_DIR COURSE [--fresh-holdout-seed SEED] [TRAIN_ARGS...]' >&2
      exit 2
    fi
    run_dir="$(realpath -m -- "$1")"
    selected_course="$(realpath -e -- "$2")"
    shift 2
    holdout_seed=()
    if [[ "${1:-}" == --fresh-holdout-seed ]]; then
      if [[ ! "${2:-}" =~ ^[1-9][0-9]{4,9}$ ]] || (( $2 >= 2147483648 )); then
        echo 'Fresh native holdout seed must be between 10000 and 2147483647.' >&2
        exit 2
      fi
      holdout_seed=("$2")
      shift 2
    fi
    for arg in "$@"; do
      case "$arg" in
        --output|--output=*|--ring-config|--ring-config=*)
          echo 'Pass the new run directory and course as the first two arguments.' >&2
          exit 2 ;;
      esac
    done
    if [[ -e "$run_dir" ]]; then
      printf 'Training directory already exists: %s\n' "$run_dir" >&2
      exit 2
    fi
    if ! /usr/bin/python3 - "$selected_course" <<'PY'
from pathlib import Path
import sys

from benchmark37.v82_sim_acceptance import V82_COURSE, load_suites, read_json

trial = Path("outputs/benchmark73_precision10_v50/ab_ws_first_precision_v82/train_seed42_20261125_094124_358318Z")
reference = load_suites(trial / "eval_model_249_reset_rng_seed7001", 7001)
if reference["cardinal"]["manifest"].get("evaluation_reset_rng") != "first_episode_dedicated_reset_rng_v1":
    sys.exit("The paired V82 baseline must use the reset-isolated protocol.")
candidate, v82 = (read_json(Path(path)) for path in (sys.argv[1], V82_COURSE))
for key in ("task", "observation_version", "imu_link_plant", "rigid_body_randomization"):
    if candidate.get(key) != v82[key]:
        sys.exit(f"Course differs from the V82 paired acceptance protocol in {key}.")
if not candidate.get("omni", {}).get("ab_pair_only"):
    sys.exit("The paired AB workflow requires an AB training course.")
for key in ("version", "nominal_fraction", "roll_pitch_gain_range"):
    a, b = (item["vertical_pretraining"]["response_robustness"].get(key) for item in (candidate, v82))
    if a != b:
        sys.exit(f"Course differs from the V82 fixed response grid in {key}.")
PY
    then
      exit 2
    fi
    if ! bash run_benchmark73_phase_motion.sh train "$@" --ring-config "$selected_course" --output "$run_dir"; then
      printf 'Training did not finish successfully: %s\n' "$run_dir" >&2
      exit 2
    fi
    if ! checkpoint="$(/usr/bin/python3 - "$run_dir" <<'PY'
import json
from pathlib import Path
import sys

folder = Path(sys.argv[1])
manifest = json.loads((folder / "manifest.json").read_text())
updates, iteration = manifest.get("requested_iterations"), manifest.get("iteration")
if (manifest.get("status") != "complete" or manifest.get("mode") != "train"
        or type(updates) is not int or updates < 1
        or type(iteration) is not int or iteration != updates - 1):
    sys.exit(f"Expected completed training at its final requested iteration in {folder}")
checkpoint = folder / f"model_{iteration}.pt"
if not checkpoint.is_file() or not checkpoint.stat().st_size:
    sys.exit(f"Missing final checkpoint {checkpoint}")
print(checkpoint)
PY
)"
    then
      exit 2
    fi
    printf 'Training complete: %s\nStarting simulation acceptance.\n' "$checkpoint"
    exec bash run_v82_sim_acceptance.sh --isolate-reset-rng "$checkpoint" "$run_dir/acceptance" "${holdout_seed[@]}"
    ;;
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
  *) echo 'Use train/train-and-eval/check/eval/eval-stationary/eval-straight/eval-straight05/eval-step-hold/eval-cardinal-queue/play; training is manual.' >&2; exit 2 ;;
esac
