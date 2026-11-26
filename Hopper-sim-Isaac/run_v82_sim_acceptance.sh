#!/usr/bin/env bash
set -euo pipefail

straight_args=()
if [[ "${1:-}" == "--isolate-reset-rng" ]]; then
  straight_args=(--isolate-eval-reset-rng)
  shift
fi
if (( $# < 1 || $# > 3 )); then
  echo 'Usage: bash run_v82_sim_acceptance.sh [--isolate-reset-rng] CHECKPOINT [NEW_OUTPUT_DIR] [FRESH_NATIVE_HOLDOUT_SEED]' >&2
  exit 2
fi

checkpoint="$(realpath -e -- "$1")"
output_input="${2:-$(dirname "$checkpoint")/acceptance_$(basename "$checkpoint" .pt)_$(date -u +%Y%m%d_%H%M%S_%NZ)}"
output="$(realpath -m -- "$output_input")"
[[ -f "$checkpoint" && ! -e "$output" ]]
holdout_seed=9109
holdout_args=()
if (( $# == 3 )); then
  if [[ ! "$3" =~ ^[1-9][0-9]{4,9}$ ]] || (( $3 >= 2147483648 )); then
    echo 'Fresh native holdout seed must be between 10000 and 2147483647.' >&2
    exit 2
  fi
  holdout_seed="$3"
  holdout_args=(--fresh-holdout-seed "$holdout_seed")
fi

cd "$(dirname "${BASH_SOURCE[0]}")"
base="$PWD/outputs/benchmark73_precision10_v50"
reference="$base/ab_ws_first_precision_v82/train_seed42_20261125_094124_358318Z/eval_model_249"
analysis="$reference/analysis"
audit="$analysis/rpy_audit.py"
if (( ${#straight_args[@]} )); then
  reference="$(dirname "$reference")/eval_model_249_reset_rng_seed7001"
  /usr/bin/python3 - "$reference" <<'PY'
from pathlib import Path
import sys

from benchmark37.v82_sim_acceptance import load_suites

suites = load_suites(Path(sys.argv[1]), 7001)
if suites["cardinal"]["manifest"].get("evaluation_reset_rng") != "first_episode_dedicated_reset_rng_v1":
    raise ValueError("The paired V82 reference must use reset-isolated evaluation")
PY
fi
isaac_root=/home/terry/Documents/isaacsim/isaac-sim-standalone-5.1.0-linux-x86_64
lab_root="$(cd ../../IsaacLab && pwd)"
[[ -f "$audit" ]]
unset PYTHONEXE PYTHONHOME CONDA_PREFIX CONDA_DEFAULT_ENV
export PYTHONPATH="$PWD:$isaac_root/exts/omni.isaac.ml_archive/pip_prebundle:$lab_root/source/isaaclab:$lab_root/source/isaaclab_rl"
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1

mkdir -p -- "$(dirname "$output")"
mkdir -- "$output"
for suite in cardinal opposite; do
  source=cardinal_half_metre
  if [[ "$suite" == opposite ]]; then source=opposite_cardinal_half_metre; fi
  printf 'Evaluating %s (seed 7001) in %s\n' "$suite" "$output"
  flock --exclusive --close "$base/isaac-gpu.lock" \
    "$isaac_root/python.sh" "$audit" eval --headless --device cuda:0 --seed 7001 \
    --checkpoint "$checkpoint" --canonical-axis --audit-all-phase-yaw \
    "${straight_args[@]}" \
    --waypoint-source "benchmark37.waypoint_stream:$source" --output "$output/$suite" 2>&1 \
    | tee "$output/$suite.log"
done

/usr/bin/python3 -m benchmark37.v82_sim_acceptance "$output" --reference "$reference" \
  --response-strata --output "$output/acceptance.json"
/usr/bin/python3 "$analysis/compare_y_release_pitch.py" "$reference" "$output" \
  --output "$output/y_release_pitch.json"
printf 'Acceptance: %s\n' "$output/acceptance.json"
/usr/bin/python3 - "$output/acceptance.json" <<'PY'
import json
import sys

with open(sys.argv[1]) as report_file:
    candidate = json.load(report_file)['candidate']
for name in ('route_first', 'target_first', 'retry'):
    hit = candidate['overall'][name]['foot_5cm']
    rate = 'n/a' if hit['rate'] is None else f"{hit['rate']:.2%}"
    print(f"{name} 5 cm: {hit['count']}/{hit['total']} ({rate})")
print('Straight gate:', 'PASS' if candidate['passed'] else 'FAIL')
for name in candidate['failed_checks']:
    gate = candidate['gates'][name]
    print(f"  {name}: {gate['value']} (gate {gate['comparison']} {gate['limit']})")
raise SystemExit(int(not candidate['passed']))
PY

printf 'Evaluating native mixed-route holdout (seed %s)\n' "$holdout_seed"
bash run_benchmark37_ring.sh eval --headless --device cuda:0 --checkpoint "$checkpoint" \
  --precision-suite holdout --canonical-axis --audit-all-phase-yaw "${holdout_args[@]}" \
  --output "$output/holdout" 2>&1 | tee "$output/holdout.log"
printf 'Evaluating saved stationary retention cases\n'
bash run_benchmark37_ring.sh eval --headless --device cuda:0 --checkpoint "$checkpoint" \
  --stationary-retention-eval --audit-all-phase-yaw --output "$output/stationary" 2>&1 \
  | tee "$output/stationary.log"

/usr/bin/python3 - "$output" "$checkpoint" "$holdout_seed" <<'PY'
import json
import sys
from pathlib import Path

output, checkpoint = Path(sys.argv[1]), Path(sys.argv[2])
with (output / 'acceptance.json').open() as handle:
    straight = json.load(handle)
straight_result = straight['candidate']
verdicts = {'straight': straight_result['passed']}
for suite, seed, scope in (('holdout', int(sys.argv[3]), 'precision10_v1_holdout'),
                           ('stationary', 7001, 'stationary_retention')):
    directory = output / suite
    manifest = json.loads((directory / 'manifest.json').read_text())
    if (manifest['status'] != 'complete' or manifest['optimizer_updates'] != 0
            or Path(manifest['checkpoint']).resolve() != checkpoint or manifest['seed'] != seed
            or manifest['evaluation_scope'] != scope):
        raise ValueError(f'{suite} has no completed evaluation for this checkpoint and protocol')
    summary = json.loads((directory / 'summary.json').read_text())
    verdicts[suite] = summary['promotion_gate']['passed']
    print(f"{suite} gate: {'PASS' if verdicts[suite] else 'FAIL'}")
height_scopes = {
    name: straight_result['overall'][name]['apex_1_to_1p10m']
    for name in ('route_first', 'target_first', 'retry')
}
rows = json.loads((output / 'holdout/episodes.json').read_text())
moving_first = [e for row in rows for e in row['events'] if e['moving'] and e['first_attempt_for_target']]
height_qualified = sum(e['valid_flight'] and 1.00 <= e['apex_root_link_world_m'] <= 1.10 for e in moving_first)
report = {
    'checkpoint': str(checkpoint),
    'defined_numeric_gates_passed': all(verdicts.values()),
    'suites': verdicts,
    'straight_height_band_1_to_1p10m': height_scopes,
    'holdout_target_first_height_band_1_to_1p10m': {'count': height_qualified, 'total': len(moving_first)},
    'height_band_review': 'No aggregate pass fraction is defined. See the joint rates in acceptance.json.',
}
(output / 'qualification.json').write_text(json.dumps(report, indent=2) + '\n')
print(f'Qualification: {output / "qualification.json"}')
raise SystemExit(int(not report['defined_numeric_gates_passed']))
PY
