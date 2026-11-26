#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
base="$PWD/outputs/benchmark73_precision10_v50/ab_architecture_comparison"
pair="$base/$(date -u +%Y%m%dT%H%M%SZ)"
mkdir -p "$base"
mkdir "$pair"
course='training_config_v51/ab_yaw_heading/course.json'
echo "Comparison directory: $pair"
# Kit shutdown can mask an exception exit status. Verify each persisted result.
verify() {
    python3 - "$1" "$2" <<'PY'
import json, sys
from pathlib import Path
folder=Path(sys.argv[1]); m=json.loads((folder/'manifest.json').read_text())
assert m['status']=='complete',m.get('error',m['status'])
if sys.argv[2]=='eval':
    assert m['optimizer_updates']==0 and (folder/'episodes.json').is_file()
else:
    step=int(sys.argv[2])-1
    assert m['iteration']==step and (folder/f'model_{step}.pt').is_file()
PY
}
train() {
    bash run_benchmark37_ring.sh train --headless --ring-config "$course" --from-scratch \
        --ab-policy "$1" --iterations "$2" --output "$3" >"$3.log" 2>&1
    verify "$3" "$2"
}
for arm in lstm mlp; do
    train "$arm" 1 "$pair/smoke_$arm"
done
for arm in lstm mlp; do
    folder="$pair/$arm"
    train "$arm" 50 "$folder"
    for suite in development stream_straight60; do
        args=(--precision-suite development)
        if [[ $suite == stream_straight60 ]]; then
            args=(--waypoint-source benchmark37.waypoint_stream:straight_half_metre)
        fi
        bash run_benchmark37_ring.sh eval --headless --checkpoint "$folder/model_49.pt" \
            "${args[@]}" --output "$folder/$suite" >"$folder/$suite.log" 2>&1
        verify "$folder/$suite" eval
    done
done
python3 -m benchmark37.ab_architecture_metrics "$pair"
