#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
mode="${1:-}"
if [[ -z "$mode" ]]; then
  echo 'Usage: bash run_benchmark73_quaternion_omni.sh train|eval|play|check [arguments]' >&2
  exit 2
fi
shift
case "$mode" in
  train|check)
    exec bash run_benchmark72_omni.sh "$mode" --ring-config benchmark37/configs/obs73_quaternion_omni_course_v5.json "$@"
    ;;
  eval|play) exec bash run_benchmark72_omni.sh "$mode" "$@" ;;
  *) echo 'Use train, eval, play or check; training is manually started by the user.' >&2; exit 2 ;;
esac
