#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
mode="${1:-}"
if [[ -z "$mode" ]]; then
  echo 'Usage: bash run_benchmark72_omni.sh train|eval|play|check [arguments]' >&2
  exit 2
fi
shift
case "$mode" in
  train) exec bash run_benchmark37_ring.sh train --ring-config benchmark37/configs/obs72_omni_course_v4.json "$@" ;;
  eval|play) exec bash run_benchmark37_ring.sh "$mode" "$@" ;;
  check)
    isaac_root=/home/terry/Documents/isaacsim/isaac-sim-standalone-5.1.0-linux-x86_64
    lab_root="$(cd ../../IsaacLab && pwd)"
    unset PYTHONEXE PYTHONHOME CONDA_PREFIX CONDA_DEFAULT_ENV
    export PYTHONPATH="$isaac_root/exts/omni.isaac.ml_archive/pip_prebundle:$lab_root/source/isaaclab:$lab_root/source/isaaclab_rl"
    export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
    exec "$isaac_root/python.sh" -m benchmark37.obs73_omni_startup --obs72 "$@"
    ;;
  *) echo 'Use train, eval, play or check (zero optimizer updates).' >&2; exit 2 ;;
esac
