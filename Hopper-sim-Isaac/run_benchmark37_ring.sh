#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
isaac_root=/home/terry/Documents/isaacsim/isaac-sim-standalone-5.1.0-linux-x86_64
lab_root="$(cd ../../IsaacLab && pwd)"
unset PYTHONEXE PYTHONHOME CONDA_PREFIX CONDA_DEFAULT_ENV
export PYTHONPATH="$isaac_root/exts/omni.isaac.ml_archive/pip_prebundle:$lab_root/source/isaaclab:$lab_root/source/isaaclab_rl"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONUNBUFFERED=1
if [[ "${1:-}" == log-precision-results ]]; then
  exec "$isaac_root/python.sh" -m benchmark37.precision_result_scalars
fi
# The precision trainer and its GUI use the same laptop GPU. Release the slot
# between stages so a queued preview can run without interrupting a checkpoint.
gpu_slot=()
case "${1:-}" in
  train|eval|eval-guarded-keys|play)
    for arg in "$@"; do
      if [[ "$arg" == *"/benchmark73_precision10_v50/"* ]]; then
        slot="$PWD/outputs/benchmark73_precision10_v50/isaac-gpu.lock"
        gpu_slot=(flock --exclusive --close "$slot")
        if [[ "$1" == play ]]; then
          echo '正在等待 Isaac GPU 时段；当前训练或评估结束后打开回放。'
        fi
        break
      fi
    done ;;
esac
if [[ "${1:-}" == eval-guarded-keys ]]; then
  shift
  exec "${gpu_slot[@]}" "$isaac_root/python.sh" -m benchmark37.guarded_keys_eval --headless "$@"
fi
if [[ "${1:-}" == distill ]]; then
  shift
  exec "$isaac_root/python.sh" -m benchmark37.obs73_distillation "$@"
fi
if [[ "${1:-}" == play ]]; then
  shift
  exec "${gpu_slot[@]}" "$isaac_root/python.sh" -m benchmark37.ring_play "$@"
fi
if [[ "${1:-}" == inspect-contacts ]]; then
  shift
  exec "$isaac_root/python.sh" -m benchmark37.ring_contact_inspection "$@"
fi
if [[ "${1:-}" == contact-check ]]; then
  shift
  exec "$isaac_root/python.sh" -m benchmark37.ring_contact_offline "$@"
fi
if [[ "${1:-}" == contacts ]]; then
  shift
  exec "$isaac_root/python.sh" -m benchmark37.ring_contact_diagnostic "$@"
fi
exec "${gpu_slot[@]}" "$isaac_root/python.sh" -m benchmark37.ring_runner "$@"
