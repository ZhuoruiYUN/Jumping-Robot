#!/usr/bin/env bash
# Use when the inherited proxy breaks NVIDIA asset HTTPS while direct access works.
# Changes apply only to this script and its children, never to the caller's shell.
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ISAAC_PYTHON="${ISAAC_PYTHON:-/home/terry/Documents/isaacsim/isaac-sim-standalone-5.1.0-linux-x86_64/python.sh}"
GROUND_URL='https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/5.1/Isaac/Environments/Grid/default_environment.usd'

if (( $# == 0 )); then
  echo "Usage: bash $0 CHECKPOINT [run_directional_eval.py options]" >&2
  exit 2
fi
if [[ ! -x "$ISAAC_PYTHON" ]]; then
  echo "Isaac Python launcher not executable: $ISAAC_PYTHON" >&2
  exit 2
fi

unset http_proxy https_proxy all_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY
unset CONDA_PREFIX CONDA_DEFAULT_ENV PYTHONHOME PYTHONPATH
echo '[ASSET-PREFLIGHT] Checking the original NVIDIA ground asset over direct HTTPS'
if ! curl --noproxy '*' --fail --silent --show-error --location \
    --connect-timeout 10 --max-time 30 --retry 1 \
    --output /dev/null "$GROUND_URL"; then
  echo '[ASSET-PREFLIGHT] Direct download failed; Isaac was not started.' >&2
  exit 1
fi
echo '[ASSET-PREFLIGHT] Download succeeded; using Isaac python.sh with proxy variables removed'
cd "$ROOT"
exec "$ISAAC_PYTHON" "$ROOT/run_directional_eval.py" "$@"
