#!/usr/bin/env bash
set -Eeuo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$ROOT/../../scripts/common.sh"
specagent_need_client
# The PyArrow compatibility shim belongs only to model workers.
unset PYTHONPATH
export CUDA_VISIBLE_DEVICES=""
cd "$ROOT"
exec "$CLIENT_PYTHON" -m dreamr_adapter.run_eval "$@"
