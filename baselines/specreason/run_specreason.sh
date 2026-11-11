#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Activate your Python environment before running this script.
unset PYTHONPATH
export CUDA_VISIBLE_DEVICES=""
cd "$ROOT"
exec "${SPECREASON_PYTHON:-python}" -m specreason_adapter.run_eval "$@"
