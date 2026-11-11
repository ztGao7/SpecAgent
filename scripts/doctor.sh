#!/usr/bin/env bash
set -Eeuo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"
source "$SPECAGENT_REPO/scripts/conda_init.sh"
if [[ "${1:-}" == --server ]]; then
    [[ -x "$SERVER_PYTHON" ]] || { printf 'Missing server environment. Run setup_env.sh --server-only.\n' >&2; exit 1; }
    specagent_activate_server
    export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-${MAIN_GPUS:-0}}"
    exec "$SERVER_PYTHON" "$SPECAGENT_REPO/scripts/doctor.py" "$@"
fi
specagent_need_client
specagent_activate_client
export PATH="$(dirname "$CLIENT_PYTHON"):$CONDA_BASE/bin:$PATH"
exec "$CLIENT_PYTHON" "$SPECAGENT_REPO/scripts/doctor.py" "$@"
