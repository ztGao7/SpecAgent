#!/usr/bin/env bash
set -Eeuo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"
if [[ "${1:-}" == --server ]]; then
    [[ -x "$SERVER_PYTHON" ]] || { printf 'Missing server environment. Run setup_env.sh --server-only.\n' >&2; exit 1; }
    exec "$SERVER_PYTHON" "$SPECAGENT_REPO/scripts/doctor.py" "$@"
fi
specagent_need_client
export PATH="$(dirname "$CLIENT_PYTHON"):$CONDA_BASE/bin:$PATH"
exec "$CLIENT_PYTHON" "$SPECAGENT_REPO/scripts/doctor.py" "$@"
