#!/usr/bin/env bash
set -Eeuo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"
source "$SPECAGENT_REPO/scripts/conda_init.sh"
server_check=false
gpus="${CUDA_VISIBLE_DEVICES:-}"
args=()
while (( $# )); do
    case "$1" in
        --server) server_check=true; args+=("$1"); shift ;;
        --gpus) [[ $# -ge 2 ]] || { printf 'Missing --gpus LIST\n' >&2; exit 2; }; gpus="$2"; shift 2 ;;
        *) args+=("$1"); shift ;;
    esac
done
if (( ${#args[@]} )); then set -- "${args[@]}"; else set --; fi
if [[ "$server_check" == true ]]; then
    [[ -n "$gpus" ]] || { printf 'Select a GPU for CUDA checks: doctor.sh --server --gpus 0\n' >&2; exit 2; }
    [[ -x "$SERVER_PYTHON" ]] || { printf 'Missing server environment. Run setup_env.sh --server-only.\n' >&2; exit 1; }
    specagent_activate_server
    export CUDA_VISIBLE_DEVICES="$gpus"
    exec "$SERVER_PYTHON" "$SPECAGENT_REPO/scripts/doctor.py" "$@"
fi
specagent_need_client
specagent_activate_client
export PATH="$(dirname "$CLIENT_PYTHON"):$CONDA_BASE/bin:$PATH"
exec "$CLIENT_PYTHON" "$SPECAGENT_REPO/scripts/doctor.py" "$@"
