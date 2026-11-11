#!/usr/bin/env bash
set -Eeuo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"
specagent_need_client
if [[ "${1:-}" == --check-only ]]; then
    shift
    exec "$CLIENT_PYTHON" "$SPECAGENT_REPO/scripts/check_models.py" --resource-dir "$RESOURCE_DIR" "$@"
fi
specagent_mkdir
unset HF_HUB_OFFLINE HF_DATASETS_OFFLINE TRANSFORMERS_OFFLINE
"$CLIENT_PYTHON" "$SPECAGENT_REPO/scripts/download_models.py" --resource-dir "$RESOURCE_DIR" "$@"
"$CLIENT_PYTHON" "$SPECAGENT_REPO/scripts/check_models.py" --resource-dir "$RESOURCE_DIR" "$@"
