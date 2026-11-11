#!/usr/bin/env bash
set -Eeuo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"
specagent_need_client
specagent_mkdir
exec "$CLIENT_PYTHON" "$SPECAGENT_REPO/scripts/download_data.py" --data-dir "$LMUData" "$@"
