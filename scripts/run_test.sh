#!/usr/bin/env bash
set -Eeuo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"
if [[ -x "$CLIENT_PYTHON" ]]; then
    exec "$CLIENT_PYTHON" "$SPECAGENT_REPO/scripts/run_test.py" "$@"
fi
# A plan can be inspected before installing Conda; real runs need the client.
for arg in "$@"; do
    if [[ "$arg" == --dry-run ]]; then
        exec python3 "$SPECAGENT_REPO/scripts/run_test.py" "$@"
    fi
done
specagent_need_client
