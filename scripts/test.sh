#!/usr/bin/env bash
set -Eeuo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"
specagent_need_client
cd "$SPECAGENT_REPO"
if [[ "${1:-}" == --full ]]; then
    shift
    exec "$CLIENT_PYTHON" -m pytest spectool/tests/spectool_agent sandbox_repo/tests tests \
        --continue-on-collection-errors "$@"
fi
# CPU-only release smoke suite. Historical tests remain accessible with --full.
exec "$CLIENT_PYTHON" -m pytest tests \
    spectool/tests/spectool_agent/test_model_parsers.py \
    spectool/tests/spectool_agent/test_tool_protocol.py \
    spectool/tests/spectool_agent/test_direct_alignment.py \
    spectool/tests/spectool_agent/test_speceyes.py \
    spectool/tests/spectool_agent/test_speceyes_token_budget.py \
    spectool/tests/spectool_agent/test_sandbox_execution_status.py \
    sandbox_repo/tests/test_runtime.py -q "$@"
