#!/usr/bin/env bash
set -Eeuo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"
exec python3 "$SPECAGENT_REPO/scripts/run_test.py" "$@"
