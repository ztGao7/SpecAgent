#!/usr/bin/env bash
# Sourced by start_test.sh. Does not change HOME or initialize shell profiles.
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"
if [[ ! -f "$CONDA_BASE/etc/profile.d/conda.sh" ]]; then
    printf 'Conda not found at %s; run bash scripts/setup_env.sh first.\n' "$CONDA_BASE" >&2
    return 1
fi
source "$CONDA_BASE/etc/profile.d/conda.sh"
