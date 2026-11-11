#!/usr/bin/env bash
# Shared paths only; sourcing this file never starts services or installs packages.
SPECAGENT_REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export SPECAGENT_WORK_ROOT="${SPECAGENT_WORK_ROOT:-$SPECAGENT_REPO/.local}"
export RESOURCE_DIR="${RESOURCE_DIR:-$SPECAGENT_WORK_ROOT/resources}"
export CONDA_BASE="${CONDA_BASE:-$SPECAGENT_WORK_ROOT/miniconda3}"
export CLIENT_PYTHON="${CLIENT_PYTHON:-$CONDA_BASE/envs/spectool/bin/python}"
export SERVER_PYTHON="${SERVER_PYTHON:-$CONDA_BASE/envs/vllm0251/bin/python}"
export TMPDIR="${SPECAGENT_TMPDIR:-$SPECAGENT_WORK_ROOT/tmp}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$SPECAGENT_WORK_ROOT/cache}"
export TORCH_HOME="${TORCH_HOME:-$SPECAGENT_WORK_ROOT/cache/torch}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-$SPECAGENT_WORK_ROOT/cache/triton}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-$SPECAGENT_WORK_ROOT/cache/pip}"
export CONDA_PKGS_DIRS="${CONDA_PKGS_DIRS:-$SPECAGENT_WORK_ROOT/cache/conda}"
export HF_HOME="${HF_HOME:-$SPECAGENT_WORK_ROOT/cache/huggingface}"
export HF_HUB_ENABLE_HF_TRANSFER=0
export LMUData="${LMUData:-$RESOURCE_DIR/LMUData}"
export MAIN_MODEL_PATH="${MAIN_MODEL_PATH:-$RESOURCE_DIR/Qwen/Qwen3.5-27B}"
export DRAFT_MODEL_PATH="${DRAFT_MODEL_PATH:-$RESOURCE_DIR/Qwen/Qwen3.5-9B}"

specagent_mkdir() {
    mkdir -p "$SPECAGENT_WORK_ROOT" "$RESOURCE_DIR" "$TMPDIR" \
        "$PIP_CACHE_DIR" "$CONDA_PKGS_DIRS" "$HF_HOME" "$LMUData"
}

specagent_need_client() {
    if [[ ! -x "$CLIENT_PYTHON" ]]; then
        printf 'Missing client Python: %s\nRun: bash scripts/setup_env.sh --client-only\n' "$CLIENT_PYTHON" >&2
        return 1
    fi
}
