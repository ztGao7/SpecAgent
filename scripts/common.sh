#!/usr/bin/env bash
# Shared paths only; sourcing this file never starts services or installs packages.
SPECAGENT_REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# Local configuration is trusted Bash syntax. Preserve exported caller overrides
# (including GPU flags forwarded by run_test.py) when this is sourced by children.
specagent_load_env() {
    local _sa_file="${SPECAGENT_ENV_FILE:-$SPECAGENT_REPO/.env}"
    local _sa_key _sa_i _sa_flags="$-" _sa_status=0
    local -a _sa_names=() _sa_values=()
    if [[ ! -f "$_sa_file" ]]; then
        if [[ -n "${SPECAGENT_ENV_FILE:-}" ]]; then
            printf 'Configuration file not found: %s\n' "$_sa_file" >&2
            return 1
        fi
        return 0
    fi
    while IFS= read -r _sa_key; do
        [[ "$_sa_key" =~ ^[a-zA-Z_][a-zA-Z_0-9]*$ ]] || continue
        _sa_names+=("$_sa_key")
        _sa_values+=("${!_sa_key}")
    done < <(compgen -e)
    set -a
    source "$_sa_file" || _sa_status=$?
    [[ "$_sa_flags" == *a* ]] || set +a
    for ((_sa_i=0; _sa_i<${#_sa_names[@]}; _sa_i++)); do
        printf -v "${_sa_names[$_sa_i]}" '%s' "${_sa_values[$_sa_i]}"
        export "${_sa_names[$_sa_i]}"
    done
    return "$_sa_status"
}
if [[ "${_SPECAGENT_ENV_LOADED:-}" != "$SPECAGENT_REPO" ]]; then
    specagent_load_env || return $?
    _SPECAGENT_ENV_LOADED="$SPECAGENT_REPO"
fi
export SPECAGENT_WORK_ROOT="${SPECAGENT_WORK_ROOT:-$SPECAGENT_REPO/.local}"
export RESOURCE_DIR="${RESOURCE_DIR:-$SPECAGENT_WORK_ROOT/resources}"
export CONDA_BASE="${CONDA_BASE:-$SPECAGENT_WORK_ROOT/miniconda3}"
export CLIENT_PYTHON="${CLIENT_PYTHON:-$CONDA_BASE/envs/spectool/bin/python}"
export SERVER_PYTHON="${SERVER_PYTHON:-$CONDA_BASE/envs/vllm0251/bin/python}"
# Installation and conda activation must use the same exact prefixes as Python.
export CLIENT_ENV_PREFIX="$(dirname "$(dirname "$CLIENT_PYTHON")")"
export SERVER_ENV_PREFIX="$(dirname "$(dirname "$SERVER_PYTHON")")"
for _sa_python in "$CLIENT_PYTHON" "$SERVER_PYTHON"; do
    [[ "$_sa_python" == /*/bin/python ]] || {
        printf 'Use an absolute Conda Python path ending in /bin/python: %s\n' "$_sa_python" >&2
        return 1
    }
done
unset _sa_python
export TMPDIR="${SPECAGENT_TMPDIR:-$SPECAGENT_WORK_ROOT/tmp}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$SPECAGENT_WORK_ROOT/cache}"
export TORCH_HOME="${TORCH_HOME:-$SPECAGENT_WORK_ROOT/cache/torch}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-$SPECAGENT_WORK_ROOT/cache/triton}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-$SPECAGENT_WORK_ROOT/cache/pip}"
export CONDA_PKGS_DIRS="${CONDA_PKGS_DIRS:-$SPECAGENT_WORK_ROOT/cache/conda}"
export HF_HOME="${HF_HOME:-$SPECAGENT_WORK_ROOT/cache/huggingface}"
export HF_HUB_ENABLE_HF_TRANSFER=0
export LMUData="${LMUData:-$RESOURCE_DIR/LMUData}"
export MAIN_MODEL_ID="${MAIN_MODEL_ID:-Qwen/Qwen3.5-27B}"
export DRAFT_MODEL_ID="${DRAFT_MODEL_ID:-Qwen/Qwen3.5-9B}"
export MAIN_MODEL_PATH="${MAIN_MODEL_PATH:-$RESOURCE_DIR/$MAIN_MODEL_ID}"
export DRAFT_MODEL_PATH="${DRAFT_MODEL_PATH:-$RESOURCE_DIR/$DRAFT_MODEL_ID}"

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
