#!/usr/bin/env bash
set -Eeuo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"
profile=all
dry_run=false
for arg in "$@"; do
    case "$arg" in
        --client-only) profile=client ;;
        --server-only) profile=server ;;
        --core-only) profile=core ;;
        --dry-run) dry_run=true ;;
        -h|--help)
            printf '%s\n' 'Usage: bash scripts/setup_env.sh [--client-only|--server-only|--core-only] [--dry-run]' \
                'Default: install full evaluation + vLLM environments on Linux x86_64.' \
                '--core-only installs the small dependency set for offline tests, not full benchmark evaluation.'
            exit 0 ;;
        *) printf 'Unknown argument: %s\n' "$arg" >&2; exit 2 ;;
    esac
done
run() {
    printf '+ '; printf '%q ' "$@"; printf '\n'
    if [[ "$dry_run" == false ]]; then "$@"; fi
}
if [[ "$dry_run" == false ]]; then
    [[ "$(uname -s)" == Linux && "$(uname -m)" == x86_64 ]] || {
        printf 'Automatic environment installation requires Linux x86_64. See docs/QUICKSTART.md.\n' >&2
        exit 2
    }
    specagent_mkdir
fi
installer="$SPECAGENT_WORK_ROOT/Miniconda3.sh"
if [[ ! -x "$CONDA_BASE/bin/conda" ]]; then
    run curl -fLsS --retry 3 \
        "${MINICONDA_URL:-https://repo.anaconda.com/miniconda/Miniconda3-py312_26.9.4-1-Linux-x86_64.sh}" -o "$installer"
    if [[ "$dry_run" == false ]]; then
        printf '%s  %s\n' 166b7a31c8526e558857a905735734836e5b7163fae86d0c377e0997171bd467 \
            "$installer" | sha256sum -c -
    fi
    run bash "$installer" -b -p "$CONDA_BASE"
fi
if [[ "$profile" != server ]]; then
    if [[ ! -x "$CLIENT_PYTHON" ]]; then
        run "$CONDA_BASE/bin/conda" create --override-channels -c conda-forge \
            -p "$CONDA_BASE/envs/spectool" python=3.10 pip -y
    fi
    if [[ "$profile" == core ]]; then requirements=core.txt; else requirements=client-linux.lock; fi
    run "$CLIENT_PYTHON" -m pip install -r "$SPECAGENT_REPO/requirements/$requirements"
    if [[ "$profile" != core ]]; then
        run "$CONDA_BASE/bin/conda" install --override-channels -c conda-forge \
            -p "$CONDA_BASE/envs/spectool" redis -y
        # Build metadata is resolved from the vendored directory, not caller CWD.
        if [[ "$dry_run" == false ]]; then cd "$SPECAGENT_REPO/spectool/VLMEvalKit"; fi
        run "$CLIENT_PYTHON" -m pip install -e "$SPECAGENT_REPO/spectool/VLMEvalKit" --no-deps --no-build-isolation
    fi
    run "$CLIENT_PYTHON" -m pip check
fi
if [[ "$profile" == all || "$profile" == server ]]; then
    if [[ ! -x "$SERVER_PYTHON" ]]; then
        run "$CONDA_BASE/bin/conda" create --override-channels -c conda-forge \
            -p "$CONDA_BASE/envs/vllm0251" python=3.12 pip -y
    fi
    wheel="$SPECAGENT_WORK_ROOT/vllm-0.25.1+cu129-cp38-abi3-manylinux_2_28_x86_64.whl"
    run curl -fLsS --retry 3 \
        https://github.com/vllm-project/vllm/releases/download/v0.25.1/vllm-0.25.1+cu129-cp38-abi3-manylinux_2_28_x86_64.whl \
        -o "$wheel"
    if [[ "$dry_run" == false ]]; then
        printf '%s  %s\n' 9e206f370c934a2d4b6b1f05d3d09708d344e05d80260189ef19f60755709431 \
            "$wheel" | sha256sum -c -
    fi
    run "$SERVER_PYTHON" -m pip install \
        --extra-index-url https://download.pytorch.org/whl/cu129 \
        -c "$SPECAGENT_REPO/requirements/vllm.constraints.txt" \
        "$wheel" transformers==5.14.1 huggingface-hub==1.24.0
    run "$SERVER_PYTHON" -m pip check
fi
printf 'Environment plan complete: %s (dry-run=%s)\n' "$profile" "$dry_run"
