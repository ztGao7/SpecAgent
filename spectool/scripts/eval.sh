#!/usr/bin/env bash
# One-shot launcher: build local benchmark TSV (idempotent) + run VLMEvalKit
# against a remote vLLM serving Qwen3.5-27B through the Spectool-Agent wrapper.
#
# Usage:
#   bash scripts/eval.sh                       # uses defaults below
#   BENCH=mathvista bash scripts/eval.sh       # MathVista official MINI split
#   BENCH=mathvision_testmini bash scripts/eval.sh
#   SPECTOOL_VLLM_BASE=http://10.x.x.x:8123/v1 bash scripts/eval.sh

set -euo pipefail

# ── Resolve repo root from this script's location ──────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# ── Knobs (override via env) ────────────────────────────────────────────────
VLMEVAL_PY="${VLMEVAL_PY:-$(command -v python3 || command -v python)}"
[[ -n "$VLMEVAL_PY" && -x "$VLMEVAL_PY" ]] || { echo "ERROR: python not found (set VLMEVAL_PY)"; exit 1; }

SPECTOOL_VLLM_BASE="${SPECTOOL_VLLM_BASE:-http://localhost:8123/v1}"
SPECTOOL_VLLM_MODEL="${SPECTOOL_VLLM_MODEL:-Qwen3.5-27B}"
SPECTOOL_VLLM_KEY="${SPECTOOL_VLLM_KEY:-EMPTY}"
SPECTOOL_SANDBOX_URL="${SPECTOOL_SANDBOX_URL:-${CODE_SERVER_BASE:-http://127.0.0.1:18901/run_jupyter}}"
SPECTOOL_SANDBOX_SHARED_FS="${SPECTOOL_SANDBOX_SHARED_FS:-1}"
RESOURCE_DIR="${RESOURCE_DIR:-$REPO_ROOT/../resources}"
LMUData="${LMUData:-$RESOURCE_DIR/LMUData}"
MODEL="${MODEL:-Agent-Qwen3.5-27B}"
MODE="${MODE:-infer}"           # infer | all
API_NPROC="${API_NPROC:-1}"
BENCH="${BENCH:-vstar}"
WORK_DIR="${WORK_DIR:-$REPO_ROOT/outputs/spectool_${BENCH}}"
MAX_IMAGE_HISTORY="${MAX_IMAGE_HISTORY:-3}"
IMAGE_MAX_SIZE="${IMAGE_MAX_SIZE:-0}"
TOOL_IMAGE_MAX_SIZE="${TOOL_IMAGE_MAX_SIZE:-1536}"
IMAGE_MAX_PIXELS="${IMAGE_MAX_PIXELS:-12845056}"
IMAGE_RESIZE_FACTOR="${IMAGE_RESIZE_FACTOR:-32}"
IMAGE_MIN_PIXELS="${IMAGE_MIN_PIXELS:-0}"
IMAGE_RESIZE_POLICY="${IMAGE_RESIZE_POLICY:-downscale}"
IMAGE_FORMAT="${IMAGE_FORMAT:-JPEG}"
IMAGE_TRANSPORT="${IMAGE_TRANSPORT:-data}"
LOCAL_MEDIA_ROOT="${LOCAL_MEDIA_ROOT:-}"
IMAGE_QUALITY="${IMAGE_QUALITY:-90}"
SPECTOOL_OPENAI_MAX_RETRIES="${SPECTOOL_OPENAI_MAX_RETRIES:-0}"
SPECTOOL_API_TIMEOUT="${SPECTOOL_API_TIMEOUT:-180}"
SPECTOOL_API_RETRY="${SPECTOOL_API_RETRY:-1}"
EVAL_RETRY="${EVAL_RETRY:-$SPECTOOL_API_RETRY}"
SPECTOOL_MMSEARCH_TOOLS="${SPECTOOL_MMSEARCH_TOOLS:-0}"

# ── Map BENCH to VLMEvalKit DATA name and optional local build ──────────────
case "$BENCH" in
    vstar)
        DATA="${DATA:-VStarBench}"
        BUILD_SCRIPT="$REPO_ROOT/scripts/build_vstar_tsv.py"
        SRC_DIR="${SRC_DIR:-}"
        if [[ -z "$SRC_DIR" ]]; then
            if [[ -d "${REPO_ROOT}/data/vstar" ]]; then
                SRC_DIR="${REPO_ROOT}/data/vstar"
            elif [[ -d "${SPECTOOL_DATA_ROOT:-}/vstar" ]]; then
                SRC_DIR="${SPECTOOL_DATA_ROOT}/vstar"
            else
                SRC_DIR="${REPO_ROOT}/data/vstar"
            fi
        fi
        ;;
    mmsearch)
        DATA="${DATA:-MMSearchMini}"
        BUILD_SCRIPT="$REPO_ROOT/scripts/build_mmsearch_tsv.py"
        SRC_DIR="${MMSEARCH_DATA_ROOT:-$RESOURCE_DIR/benchmarks/MMSearch/end2end}"
        ;;
    mathvision)
        DATA="${DATA:-MathVision}"
        BUILD_SCRIPT=""  # full MathVision downloads from VLMEvalKit URL
        SRC_DIR=""       # not needed for auto-download
        ;;
    mathvista|mathvista_mini|MathVista_MINI)
        # MathVista is registered by VLMEvalKit as MathVista_MINI.  Keep
        # explicit aliases so it cannot be confused with MathVision.
        DATA="${DATA:-MathVista_MINI}"
        BUILD_SCRIPT=""  # reuse LMUData/MathVista_MINI.tsv or auto-download
        SRC_DIR=""       # not needed when the TSV already exists
        ;;
    mmstar)
        DATA="${DATA:-MMStar}"
        BUILD_SCRIPT=""  # MMStar downloads from VLMEvalKit URL (opencompass)
        SRC_DIR=""       # not needed for auto-download
        ;;
    pope)
        DATA="${DATA:-POPE}"
        BUILD_SCRIPT=""
        SRC_DIR=""
        ;;
    hrbench4k)
        DATA="${DATA:-HRBench4K}"
        BUILD_SCRIPT=""
        SRC_DIR=""
        ;;
    hrbench8k)
        DATA="${DATA:-HRBench8K}"
        BUILD_SCRIPT=""
        SRC_DIR=""
        ;;
    mathvision_testmini)
        DATA="${DATA:-MathVision_MINI}"
        BUILD_SCRIPT="$REPO_ROOT/scripts/build_mathvision_mini_tsv.py"
        SRC_DIR="${SRC_DIR:-}"
        if [[ -z "$SRC_DIR" ]]; then
            if [[ -d "${REPO_ROOT}/data/mathvision_testmini" ]]; then
                SRC_DIR="${REPO_ROOT}/data/mathvision_testmini"
            elif [[ -d "${SPECTOOL_DATA_ROOT:-}/mathvision_testmini" ]]; then
                SRC_DIR="${SPECTOOL_DATA_ROOT}/mathvision_testmini"
            else
                SRC_DIR="${REPO_ROOT}/data/mathvision_testmini"
            fi
        fi
        # IMG_ROOT for mathvision images (shared between full and testmini)
        IMG_ROOT="${IMG_ROOT:-}"
        if [[ -z "$IMG_ROOT" ]]; then
            if [[ -d "${REPO_ROOT}/data/mathvision" ]]; then
                IMG_ROOT="${REPO_ROOT}/data/mathvision"
            elif [[ -d "${SPECTOOL_DATA_ROOT:-}/mathvision" ]]; then
                IMG_ROOT="${SPECTOOL_DATA_ROOT}/mathvision"
            fi
        fi
        ;;
    *)
        echo "ERROR: Unknown benchmark: $BENCH"
        echo "Supported: vstar, mmsearch, mathvision, mathvista, mathvista_mini, mathvision_testmini, mmstar, pope, hrbench4k, hrbench8k"
        exit 1
        ;;
esac

export LMUData
export SPECTOOL_AGENT_ROOT="$REPO_ROOT"
export SPECTOOL_VLLM_BASE SPECTOOL_VLLM_MODEL SPECTOOL_VLLM_KEY SPECTOOL_SANDBOX_URL
export SPECTOOL_SANDBOX_SHARED_FS
export SPECTOOL_MAX_IMAGE_HISTORY="$MAX_IMAGE_HISTORY"
export SPECTOOL_IMAGE_MAX_SIZE="$IMAGE_MAX_SIZE"
export SPECTOOL_TOOL_IMAGE_MAX_SIZE="$TOOL_IMAGE_MAX_SIZE"
export SPECTOOL_IMAGE_MAX_PIXELS="$IMAGE_MAX_PIXELS"
export SPECTOOL_IMAGE_RESIZE_FACTOR="$IMAGE_RESIZE_FACTOR"
export SPECTOOL_IMAGE_MIN_PIXELS="$IMAGE_MIN_PIXELS"
export SPECTOOL_IMAGE_RESIZE_POLICY="$IMAGE_RESIZE_POLICY"
export SPECTOOL_IMAGE_FORMAT="$IMAGE_FORMAT"
export SPECTOOL_IMAGE_TRANSPORT="$IMAGE_TRANSPORT"
export SPECTOOL_LOCAL_MEDIA_ROOT="$LOCAL_MEDIA_ROOT"
export SPECTOOL_IMAGE_QUALITY="$IMAGE_QUALITY"
export SPECTOOL_API_RETRY="$EVAL_RETRY"
export SPECTOOL_OPENAI_MAX_RETRIES SPECTOOL_API_TIMEOUT SPECTOOL_API_RETRY
export SPECTOOL_MMSEARCH_TOOLS
# Hard-disable any accidental HF network access (datasets/hub/transformers)
export HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1

# ── Banner ──────────────────────────────────────────────────────────────────
cat <<EOF
 dataset         : $DATA
 vlmeval model   : $MODEL
 mode            : $MODE
 api-nproc       : $API_NPROC
 main timeout    : $SPECTOOL_API_TIMEOUT s
 draft timeout   : ${SPECTOOL_DRAFT_TIMEOUT:-200} s
 eval attempts   : $EVAL_RETRY
 SDK retries     : $SPECTOOL_OPENAI_MAX_RETRIES
 image transport : $IMAGE_TRANSPORT
 image max pixels: $IMAGE_MAX_PIXELS
 image resize    : $IMAGE_RESIZE_POLICY factor=$IMAGE_RESIZE_FACTOR min_pixels=$IMAGE_MIN_PIXELS format=$IMAGE_FORMAT
 work dir        : $WORK_DIR
EOF
[[ -d "$REPO_ROOT/VLMEvalKit" ]] || { echo "ERROR: VLMEvalKit dir missing: $REPO_ROOT/VLMEvalKit"; exit 1; }

mkdir -p "$LMUData" "$WORK_DIR"

# Silence the harmless .env loader log spam if no .env present
touch "$REPO_ROOT/VLMEvalKit/.env"

# ── 2. Build the local tsv if absent / source newer ─────────────────────────
TSV="$LMUData/$DATA.tsv"
if [[ -n "$BUILD_SCRIPT" ]]; then
    needs_build=false
    if [[ ! -f "$TSV" ]]; then
        needs_build=true
    elif [[ -d "$SRC_DIR" && -f "$SRC_DIR/metadata.json" && "$SRC_DIR/metadata.json" -nt "$TSV" ]]; then
        needs_build=true
    fi
    if [[ "$needs_build" == true ]]; then
        echo "[step] Building $TSV from $SRC_DIR ..."
        SRC_DIR="$SRC_DIR" LMUData="$LMUData" IMG_ROOT="${IMG_ROOT:-}" \
            "$VLMEVAL_PY" "$BUILD_SCRIPT"
    else
        echo "[step] Reusing existing $TSV (delete it to rebuild)"
    fi
else
    echo "[step] No local build script for $BENCH — VLMEvalKit may auto-download TSV"
fi

# ── 3. Sanity-check the wrapper is importable ───────────────────────────────
echo "[step] Verifying VLMEvalKit can see the wrapper ..."
MODEL="$MODEL" "$VLMEVAL_PY" - <<'PY'
import os, sys
sys.path.insert(0, os.path.join(os.environ["SPECTOOL_AGENT_ROOT"], "VLMEvalKit"))
from vlmeval.config import supported_VLM
model = os.environ["MODEL"]
assert model in supported_VLM, f"wrapper not registered: {model}"
print(f"ok — wrapper registered: {model}")
PY

# ── 4. Optionally probe the vLLM endpoint ───────────────────────────────────
echo "[step] Probing vLLM at $SPECTOOL_VLLM_BASE/models ..."
if curl -sf --max-time 5 "$SPECTOOL_VLLM_BASE/models" >/dev/null; then
    echo "ok — vLLM reachable"
else
    echo "WARNING: could not reach $SPECTOOL_VLLM_BASE/models (continuing anyway)"
fi

# ── 4b. Optionally probe the sandbox endpoint ────────────────────────────────
if [[ "$MODEL" == Qwen3.5-27B-vLLM ]]; then
    echo "[step] Direct answer: sandbox disabled"
elif curl -sf --max-time 5 -X POST \
    -H 'Content-Type: application/json' \
    -d '{"session_id":"spectool-probe","code":"print(1)","timeout":5}' \
    "$SPECTOOL_SANDBOX_URL" >/dev/null; then
    echo "ok — sandbox reachable"
else
    echo "WARNING: could not reach $SPECTOOL_SANDBOX_URL (continuing anyway)"
fi

# ── 5. Run ──────────────────────────────────────────────────────────────────
echo "[step] Launching run.py ..."
cd "$REPO_ROOT/VLMEvalKit"
START_TS=$(date +%s)
"$VLMEVAL_PY" run.py \
    --data "$DATA" \
    --model "$MODEL" \
    --mode "$MODE" \
    --api-nproc "$API_NPROC" \
    --retry "$EVAL_RETRY" \
    --timeout "$SPECTOOL_API_TIMEOUT" \
    --work-dir "$WORK_DIR" \
    --verbose
END_TS=$(date +%s)
ELAPSED=$((END_TS - START_TS))
printf "[elapsed] %02d:%02d:%02d (total %d seconds)\n" $((ELAPSED/3600)) $((ELAPSED%3600/60)) $((ELAPSED%60)) "$ELAPSED"

echo "[done] Outputs under $WORK_DIR"
