#!/usr/bin/env bash
# Native 4090 launcher for Spectool: start vLLM, sandbox, then VLMEvalKit eval.

set -Eeuo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CODE_ROOT="${CODE_ROOT:-$REPO_ROOT}"
source "$CODE_ROOT/scripts/common.sh"
RESOURCE_DIR="${RESOURCE_DIR:-$CODE_ROOT/resources}"
CONDA_INIT="${CONDA_INIT:-$CODE_ROOT/scripts/conda_init.sh}"
VLLM_ENV="${VLLM_ENV:-vllm0251}"
VLLM_SERVER_PYTHONPATH="${VLLM_SERVER_PYTHONPATH:-$CODE_ROOT/compat/vllm_no_pyarrow}"
SPECTOOL_ROOT="${SPECTOOL_ROOT:-$CODE_ROOT/spectool}"
NUMA_EXEC="${NUMA_EXEC:-$SPECTOOL_ROOT/scripts/numa_exec.py}"
SANDBOX_ROOT="${SANDBOX_ROOT:-$CODE_ROOT/sandbox_repo}"
RUN_ID="${RUN_ID:-native_$(date +%Y%m%d_%H%M%S)_$$}"
RUN_DIR="${RUN_DIR:-}"
OUTPUT_DIR_CLI=""
TOKEN_ACCOUNTING_STARTED=false
RUN_PATHS_READY=false

BENCH="${BENCH:-vstar}"
MODE="${MODE:-infer}"
API_NPROC="${API_NPROC:-1}"
EVAL_RETRY="${EVAL_RETRY:-${SPECTOOL_API_RETRY:-1}}"
SPECTOOL_OPENAI_MAX_RETRIES="${SPECTOOL_OPENAI_MAX_RETRIES:-0}"
SPECTOOL_API_TIMEOUT="${SPECTOOL_API_TIMEOUT:-180}"
SPECTOOL_API_RETRY="${SPECTOOL_API_RETRY:-1}"
MAX_STEPS="${MAX_STEPS:-8}"
MAX_IMAGE_HISTORY="${MAX_IMAGE_HISTORY:-5}"
IMAGE_MAX_SIZE="${IMAGE_MAX_SIZE:-0}"
TOOL_IMAGE_MAX_SIZE="${TOOL_IMAGE_MAX_SIZE:-1536}"
IMAGE_MAX_PIXELS="${IMAGE_MAX_PIXELS:-12845056}"
IMAGE_RESIZE_FACTOR="${IMAGE_RESIZE_FACTOR:-32}"
IMAGE_MIN_PIXELS="${IMAGE_MIN_PIXELS:-0}"
IMAGE_RESIZE_POLICY="${IMAGE_RESIZE_POLICY:-downscale}"
IMAGE_FORMAT="${IMAGE_FORMAT:-JPEG}"
IMAGE_TRANSPORT="${IMAGE_TRANSPORT:-file}"
LOCAL_MEDIA_ROOT="${LOCAL_MEDIA_ROOT:-}"
QJ_MAX_IMAGES="${QJ_MAX_IMAGES:-15}"
QJ_IMAGE_MAX_SIZE="${QJ_IMAGE_MAX_SIZE:-0}"
SPECTOOL_SANDBOX_SHARED_FS="${SPECTOOL_SANDBOX_SHARED_FS:-1}"
QUALITY_JUDGE_ENABLED="${QUALITY_JUDGE_ENABLED:-true}"
QUALITY_JUDGE_THRESHOLD="${QUALITY_JUDGE_THRESHOLD:-0.7}"
FAST_ANSWER_ENABLED="${FAST_ANSWER_ENABLED:-true}"
FAST_ANSWER_HANDOFF_GRACE_SECONDS="${FAST_ANSWER_HANDOFF_GRACE_SECONDS:-1.0}"
ANSWERABILITY_THRESHOLD="${ANSWERABILITY_THRESHOLD:-0.95}"
FALLBACK_REUSE_VALIDATED_EVIDENCE="${FALLBACK_REUSE_VALIDATED_EVIDENCE:-false}"
SAMPLE_WALL_BUDGET_SECONDS="${SAMPLE_WALL_BUDGET_SECONDS:-0}"

MAIN_MODEL_PATH="${MAIN_MODEL_PATH:-$RESOURCE_DIR/Qwen/Qwen3.5-27B}"
DRAFT_MODEL_PATH="${DRAFT_MODEL_PATH:-$RESOURCE_DIR/Qwen/Qwen3.5-9B}"
# SpecEyes uses top-K draft logprobs for the original separability metric.
# This value is also passed to the 9B vLLM server in --speceyes mode.
SPECEYE_TOP_LOGPROBS="${SPECEYE_TOP_LOGPROBS:-64}"
SPECEYE_CONFIDENCE_THRESHOLD="${SPECEYE_CONFIDENCE_THRESHOLD:-0.98}"
# Official SpecEyes consumes six independent questions per Phase-I/Phase-II
# funnel.  This is deliberately distinct from EvalKit API worker concurrency.
SPECEYE_BATCH_SIZE="${SPECEYE_BATCH_SIZE:-6}"
# Official SpecEyes routing/preprocessing defaults. Fallback output tokens
# inherit Spectool's request default unless explicitly overridden in the env.
# Step count is shared through MAX_STEPS / --max-steps.
SPECEYE_ROUTER_MAX_TOKENS="${SPECEYE_ROUTER_MAX_TOKENS:-10}"
SPECEYE_DIRECT_MAX_TOKENS="${SPECEYE_DIRECT_MAX_TOKENS:-20}"
SPECEYE_IMAGE_RESIZE_FACTOR="${SPECEYE_IMAGE_RESIZE_FACTOR:-28}"
SPECEYE_IMAGE_MIN_PIXELS="${SPECEYE_IMAGE_MIN_PIXELS:-3136}"
SPECEYE_IMAGE_RESIZE_POLICY="${SPECEYE_IMAGE_RESIZE_POLICY:-speceyes}"
SPECEYE_IMAGE_FORMAT="${SPECEYE_IMAGE_FORMAT:-PNG}"
MAIN_PORT="${MAIN_PORT:-8126}"
DRAFT_PORT="${DRAFT_PORT:-8129}"
DRAFT_PORTS="${DRAFT_PORTS:-}"
SANDBOX_PORT="${SANDBOX_PORT:-18901}"
SANDBOX_REDIS_PORT="${SANDBOX_REDIS_PORT:-26379}"
SANDBOX_REDIS_PASSWORD="${SANDBOX_REDIS_PASSWORD:-}"
SANDBOX_WORKERS="${SANDBOX_WORKERS:-4}"
PORT_OFFSET="${PORT_OFFSET:-100}"
PORT_MAX_TRIES="${PORT_MAX_TRIES:-50}"
GPU_CHECK="${GPU_CHECK:-true}"
GPU_CHECK_STRICT="${GPU_CHECK_STRICT:-true}"
VLLM_NUMA_BIND="${VLLM_NUMA_BIND:-auto}"

MAIN_GPUS="${MAIN_GPUS:-}"
DRAFT_GPUS="${DRAFT_GPUS:-}"
MAIN_TP="${MAIN_TP:-1}"
DRAFT_TP="${DRAFT_TP:-1}"
DRAFT_REPLICAS="${DRAFT_REPLICAS:-1}"
MAIN_GPU_MEMORY_UTILIZATION="${MAIN_GPU_MEMORY_UTILIZATION:-0.9}"
DRAFT_GPU_MEMORY_UTILIZATION="${DRAFT_GPU_MEMORY_UTILIZATION:-0.8}"
# Resolve benchmark-dependent defaults after CLI parsing.
MAIN_MAX_MODEL_LEN="${MAIN_MAX_MODEL_LEN:-}"
DRAFT_MAX_MODEL_LEN="${DRAFT_MAX_MODEL_LEN:-}"
MODEL_MAX_LEN="${MODEL_MAX_LEN:-80000}"
MAIN_MAX_NUM_BATCHED_TOKENS="${MAIN_MAX_NUM_BATCHED_TOKENS:-32768}"
DRAFT_MAX_NUM_BATCHED_TOKENS="${DRAFT_MAX_NUM_BATCHED_TOKENS:-32768}"
LIMIT_MM_PER_PROMPT="${LIMIT_MM_PER_PROMPT:-{\"image\": 15}}"
# QJ and answerability requests share a long prompt prefix.
MAIN_EXTRA_ARGS="${MAIN_EXTRA_ARGS:---dtype float16 --max-num-seqs 8 --enable-prefix-caching}"
# Qwen3.5 is a hybrid model, so vLLM leaves prefix caching off unless asked;
# without it every Draft step re-prefills the whole growing history.
DRAFT_EXTRA_ARGS="${DRAFT_EXTRA_ARGS:---dtype float16 --max-num-seqs 8 --enable-prefix-caching}"


WAIT_RETRIES="${WAIT_RETRIES:-120}"
WAIT_INTERVAL="${WAIT_INTERVAL:-5}"
START_SERVICES=true
REUSE_SERVICES=false   # --reuse-services: use resident vLLM/sandbox, never start or stop them
MEDIA_PARENT="${MEDIA_PARENT:-}"   # --media-parent: file-transport media goes to MEDIA_PARENT/RUN_ID
INDEX_FILE="${INDEX_FILE:-}"       # --index-file: JSON {dataset: [index,...]} subset passed to EvalKit
RUN_EVAL=true
KILL_EXISTING=false
RUN_MODE="${RUN_MODE:-spectool}"  # spectool | speceyes | baseline | direct | draft-only
PORT_RESERVATION_PID=""
STARTED_VLLM_MAIN=false
STARTED_VLLM_DRAFT_SERVICES=()
STARTED_SANDBOX=false
SANDBOX_TMP_BASE=""
OWN_LOCAL_MEDIA_ROOT=false
DRAFT_GPU_GROUP_LIST=()
DRAFT_PORT_LIST=()
DRAFT_SERVICE_NAMES=()

if [[ "$VLLM_ENV" != "vllm0251" ]]; then
    printf 'ERROR: only the validated vllm0251 environment is allowed; got %q\n' "$VLLM_ENV" >&2
    exit 2
fi

usage() {
    printf '%s\n' \
        "Usage: $(basename "$0") [OPTIONS]" \
        "" \
        "Starts the local Spectool / baseline evaluation stack:" \
        "  1. vLLM main model" \
        "  2. optional vLLM draft model" \
        "  3. Redis + sandbox API" \
        "  4. spectool/scripts/eval.sh" \
        "" \
        "Modes:" \
        "  --spectool              Main 27B + draft 9B speculative eval (default)" \
        "  --dreamr                DREAM-R visual-tool adapter (Main + Draft)" \
        "  --speceyes               27B router + 9B confidence gate + 27B sandbox fallback" \
        "                          (router=10, draft=20, fallback=512 tokens; turns use --max-steps)" \
        "  --baseline" \
        "                          Main 27B standard Agent baseline" \
        "  --direct                Main 27B single-turn answer, no tools or sandbox" \
        "  --draft-only            9B standard Agent baseline" \
        "" \
        "Eval options mirror spectool/scripts/eval.sh:" \
        "  --bench NAME            vstar, mathvista, mathvision, mathvision_testmini, ..." \
        "  --mode MODE             infer or all (default: infer)" \
        "  --api-nproc N           Eval workers; for --speceyes, parallel tool-agent items per batch (default: 1)" \
        "  --speceye-batch-size N  Questions per official SpecEyes funnel (default: 6)" \
        "  --output-dir DIR        One directory for all files from this run" \
        "  --work-dir DIR          Alias for --output-dir (predictions go in DIR/predictions)" \
        "  --max-steps N           Unified Agent step limit for every mode and benchmark (default: 15)" \
        "  --max-image-history N   Spectool image history cap (default: 5)" \
        "  --image-max-size N      Input image max edge (0 disables, default: 0)" \
        "  --tool-image-max-size N Tool image max edge (default: 1536)" \
        "  --image-max-pixels N    Model-input image area cap (default: 12845056)" \
        "  --image-resize-factor N Align resized dimensions down to N (default: 32; --speceyes uses 28)" \
        "  --image-transport MODE  file (default) or data" \
        "  --local-media-root DIR  Run-local image cache for file transport" \
        "  --qj-max-images N       QJ images per request (default: 15)" \
        "  --qj-image-max-size N   QJ image max edge (0 disables, default: 0)" \
        "  --quality-judge         Enable quality judge (enabled by default)" \
        "  --no-quality-judge      Disable quality judge for ablation runs" \
        "  --quality-threshold N   P(GOOD) threshold for quality judge (default: 0.7)" \
        "  --fast-answer           Enable conservative early Main answer gate (default)" \
        "  --no-fast-answer        Disable the early Main answer gate" \
        "  --answerability-threshold N" \
        "                         Shared P(answerable) gate for initial and Draft-evidence answers (default: 0.95)" \
        "  --fast-answer-handoff-grace N" \
        "                         Wait up to N seconds for an in-flight initial fast answer before QJ (0-10; default: 1)" \
        "  --fallback-reuse-validated-evidence" \
        "                         Reuse only QJ-approved Draft tool turns in a clean Main fallback (default: off)" \
        "  --sample-wall-budget N  Whole-sample wall budget in seconds (0 disables)" \
        "" \
        "Service options:" \
        "  --main-port PORT        Main starting port (default: 8126; skips busy ports)" \
        "  --draft-port PORT       First Draft starting port (default: 8129)" \
        "  --draft-ports LIST      Explicit comma-separated Draft replica ports" \
        "  --sandbox-port PORT     Sandbox API port (default: 18901)" \
        "  --redis-port PORT       Sandbox Redis port (default: 26379)" \
        "  --port-offset N         Add N until an unused port is found (default: 100)" \
        "  --port-max-tries N      Max attempts per port family (default: 50)" \
        "  --main-gpus LIST        Main GPU IDs; required when starting Main" \
        "  --draft-gpus LIST       Draft GPU IDs; required when starting Draft. For TP=1 replicas," \
        "                          pass one GPU per replica, e.g. 4,5" \
        "  --main-tp N             Main tensor parallel size (default: 1)" \
        "  --draft-tp N            Tensor parallel size per Draft replica (default: 1)" \
        "  --draft-replicas N      Independent Draft services (default: 1)" \
        "  --main-max-num-batched-tokens N" \
        "                          Main vLLM encoder/prefill budget (default: 32768)" \
        "  --draft-max-num-batched-tokens N" \
        "                          Draft vLLM encoder/prefill budget (default: 32768)" \
        "  --numa-bind            Bind each vLLM service to its GPUs' NUMA node (default)" \
        "  --no-numa-bind         Disable automatic vLLM CPU/NUMA binding" \
        "  --no-gpu-check         Skip startup GPU memory checks" \
        "  --main-model PATH       Main model path" \
        "  --draft-model PATH      Draft model path" \
        "  --no-services           Run eval against already-running services" \
        "  --reuse-services        Like --no-services, but first checks that the given ports serve" \
        "                          the expected Main/Draft models and sandbox; never starts or stops them" \
        "  --media-parent DIR      With --image-transport file: put media in DIR/<run-id> (e.g. a" \
        "                          directory the resident vLLM allows), removed after the run" \
        "  --index-file FILE       Run only the listed samples: JSON {dataset_name: [index, ...]}" \
        "  --services-only         Start services and exit without eval" \
        "  --kill-existing         Stop services recorded in this RUN_DIR before starting" \
        "  --run-id ID             Run directory name under .runs/" \
        "" \
        "Examples:" \
        "  bash start_test.sh --spectool --bench vstar --main-gpus 0 --draft-gpus 1" \
        "  bash start_test.sh --baseline --main-gpus 0 --api-nproc 4" \
        "  bash start_test.sh --draft-only --draft-gpus 4 --draft-tp 1 --draft-port 8129" \
        "  bash start_test.sh --spectool --main-gpus 0 --draft-tp 1 --draft-replicas 2 --draft-gpus 4,5" \
        "  bash start_test.sh --no-services --spectool --main-port 8128 --draft-ports 8129,8130 --draft-replicas 2 --draft-gpus 4,5"
}

log() {
    printf '[%s] %s\n' "$(date '+%F %T')" "$*"
}

die() {
    printf '[%s] ERROR: %s\n' "$(date '+%F %T')" "$*" >&2
    exit 1
}

parse_args() {
    while [[ $# -gt 0 ]]; do
        case "$1" in
            --spectool) RUN_MODE=spectool; shift ;;
            --dreamr) RUN_MODE="spectool"; SPECAGENT_EVAL_DRIVER=dreamr; shift ;;
            --speceyes) RUN_MODE=speceyes; shift ;;
            --baseline) RUN_MODE=baseline; shift ;;
            --direct) RUN_MODE=direct; shift ;;
            --draft-only) RUN_MODE=draft-only; shift ;;
            --bench) BENCH="$2"; shift 2 ;;
            --mode) MODE="$2"; shift 2 ;;
            --api-nproc) API_NPROC="$2"; shift 2 ;;
            --speceye-batch-size) SPECEYE_BATCH_SIZE="$2"; shift 2 ;;
            --output-dir|--work-dir) OUTPUT_DIR_CLI="$2"; shift 2 ;;
            --max-steps) MAX_STEPS="$2"; shift 2 ;;
            --max-image-history) MAX_IMAGE_HISTORY="$2"; shift 2 ;;
            --image-max-size) IMAGE_MAX_SIZE="$2"; shift 2 ;;
            --tool-image-max-size) TOOL_IMAGE_MAX_SIZE="$2"; shift 2 ;;
            --image-max-pixels) IMAGE_MAX_PIXELS="$2"; shift 2 ;;
            --image-resize-factor) IMAGE_RESIZE_FACTOR="$2"; shift 2 ;;
            --image-transport) IMAGE_TRANSPORT="$2"; shift 2 ;;
            --local-media-root) LOCAL_MEDIA_ROOT="$2"; shift 2 ;;
            --qj-max-images) QJ_MAX_IMAGES="$2"; shift 2 ;;
            --qj-image-max-size) QJ_IMAGE_MAX_SIZE="$2"; shift 2 ;;
            --quality-judge) QUALITY_JUDGE_ENABLED=true; shift ;;
            --no-quality-judge) QUALITY_JUDGE_ENABLED=false; shift ;;
            --quality-threshold) QUALITY_JUDGE_THRESHOLD="$2"; shift 2 ;;
            --fast-answer) FAST_ANSWER_ENABLED=true; shift ;;
            --no-fast-answer) FAST_ANSWER_ENABLED=false; shift ;;
            --no-fast-answer-evidence-first) shift ;; # Legacy no-op; this gate has been removed.
            --fast-answer-evidence-first) die "The evidence-first gate has been removed" ;;
            --fast-answer-handoff-grace) FAST_ANSWER_HANDOFF_GRACE_SECONDS="$2"; shift 2 ;;
            --fallback-reuse-validated-evidence) FALLBACK_REUSE_VALIDATED_EVIDENCE=true; shift ;;
            --no-fallback-reuse-validated-evidence) FALLBACK_REUSE_VALIDATED_EVIDENCE=false; shift ;;
            --answerability-threshold)
                ANSWERABILITY_THRESHOLD="$2"
                shift 2
                ;;
            --sample-wall-budget) SAMPLE_WALL_BUDGET_SECONDS="$2"; shift 2 ;;
            --main-port) MAIN_PORT="$2"; shift 2 ;;
            --draft-port) DRAFT_PORT="$2"; shift 2 ;;
            --draft-ports) DRAFT_PORTS="$2"; shift 2 ;;
            --sandbox-port) SANDBOX_PORT="$2"; shift 2 ;;
            --redis-port) SANDBOX_REDIS_PORT="$2"; shift 2 ;;
            --port-offset) PORT_OFFSET="$2"; shift 2 ;;
            --port-max-tries) PORT_MAX_TRIES="$2"; shift 2 ;;
            --main-gpus) MAIN_GPUS="$2"; shift 2 ;;
            --draft-gpus) DRAFT_GPUS="$2"; shift 2 ;;
            --main-tp) MAIN_TP="$2"; shift 2 ;;
            --draft-tp) DRAFT_TP="$2"; shift 2 ;;
            --draft-replicas) DRAFT_REPLICAS="$2"; shift 2 ;;
            --main-max-num-batched-tokens) MAIN_MAX_NUM_BATCHED_TOKENS="$2"; shift 2 ;;
            --draft-max-num-batched-tokens) DRAFT_MAX_NUM_BATCHED_TOKENS="$2"; shift 2 ;;
            --numa-bind) VLLM_NUMA_BIND=auto; shift ;;
            --no-numa-bind) VLLM_NUMA_BIND=off; shift ;;
            --no-gpu-check) GPU_CHECK=false; shift ;;
            --main-model) MAIN_MODEL_PATH="$2"; shift 2 ;;
            --draft-model) DRAFT_MODEL_PATH="$2"; shift 2 ;;
            --no-services) START_SERVICES=false; shift ;;
            --reuse-services) START_SERVICES=false; REUSE_SERVICES=true; shift ;;
            --media-parent) MEDIA_PARENT="$2"; shift 2 ;;
            --index-file) INDEX_FILE="$2"; shift 2 ;;
            --services-only) RUN_EVAL=false; shift ;;
            --kill-existing) KILL_EXISTING=true; shift ;;
            --run-id)
                RUN_ID="$2"
                shift 2
                ;;
            -h|--help) usage; exit 0 ;;
            *) die "Unknown option: $1" ;;
        esac
    done
}

resolve_draft_layout() {
    local compact group port idx gpu gpu_count seen main_gpu
    local -a raw_groups=() raw_ports=() group_gpus=() main_gpus=() seen_gpus=()

    [[ "$DRAFT_REPLICAS" =~ ^[1-9][0-9]*$ ]] \
        || die "DRAFT_REPLICAS must be a positive integer: $DRAFT_REPLICAS"
    [[ "$DRAFT_TP" =~ ^[1-9][0-9]*$ ]] \
        || die "DRAFT_TP must be a positive integer: $DRAFT_TP"

    compact="${DRAFT_GPUS// /}"
    if [[ -z "$compact" && "$START_SERVICES" == false ]]; then
        # Resident endpoints need ports, not a GPU allocation.
        for ((idx=0; idx<DRAFT_REPLICAS; idx++)); do raw_groups+=(""); done
    elif [[ -z "$compact" ]]; then
        die "Specify --draft-gpus LIST when starting Draft (GPU IDs do not belong in .env)"
    elif (( DRAFT_REPLICAS == 1 )); then
        raw_groups=("$compact")
    elif (( DRAFT_TP == 1 )) && [[ "$compact" != *';'* ]]; then
        # Ergonomic form for two independent TP=1 replicas: --draft-gpus 4,5
        IFS=',' read -r -a raw_groups <<< "$compact"
    else
        # Multi-GPU replicas require explicit groups, e.g. '4,5;6,7'.
        IFS=';' read -r -a raw_groups <<< "$compact"
    fi
    (( ${#raw_groups[@]} == DRAFT_REPLICAS )) || die \
        "DRAFT_REPLICAS=$DRAFT_REPLICAS needs $DRAFT_REPLICAS GPU group(s); got '$DRAFT_GPUS'"

    DRAFT_GPU_GROUP_LIST=()
    for group in "${raw_groups[@]}"; do
        if [[ -z "$group" && "$START_SERVICES" == false ]]; then
            DRAFT_GPU_GROUP_LIST+=("")
            continue
        fi
        [[ -n "$group" ]] || die "Draft GPU groups must not be empty: $DRAFT_GPUS"
        [[ "$group" =~ ^[0-9]+(,[0-9]+)*$ ]] || die "Draft GPU group must be comma-separated GPU IDs: $group"
        IFS=',' read -r -a group_gpus <<< "$group"
        gpu_count="${#group_gpus[@]}"
        (( gpu_count == DRAFT_TP )) || die \
            "Draft replica GPU group '$group' has $gpu_count GPU(s), expected TP=$DRAFT_TP"
        for gpu in "${group_gpus[@]}"; do
            [[ "$gpu" =~ ^[0-9]+$ ]] || die "Draft GPU id must be numeric: $gpu"
            for seen in "${seen_gpus[@]:-}"; do
                [[ "$seen" != "$gpu" ]] || die \
                    "Draft GPU $gpu is assigned to more than one replica"
            done
            seen_gpus+=("$gpu")
        done
        DRAFT_GPU_GROUP_LIST+=("$group")
    done

    if [[ -n "$MAIN_GPUS" && -n "$DRAFT_GPUS" && ( "$RUN_MODE" == "spectool" || "$RUN_MODE" == "speceyes" ) ]]; then
        compact="${MAIN_GPUS// /}"
        [[ "$compact" =~ ^[0-9]+(,[0-9]+)*$ ]] || die "Main GPUs must be comma-separated GPU IDs: $MAIN_GPUS"
        IFS=',' read -r -a main_gpus <<< "$compact"
        for gpu in "${seen_gpus[@]}"; do
            for main_gpu in "${main_gpus[@]}"; do
                [[ "$gpu" != "$main_gpu" ]] || die \
                    "GPU $gpu is assigned to both Main and Draft"
            done
        done
    fi

    DRAFT_PORT_LIST=()
    if [[ -n "$DRAFT_PORTS" ]]; then
        compact="${DRAFT_PORTS// /}"
        IFS=',' read -r -a raw_ports <<< "$compact"
        (( ${#raw_ports[@]} == DRAFT_REPLICAS )) || die \
            "DRAFT_PORTS must contain one port per replica: $DRAFT_PORTS"
        DRAFT_PORT_LIST=("${raw_ports[@]}")
    else
        [[ "$DRAFT_PORT" =~ ^[0-9]+$ ]] || die "DRAFT_PORT must be numeric: $DRAFT_PORT"
        for ((idx=0; idx<DRAFT_REPLICAS; idx++)); do
            DRAFT_PORT_LIST+=("$((DRAFT_PORT + idx))")
        done
    fi
    for port in "${DRAFT_PORT_LIST[@]}"; do
        [[ "$port" =~ ^[0-9]+$ ]] || die "Draft port must be numeric: $port"
        (( port > 0 && port <= 65535 )) || die "Draft port out of range: $port"
    done

    DRAFT_SERVICE_NAMES=()
    for ((idx=0; idx<DRAFT_REPLICAS; idx++)); do
        if (( DRAFT_REPLICAS == 1 )); then
            DRAFT_SERVICE_NAMES+=("draft")
        else
            DRAFT_SERVICE_NAMES+=("draft_$((idx + 1))")
        fi
    done
    DRAFT_PORT="${DRAFT_PORT_LIST[0]}"
}

draft_base_urls() {
    local idx
    local -a urls=()
    for idx in "${!DRAFT_PORT_LIST[@]}"; do
        urls+=("http://127.0.0.1:${DRAFT_PORT_LIST[$idx]}/v1")
    done
    local IFS=','
    printf '%s\n' "${urls[*]}"
}

setup_paths() {
    [[ -f "$CONDA_INIT" ]] || die "Missing conda init: $CONDA_INIT"
    [[ -d "$SPECTOOL_ROOT" ]] || die "Missing Spectool root: $SPECTOOL_ROOT"
    if [[ "$RUN_MODE" != direct ]]; then
        [[ -d "$SANDBOX_ROOT" ]] || die "Missing sandbox root: $SANDBOX_ROOT"
    fi
    [[ -d "$MAIN_MODEL_PATH" ]] || die "Missing main model: $MAIN_MODEL_PATH"
    if [[ "$RUN_MODE" == "spectool" ]]; then
        [[ -d "$DRAFT_MODEL_PATH" ]] || die "Missing draft model: $DRAFT_MODEL_PATH"
    fi
    if [[ -n "$INDEX_FILE" ]]; then
        [[ -f "$INDEX_FILE" ]] || die "Missing index file: $INDEX_FILE"
        INDEX_FILE="$(readlink -f "$INDEX_FILE")"
    fi
    [[ ! -f "$RUN_DIR/run_info.json" ]] || die "Run directory already contains a previous run; choose a new --output-dir: $RUN_DIR"
    mkdir -p "$RUN_DIR/logs" "$RUN_DIR/pids"
    mkdir "$RUN_DIR/.run-owner" 2>/dev/null \
        || die "Run directory is already in use or initialized; choose a new --output-dir: $RUN_DIR"
    RUN_PATHS_READY=true
    RUN_DIR="$(cd "$RUN_DIR" && pwd -P)"
    WORK_DIR="$RUN_DIR/predictions"
    LOG_DIR="$RUN_DIR/logs"
    if [[ -n "$INDEX_FILE" ]]; then
        if [[ "$INDEX_FILE" != "$RUN_DIR/indices.json" && ! "$INDEX_FILE" -ef "$RUN_DIR/indices.json" ]]; then
            cp -- "$INDEX_FILE" "$RUN_DIR/indices.json"
        fi
        INDEX_FILE="$RUN_DIR/indices.json"
    fi
    case "$IMAGE_TRANSPORT" in
        file)
            if [[ -z "$LOCAL_MEDIA_ROOT" ]]; then
                LOCAL_MEDIA_ROOT="$(media_parent)/$RUN_ID"
                OWN_LOCAL_MEDIA_ROOT=true
            fi
            mkdir -p "$LOCAL_MEDIA_ROOT"
            ;;
        data) ;;
        *) die "IMAGE_TRANSPORT must be 'file' or 'data': $IMAGE_TRANSPORT" ;;
    esac
}

media_parent() {
    printf '%s' "${MEDIA_PARENT:-$RESOURCE_DIR/.tmp/spectool_media}"
}

# --reuse-services: the eval talks to resident services, so verify them first.
check_reused_services() {
    local url port
    url="http://127.0.0.1:${EVAL_MAIN_PORT}/v1/models"
    curl --noproxy '*' -fsS --max-time 10 "$url" | grep -qF "\"$EVAL_MAIN_MODEL\"" \
        || die "No resident Main model $EVAL_MAIN_MODEL at $url"
    log "Reusing Main $EVAL_MAIN_MODEL at port $EVAL_MAIN_PORT"
    if [[ "$RUN_MODE" == spectool || "$RUN_MODE" == speceyes ]]; then
        for port in "${DRAFT_PORT_LIST[@]}"; do
            url="http://127.0.0.1:${port}/v1/models"
            curl --noproxy '*' -fsS --max-time 10 "$url" | grep -qF "\"$DRAFT_MODEL_PATH\"" \
                || die "No resident Draft model $DRAFT_MODEL_PATH at $url"
            log "Reusing Draft $DRAFT_MODEL_PATH at port $port"
        done
    fi
    if [[ "$RUN_MODE" != direct ]]; then
        probe_sandbox "http://127.0.0.1:${SANDBOX_PORT}/run_jupyter" \
            || die "No resident sandbox at port $SANDBOX_PORT"
        log "Reusing sandbox at port $SANDBOX_PORT"
    fi
}

pid_file() {
    printf '%s/pids/%s.pid\n' "$RUN_DIR" "$1"
}

cleanup_pid_file() {
    printf '%s/pids/%s_cleanup.pid\n' "$RUN_DIR" "$1"
}

log_file() {
    printf '%s/logs/%s.log\n' "$RUN_DIR" "$1"
}

record_process_group() {
    local file="$1" pid="$2" pgid="" i
    for ((i=1; i<=50; i++)); do
        pgid="$(ps -o pgid= -p "$pid" 2>/dev/null | tr -d ' ' || true)"
        if [[ "$pgid" == "$pid" ]]; then
            printf '%s %s process-group-v1\n' "$pid" "$pgid" > "$file"
            return 0
        fi
        sleep 0.1
    done
    kill -TERM "$pid" 2>/dev/null || true
    die "Failed to isolate PID $pid in its own process group"
}

collect_descendant_pids() {
    local parent="$1" child
    while read -r child; do
        [[ -n "$child" ]] || continue
        collect_descendant_pids "$child"
        printf '%s\n' "$child"
    done < <(pgrep -P "$parent" 2>/dev/null || true)
}

process_group_has_live_members() {
    local pgid="$1"
    ps -eo pgid=,stat= | awk -v expected="$pgid" \
        '$1 == expected && $2 !~ /^Z/ { found=1 } END { exit !found }'
}

pid_list_has_live_members() {
    local pid stat
    for pid in "$@"; do
        stat="$(ps -o stat= -p "$pid" 2>/dev/null | tr -d ' ' || true)"
        if [[ -n "$stat" && "$stat" != Z* ]]; then
            return 0
        fi
    done
    return 1
}

stop_legacy_pid_tree() {
    local pid="$1" i target
    local -a descendants=() targets=()
    kill -0 "$pid" 2>/dev/null || return 0
    mapfile -t descendants < <(collect_descendant_pids "$pid")
    targets=("${descendants[@]}" "$pid")
    kill -TERM "${targets[@]}" 2>/dev/null || true
    for ((i=1; i<=20; i++)); do
        local alive=false
        for target in "${targets[@]}"; do
            if kill -0 "$target" 2>/dev/null; then
                alive=true
                break
            fi
        done
        [[ "$alive" == false ]] && return 0
        sleep 1
    done
    for target in "${targets[@]}"; do
        kill -KILL "$target" 2>/dev/null || true
    done
}

stop_pid_file() {
    local file="$1" pid="" pgid="" marker="" current_pgid="" i
    local -a descendants=()
    [[ -f "$file" ]] || return 0
    read -r pid pgid marker < "$file" || true
    if [[ "$marker" == "process-group-v1" && "$pid" =~ ^[0-9]+$ \
            && "$pgid" =~ ^[0-9]+$ && "$pid" == "$pgid" && "$pgid" -gt 1 ]]; then
        current_pgid="$(ps -o pgid= -p "$$" 2>/dev/null | tr -d ' ' || true)"
        if [[ "$pgid" == "$current_pgid" ]]; then
            log "WARNING: refusing to stop current process group $pgid from $file"
            return 1
        fi
        if kill -0 "$pid" 2>/dev/null; then
            mapfile -t descendants < <(collect_descendant_pids "$pid")
        fi
        if process_group_has_live_members "$pgid"; then
            log "Stopping process group $pgid from $file"
            kill -TERM -- "-$pgid" 2>/dev/null || true
        fi
        if [[ "${#descendants[@]}" -gt 0 ]]; then
            kill -TERM "${descendants[@]}" 2>/dev/null || true
        fi
        for ((i=1; i<=20; i++)); do
            if ! process_group_has_live_members "$pgid" \
                    && ! pid_list_has_live_members "${descendants[@]}"; then
                break
            fi
            sleep 1
        done
        if process_group_has_live_members "$pgid"; then
            log "Process group $pgid still alive after TERM; sending KILL"
            kill -KILL -- "-$pgid" 2>/dev/null || true
        fi
        if [[ "${#descendants[@]}" -gt 0 ]] \
                && pid_list_has_live_members "${descendants[@]}"; then
            log "Some descendants of process group $pgid are still alive; sending KILL"
            kill -KILL "${descendants[@]}" 2>/dev/null || true
        fi
        wait "$pid" 2>/dev/null || true
    elif [[ "$pid" =~ ^[0-9]+$ ]]; then
        log "Stopping legacy PID tree rooted at $pid from $file"
        stop_legacy_pid_tree "$pid"
    else
        log "WARNING: invalid PID metadata in $file; leaving it untouched"
        return 1
    fi
    rm -f "$file"
}

cleanup_vllm_ipc_dirs() {
    local record path service
    for record in "$RUN_DIR"/pids/vllm_*.ipc_dir; do
        [[ -f "$record" ]] || continue
        IFS= read -r path < "$record" || true
        service="${record##*/vllm_}"
        service="${service%.ipc_dir}"
        if "$SERVER_PYTHON" "$CODE_ROOT/scripts/vllm_ipc.py" cleanup \
                --path "$path" --tag "$RUN_ID-$service"; then
            rm -f "$record"
        else
            log "WARNING: could not clean private IPC directory from $record"
        fi
    done
}

stop_recorded_services() {
    local name
    stop_pid_file "$(pid_file eval)"
    # Preserve cleanup compatibility with pre-replica runs.
    stop_pid_file "$(pid_file vllm_draft)"
    if (( ${#DRAFT_SERVICE_NAMES[@]} > 0 )); then
        for name in "${DRAFT_SERVICE_NAMES[@]}"; do
            [[ "$name" == "draft" ]] && continue
            stop_pid_file "$(pid_file "vllm_${name}")"
        done
    fi
    stop_pid_file "$(pid_file vllm_main)"
    stop_pid_file "$(pid_file sandbox)"
    cleanup_vllm_ipc_dirs
}

cancel_cleanup_job() {
    local file="$1" pid
    [[ -f "$file" ]] || return 0
    pid="$(tr -cd '0-9' < "$file")"
    if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then
        log "Cancelling pending cleanup PID $pid from $file"
        kill -TERM "$pid" 2>/dev/null || true
        wait "$pid" 2>/dev/null || true
    fi
    rm -f "$file"
}

cancel_pending_vllm_cleanups() {
    local service="$1" run_dir file
    [[ -d "$CODE_ROOT/.runs" ]] || return 0
    for run_dir in "$CODE_ROOT"/.runs/*; do
        [[ -d "$run_dir/pids" ]] || continue
        file="$run_dir/pids/${service}_cleanup.pid"
        cancel_cleanup_job "$file"
    done
}

cleanup_sandbox_temp_dir() {
    local tmp_root expected
    [[ -n "$SANDBOX_TMP_BASE" ]] || return 0
    tmp_root="${TMPDIR:-$RESOURCE_DIR/.tmp}"
    expected="$tmp_root/spectool_sandbox_${RUN_ID}_${SANDBOX_PORT}"
    if [[ "$RUN_ID" == *"/"* || "$RUN_ID" == *".."* \
            || "$SANDBOX_TMP_BASE" != "$expected" \
            || "$SANDBOX_TMP_BASE" != "$tmp_root"/spectool_sandbox_* ]]; then
        log "WARNING: refusing to remove unexpected sandbox temp path: $SANDBOX_TMP_BASE"
        return 1
    fi
    if [[ -d "$SANDBOX_TMP_BASE" ]]; then
        log "Removing sandbox temp directory $SANDBOX_TMP_BASE"
        rm -rf -- "$SANDBOX_TMP_BASE"
    fi
}

cleanup_local_media_dir() {
    local expected
    [[ "$IMAGE_TRANSPORT" == "file" ]] || return 0
    [[ "$OWN_LOCAL_MEDIA_ROOT" == true ]] || return 0
    expected="$(media_parent)/$RUN_ID"
    if [[ -z "$LOCAL_MEDIA_ROOT" || -z "$RUN_ID" || "$LOCAL_MEDIA_ROOT" != "$expected" \
            || "$LOCAL_MEDIA_ROOT" != "$(media_parent)"/* ]]; then
        log "WARNING: refusing to remove unexpected media cache path: $LOCAL_MEDIA_ROOT"
        return 1
    fi
    if [[ -d "$LOCAL_MEDIA_ROOT" ]]; then
        log "Removing model media cache $LOCAL_MEDIA_ROOT"
        rm -rf -- "$LOCAL_MEDIA_ROOT"
    fi
}

cleanup_services_on_exit() {
    local status=$? name
    trap - EXIT INT TERM
    set +e
    if [[ "$RUN_PATHS_READY" == true ]]; then stop_pid_file "$(pid_file eval)"; fi
    if [[ "$TOKEN_ACCOUNTING_STARTED" == true ]]; then
        bench_token_snapshot after "$status" >>"$RUN_DIR/logs/tokens.log" 2>&1 \
            || log "Token summary failed; see $RUN_DIR/logs/tokens.log and metrics/"
        log "Results retained in $RUN_DIR (token_usage.json contains bench totals)"
    fi
    if [[ "$RUN_PATHS_READY" == true && "$START_SERVICES" == true && "$RUN_EVAL" == true ]]; then
        log "Stopping services immediately for run $RUN_ID"
        if (( ${#STARTED_VLLM_DRAFT_SERVICES[@]} > 0 )); then
            for name in "${STARTED_VLLM_DRAFT_SERVICES[@]}"; do
                stop_pid_file "$(pid_file "vllm_${name}")"
            done
        fi
        if [[ "$STARTED_VLLM_MAIN" == true ]]; then
            stop_pid_file "$(pid_file vllm_main)"
        fi
        if [[ "$STARTED_SANDBOX" == true ]]; then
            stop_pid_file "$(pid_file sandbox)"
            cleanup_sandbox_temp_dir
        fi
        cleanup_vllm_ipc_dirs
        log "Services stopped for run $RUN_ID"
    fi
    if [[ "$RUN_EVAL" == true ]]; then
        cleanup_local_media_dir
    fi
    release_port_reservations
    exit "$status"
}

probe_http() {
    curl --noproxy '*' -fsS --max-time 5 "$1" >/dev/null
}

probe_sandbox() {
    curl --noproxy '*' -fsS --max-time 10 \
        -H 'Content-Type: application/json' \
        -d '{"session_id":"spectool-native-probe","code":"print(1)","timeout":5}' \
        "$1" >/dev/null
}

wait_for() {
    local label="$1"
    shift
    local i
    for ((i=1; i<=WAIT_RETRIES; i++)); do
        if "$@"; then
            log "$label ready"
            return 0
        fi
        sleep "$WAIT_INTERVAL"
    done
    return 1
}

port_is_listening() {
    local port="$1"
    if command -v ss >/dev/null 2>&1; then
        ss -ltn "sport = :$port" 2>/dev/null | grep -q LISTEN
        return $?
    fi
    if command -v lsof >/dev/null 2>&1; then
        lsof -nP -iTCP:"$port" -sTCP:LISTEN >/dev/null 2>&1
        return $?
    fi
    if command -v netstat >/dev/null 2>&1; then
        netstat -ltn 2>/dev/null | awk '{print $4}' | grep -Eq "[:.]${port}$"
        return $?
    fi
    (echo >/dev/tcp/127.0.0.1/"$port") >/dev/null 2>&1
}

release_port_reservations() {
    if [[ -n "$PORT_RESERVATION_PID" ]]; then
        kill -TERM "$PORT_RESERVATION_PID" 2>/dev/null || true
        wait "$PORT_RESERVATION_PID" 2>/dev/null || true
        PORT_RESERVATION_PID=""
    fi
}

resolve_ports() {
    local idx port name old ready_file i
    local -a requests=(--port "main=$MAIN_PORT")
    [[ "$PORT_OFFSET" =~ ^[1-9][0-9]*$ ]] || die "PORT_OFFSET must be a positive integer: $PORT_OFFSET"
    [[ "$PORT_MAX_TRIES" =~ ^[1-9][0-9]*$ ]] || die "PORT_MAX_TRIES must be a positive integer: $PORT_MAX_TRIES"
    if [[ "$RUN_MODE" != direct ]]; then
        requests+=(--port "sandbox=$SANDBOX_PORT" --port "redis=$SANDBOX_REDIS_PORT")
    fi
    if [[ "$START_DRAFT" == true ]]; then
        for idx in "${!DRAFT_PORT_LIST[@]}"; do
            requests+=(--port "${DRAFT_SERVICE_NAMES[$idx]}=${DRAFT_PORT_LIST[$idx]}")
        done
    fi
    ready_file="$RUN_DIR/pids/ports.ready"
    rm -f -- "$ready_file"
    "$CLIENT_PYTHON" "$CODE_ROOT/scripts/service_ports.py" \
        "${requests[@]}" --offset "$PORT_OFFSET" --tries "$PORT_MAX_TRIES" \
        --parent "$$" --output "$ready_file" >"$RUN_DIR/logs/ports.log" 2>&1 &
    PORT_RESERVATION_PID=$!
    for ((i=0; i<200; i++)); do
        [[ ! -f "$ready_file" ]] || break
        kill -0 "$PORT_RESERVATION_PID" 2>/dev/null \
            || die "Port allocation failed; see $RUN_DIR/logs/ports.log"
        sleep 0.1
    done
    [[ -f "$ready_file" ]] || die "Port allocation timed out; see $RUN_DIR/logs/ports.log"
    while read -r name port; do
        case "$name" in
            main) old="$MAIN_PORT"; MAIN_PORT="$port" ;;
            sandbox) old="$SANDBOX_PORT"; SANDBOX_PORT="$port" ;;
            redis) old="$SANDBOX_REDIS_PORT"; SANDBOX_REDIS_PORT="$port" ;;
            *)
                for idx in "${!DRAFT_SERVICE_NAMES[@]}"; do
                    if [[ "$name" == "${DRAFT_SERVICE_NAMES[$idx]}" ]]; then
                        old="${DRAFT_PORT_LIST[$idx]}"
                        DRAFT_PORT_LIST[$idx]="$port"
                        break
                    fi
                done
                ;;
        esac
        [[ "$old" == "$port" ]] || log "$name port $old is busy/reserved; using $port"
    done < "$ready_file"
    if [[ "$START_DRAFT" == true ]]; then DRAFT_PORT="${DRAFT_PORT_LIST[0]}"; fi
    EVAL_MAIN_PORT="$MAIN_PORT"
}

print_gpu_snapshot() {
    if [[ "$GPU_CHECK" != true ]]; then
        log "GPU check disabled"
        return 0
    fi
    if ! command -v nvidia-smi >/dev/null 2>&1; then
        die "nvidia-smi not found; use --no-gpu-check to skip GPU checks"
    fi
    log "GPU memory snapshot before vLLM startup:"
    nvidia-smi --query-gpu=index,name,memory.used,memory.free,memory.total \
        --format=csv,noheader,nounits \
        | sed 's/^/[GPU] /'
}

check_gpu_group() {
    local label="$1" gpus="$2" mem_util="$3" tp="$4"
    local compact gpu count line free total required status
    [[ "$GPU_CHECK" == true ]] || return 0
    compact="${gpus// /}"
    [[ -n "$compact" ]] || die "$label GPU list is empty"
    gpu_array=()
    while [[ "$compact" == *,* ]]; do
        gpu_array+=("${compact%%,*}")
        compact="${compact#*,}"
    done
    gpu_array+=("$compact")
    count="${#gpu_array[@]}"
    if [[ "$tp" =~ ^[0-9]+$ && "$count" -ne "$tp" ]]; then
        die "$label TP=$tp but GPU list '$gpus' has $count GPU(s)"
    fi
    status=0
    for gpu in "${gpu_array[@]}"; do
        [[ "$gpu" =~ ^[0-9]+$ ]] || die "$label GPU id must be numeric: $gpu"
        line="$(nvidia-smi -i "$gpu" --query-gpu=memory.free,memory.total --format=csv,noheader,nounits 2>/dev/null || true)"
        [[ -n "$line" ]] || die "$label GPU $gpu is not visible to nvidia-smi"
        free="${line%%,*}"
        total="${line##*,}"
        free="${free// /}"
        total="${total// /}"
        required="$(awk -v total="$total" -v util="$mem_util" 'BEGIN { printf "%d", (total * util) + 0.999 }')"
        log "$label GPU $gpu memory: free=${free}MiB total=${total}MiB required~=${required}MiB util=${mem_util}"
        if (( free < required )); then
            status=1
        fi
    done
    if (( status != 0 )); then
        if [[ "$GPU_CHECK_STRICT" == true ]]; then
            die "$label requested GPUs '$gpus' do not have enough free memory for gpu_memory_utilization=$mem_util"
        fi
        log "WARNING: $label requested GPUs '$gpus' look low on memory, continuing because GPU_CHECK_STRICT=false"
    fi
}

resolve_gpu_numa_binding() {
    local gpus="$1" compact topology gpu bus sys_bus node first_node="" cpus
    local -a gpu_array=()
    compact="${gpus// /}"
    IFS=',' read -r -a gpu_array <<< "$compact"
    topology="$(nvidia-smi --query-gpu=index,pci.bus_id \
        --format=csv,noheader,nounits 2>/dev/null || true)"
    [[ -n "$topology" ]] || { printf 'unknown|nvidia-smi topology query failed\n'; return 0; }

    for gpu in "${gpu_array[@]}"; do
        bus="$(awk -F, -v target="$gpu" '
            { gsub(/ /, "", $1); if ($1 == target) { gsub(/ /, "", $2); print $2; exit } }
        ' <<< "$topology")"
        [[ -n "$bus" ]] || { printf 'unknown|GPU %s PCI address not found\n' "$gpu"; return 0; }
        # nvidia-smi uses an 8-digit PCI domain while sysfs uses 4 digits.
        if [[ "$bus" =~ ^[[:xdigit:]]{8}: ]]; then
            bus="${bus:4}"
        fi
        sys_bus="${bus,,}"
        [[ -r "/sys/bus/pci/devices/$sys_bus/numa_node" ]] \
            || { printf 'unknown|GPU %s NUMA sysfs entry missing\n' "$gpu"; return 0; }
        node="$(<"/sys/bus/pci/devices/$sys_bus/numa_node")"
        [[ "$node" =~ ^[0-9]+$ ]] \
            || { printf 'unknown|GPU %s has no NUMA node\n' "$gpu"; return 0; }
        if [[ -z "$first_node" ]]; then
            first_node="$node"
        elif [[ "$node" != "$first_node" ]]; then
            printf 'mixed|GPUs %s span NUMA nodes %s and %s\n' \
                "$gpus" "$first_node" "$node"
            return 0
        fi
    done

    [[ -r "/sys/devices/system/node/node${first_node}/cpulist" ]] \
        || { printf 'unknown|NUMA node %s CPU list missing\n' "$first_node"; return 0; }
    cpus="$(<"/sys/devices/system/node/node${first_node}/cpulist")"
    [[ -n "$cpus" ]] \
        || { printf 'unknown|NUMA node %s CPU list is empty\n' "$first_node"; return 0; }
    printf 'ok|%s|%s\n' "$first_node" "$cpus"
}

start_vllm() {
    local name="$1" gpus="$2" model_path="$3" port="$4" tp="$5" mem="$6" max_len="$7" max_batched_tokens="$8" extra="$9"
    local lf pf service_pid numa_info numa_status numa_node numa_cpus="" ipc_dir
    lf="$(log_file "vllm_${name}")"
    pf="$(pid_file "vllm_${name}")"
    ! port_is_listening "$port" || die "Selected $name port $port was taken before startup; retry the launcher"
    check_gpu_group "vLLM $name" "$gpus" "$mem" "$tp"
    if [[ "$VLLM_NUMA_BIND" == "auto" ]]; then
        if [[ ! -r "$NUMA_EXEC" ]]; then
            log "WARNING: NUMA launcher is unavailable at $NUMA_EXEC; vLLM $name NUMA binding disabled"
        else
            numa_info="$(resolve_gpu_numa_binding "$gpus")"
            IFS='|' read -r numa_status numa_node numa_cpus <<< "$numa_info"
            case "$numa_status" in
                ok)
                    log "vLLM $name NUMA binding: node=$numa_node CPUs=$numa_cpus"
                    ;;
                mixed)
                    log "WARNING: $numa_node; vLLM $name NUMA binding disabled. Use GPUs from one NUMA node for TP."
                    numa_node=""
                    numa_cpus=""
                    ;;
                *)
                    log "WARNING: cannot resolve vLLM $name NUMA binding: $numa_node"
                    numa_node=""
                    numa_cpus=""
                    ;;
            esac
        fi
    elif [[ "$VLLM_NUMA_BIND" != "off" ]]; then
        die "VLLM_NUMA_BIND must be 'auto' or 'off', got '$VLLM_NUMA_BIND'"
    else
        log "vLLM $name NUMA binding disabled"
    fi
    ipc_dir="$("$SERVER_PYTHON" "$CODE_ROOT/scripts/vllm_ipc.py" create \
        --tmp-root "$TMPDIR" --tag "$RUN_ID-$name" --root "${SPECAGENT_IPC_ROOT:-}")"
    printf '%s\n' "$ipc_dir" > "$RUN_DIR/pids/vllm_${name}.ipc_dir"
    log "Starting vLLM $name on port $port, GPUs=$gpus, TP=$tp, IPC=$ipc_dir"
    (
        local -a extra_args vllm_cmd
        source "$CONDA_INIT"
        specagent_activate_server
        export VLLM_RPC_BASE_PATH="$ipc_dir"
        export VLLM_USE_FLASHINFER_SAMPLER="${VLLM_USE_FLASHINFER_SAMPLER:-0}"
        export VLLM_REGEX_COMPILATION_TIMEOUT_S="${VLLM_REGEX_COMPILATION_TIMEOUT_S:-60}"
        if [[ -d "$VLLM_SERVER_PYTHONPATH" ]]; then
            if [[ -n "${PYTHONPATH:-}" ]]; then
                export PYTHONPATH="$VLLM_SERVER_PYTHONPATH:$PYTHONPATH"
            else
                export PYTHONPATH="$VLLM_SERVER_PYTHONPATH"
            fi
        fi
        cd "$CODE_ROOT"
        export CUDA_VISIBLE_DEVICES="$gpus"
        read -r -a extra_args <<< "$extra"
        vllm_cmd=(vllm serve "$model_path" \
            --host 127.0.0.1 \
            --port "$port" \
            --tensor-parallel-size "$tp" \
            --gpu-memory-utilization "$mem" \
            --max-model-len "$max_len" \
            --max-num-batched-tokens "$max_batched_tokens" \
            --trust-remote-code \
            --limit-mm-per-prompt "$LIMIT_MM_PER_PROMPT" \
            --reasoning-parser qwen3 \
            --enable-auto-tool-choice \
            --tool-call-parser qwen3_coder)
        if [[ "$IMAGE_TRANSPORT" == "file" ]]; then
            vllm_cmd+=(--allowed-local-media-path "$LOCAL_MEDIA_ROOT")
        fi
        vllm_cmd+=("${extra_args[@]}")
        if [[ -n "$numa_node" ]]; then
            exec setsid python "$NUMA_EXEC" "$numa_node" "${vllm_cmd[@]}"
        fi
        exec setsid "${vllm_cmd[@]}"
    ) >"$lf" 2>&1 < /dev/null &
    service_pid="$!"
    record_process_group "$pf" "$service_pid"
    if [[ "$name" == "main" ]]; then
        STARTED_VLLM_MAIN=true
    else
        STARTED_VLLM_DRAFT_SERVICES+=("$name")
    fi
    disown "$service_pid" 2>/dev/null || true
    wait_for "vLLM $name" probe_http "http://127.0.0.1:${port}/v1/models" \
        || die "vLLM $name did not become ready; see $lf"
}

start_sandbox() {
    local lf pf tmp_base redis_override ipython_override service_pid
    lf="$(log_file sandbox)"
    pf="$(pid_file sandbox)"
    tmp_base="${TMPDIR:-$RESOURCE_DIR/.tmp}/spectool_sandbox_${RUN_ID}_${SANDBOX_PORT}"
    ! port_is_listening "$SANDBOX_PORT" || die "Selected sandbox port $SANDBOX_PORT was taken before startup; retry the launcher"
    ! port_is_listening "$SANDBOX_REDIS_PORT" || die "Selected Redis port $SANDBOX_REDIS_PORT was taken before startup; retry the launcher"
    SANDBOX_TMP_BASE="$tmp_base"
    redis_override="$tmp_base/redis"
    ipython_override="$tmp_base/ipython"
    mkdir -p "$redis_override" "$ipython_override/profile_default"
    log "Starting sandbox on port $SANDBOX_PORT, redis=$SANDBOX_REDIS_PORT"
    (
        source "$CONDA_INIT"
        specagent_activate_client
        cd "$SANDBOX_ROOT"
        export SANDBOX_REDIS_PORT SANDBOX_REDIS_PASSWORD
        cp redis_client.py "$redis_override/redis_client.py"
        printf "%s\n%s\n" \
            "c.HistoryManager.enabled = False" \
            "c.HistoryManager.hist_file = ':memory:'" \
            > "$ipython_override/profile_default/ipython_kernel_config.py"
        export PYTHONPATH="$redis_override:$SANDBOX_ROOT${PYTHONPATH:+:$PYTHONPATH}"
        export IPYTHONDIR="$ipython_override"
        cd "$redis_override"
        exec setsid bash -c '
            set -Eeuo pipefail
            redis_port="$1"
            redis_password="$2"
            sandbox_port="$3"
            sandbox_workers="$4"
            redis-server \
                --port "$redis_port" \
                --bind 127.0.0.1 \
                --protected-mode yes \
                --daemonize no \
                --requirepass "$redis_password" \
                --save "" &
            redis_pid=$!
            uvicorn fast_api_server_v2:app \
                --host 127.0.0.1 \
                --port "$sandbox_port" \
                --workers "$sandbox_workers" &
            uvicorn_pid=$!
            cleanup_children() {
                trap - EXIT INT TERM
                kill -TERM "$uvicorn_pid" "$redis_pid" 2>/dev/null || true
                wait "$uvicorn_pid" "$redis_pid" 2>/dev/null || true
            }
            trap cleanup_children EXIT INT TERM
            wait -n "$redis_pid" "$uvicorn_pid"
        ' sandbox-service \
            "$SANDBOX_REDIS_PORT" \
            "$SANDBOX_REDIS_PASSWORD" \
            "$SANDBOX_PORT" \
            "$SANDBOX_WORKERS"
    ) >"$lf" 2>&1 < /dev/null &
    service_pid="$!"
    record_process_group "$pf" "$service_pid"
    STARTED_SANDBOX=true
    disown "$service_pid" 2>/dev/null || true
    wait_for "sandbox docs" probe_http "http://127.0.0.1:${SANDBOX_PORT}/docs" \
        || die "Sandbox docs did not become ready; see $lf"
    wait_for "sandbox run_jupyter" probe_sandbox "http://127.0.0.1:${SANDBOX_PORT}/run_jupyter" \
        || die "Sandbox run_jupyter did not become ready; see $lf"
}

resolve_mode() {
    [[ "$BENCH" != vdr ]] || die "VDR benchmark scripts have been removed; choose a supported --bench"
    [[ "$MAX_STEPS" =~ ^[1-9][0-9]*$ ]] \
        || die "MAX_STEPS must be a positive integer: $MAX_STEPS"
    [[ "$SAMPLE_WALL_BUDGET_SECONDS" =~ ^[0-9]+([.][0-9]+)?$ ]] \
        || die "SAMPLE_WALL_BUDGET_SECONDS must be non-negative: $SAMPLE_WALL_BUDGET_SECONDS"
    [[ "$FAST_ANSWER_HANDOFF_GRACE_SECONDS" =~ ^[0-9]+([.][0-9]+)?$ ]] \
        || die "FAST_ANSWER_HANDOFF_GRACE_SECONDS must be non-negative: $FAST_ANSWER_HANDOFF_GRACE_SECONDS"
    if [[ -z "$MAIN_MAX_MODEL_LEN" ]]; then
        case "$BENCH" in
            hrbench*) MAIN_MAX_MODEL_LEN=160000 ;;
            *) MAIN_MAX_MODEL_LEN="$MODEL_MAX_LEN" ;;
        esac
    fi
    if [[ -z "$DRAFT_MAX_MODEL_LEN" ]]; then
        DRAFT_MAX_MODEL_LEN="$MAIN_MAX_MODEL_LEN"
    fi
    if [[ "$RUN_MODE" != "baseline" && "$RUN_MODE" != "direct" ]]; then
        resolve_draft_layout
    fi
    case "$RUN_MODE" in
        direct)
            MODEL="${MODEL:-Qwen3.5-27B-vLLM}"
            [[ "$MODEL" == Qwen3.5-27B-vLLM ]] \
                || die "--direct requires MODEL=Qwen3.5-27B-vLLM"
            EVAL_MAIN_MODEL="$MAIN_MODEL_PATH"
            EVAL_MAIN_PORT="$MAIN_PORT"
            START_DRAFT=false
            ;;
        baseline)
            MODEL="${MODEL:-Agent-Qwen3.5-27B}"
            EVAL_MAIN_MODEL="$MAIN_MODEL_PATH"
            EVAL_MAIN_PORT="$MAIN_PORT"
            START_DRAFT=false
            ;;
        draft-only)
            (( DRAFT_REPLICAS == 1 )) \
                || die "--draft-only supports exactly one Draft replica"
            MODEL="${MODEL:-Agent-Qwen3.5-9B}"
            EVAL_MAIN_MODEL="$DRAFT_MODEL_PATH"
            EVAL_MAIN_PORT="$DRAFT_PORT"
            MAIN_MODEL_PATH="$DRAFT_MODEL_PATH"
            MAIN_PORT="$DRAFT_PORT"
            MAIN_GPUS="${DRAFT_GPU_GROUP_LIST[0]}"
            MAIN_TP="$DRAFT_TP"
            MAIN_GPU_MEMORY_UTILIZATION="$DRAFT_GPU_MEMORY_UTILIZATION"
            MAIN_MAX_MODEL_LEN="$DRAFT_MAX_MODEL_LEN"
            MAIN_MAX_NUM_BATCHED_TOKENS="$DRAFT_MAX_NUM_BATCHED_TOKENS"
            MAIN_EXTRA_ARGS="$DRAFT_EXTRA_ARGS"
            START_DRAFT=false
            ;;
        spectool)
            MODEL="${MODEL:-Spectool-Qwen3.5}"
            EVAL_MAIN_MODEL="$MAIN_MODEL_PATH"
            EVAL_MAIN_PORT="$MAIN_PORT"
            DRAFT_MAX_MODEL_LEN="$MAIN_MAX_MODEL_LEN"
            START_DRAFT=true
            ;;
        speceyes)
            MODEL="${MODEL:-SpecEyes-Qwen3.5}"
            EVAL_MAIN_MODEL="$MAIN_MODEL_PATH"
            EVAL_MAIN_PORT="$MAIN_PORT"
            DRAFT_MAX_MODEL_LEN="$MAIN_MAX_MODEL_LEN"
            # Match official smart_resize: factor=28, min=4*28*28 and
            # max=16384*28*28 (the already-default IMAGE_MAX_PIXELS value).
            IMAGE_RESIZE_FACTOR="$SPECEYE_IMAGE_RESIZE_FACTOR"
            IMAGE_MIN_PIXELS="$SPECEYE_IMAGE_MIN_PIXELS"
            IMAGE_RESIZE_POLICY="$SPECEYE_IMAGE_RESIZE_POLICY"
            IMAGE_FORMAT="$SPECEYE_IMAGE_FORMAT"
            # The client requests top_logprobs=64; vLLM must permit it.
            DRAFT_EXTRA_ARGS="$DRAFT_EXTRA_ARGS --max-logprobs $SPECEYE_TOP_LOGPROBS"
            log "--speceyes: batch=$SPECEYE_BATCH_SIZE, Phase-III concurrency=$API_NPROC (one batch at a time)."
            START_DRAFT=true
            ;;
        *) die "Invalid RUN_MODE: $RUN_MODE" ;;
    esac
    if [[ "$START_SERVICES" == true ]]; then
        local compact gpu seen
        local -a main_gpus=() seen_main=()
        [[ -n "$MAIN_GPUS" ]] || die "Specify --main-gpus LIST when starting Main (GPU IDs do not belong in .env)"
        [[ "$MAIN_TP" =~ ^[1-9][0-9]*$ ]] || die "MAIN_TP must be a positive integer: $MAIN_TP"
        compact="${MAIN_GPUS// /}"
        [[ "$compact" =~ ^[0-9]+(,[0-9]+)*$ ]] || die "Main GPUs must be comma-separated GPU IDs: $MAIN_GPUS"
        IFS=',' read -r -a main_gpus <<< "$compact"
        (( ${#main_gpus[@]} == MAIN_TP )) || die "Main GPU count must match --main-tp $MAIN_TP: $MAIN_GPUS"
        for gpu in "${main_gpus[@]}"; do
            [[ "$gpu" =~ ^[0-9]+$ ]] || die "Main GPU id must be numeric: $gpu"
            for seen in "${seen_main[@]:-}"; do
                [[ "$gpu" != "$seen" ]] || die "Main GPU $gpu is assigned more than once"
            done
            seen_main+=("$gpu")
        done
    fi
    RUN_DIR="${OUTPUT_DIR_CLI:-${RUN_DIR:-${WORK_DIR:-$SPECAGENT_WORK_ROOT/runs/$RUN_ID}}}"
    WORK_DIR="$RUN_DIR/predictions"
    LOG_DIR="$RUN_DIR/logs"
}

bench_token_snapshot() {
    local phase="$1" exit_code="${2:-0}" idx role method
    local -a args=("$phase" --run-dir "$RUN_DIR" --bench "$BENCH" --exit-code "$exit_code")
    method="$RUN_MODE"
    if [[ "${SPECAGENT_EVAL_DRIVER:-}" == dreamr || "${SPECAGENT_EVAL_DRIVER:-}" == specreason ]]; then
        method="$SPECAGENT_EVAL_DRIVER"
    fi
    args+=(--method "$method")
    role=main
    [[ "$RUN_MODE" != draft-only ]] || role=draft
    args+=(--service "$role=http://127.0.0.1:${MAIN_PORT}/metrics")
    if [[ "$START_DRAFT" == true ]]; then
        for idx in "${!DRAFT_PORT_LIST[@]}"; do
            args+=(--service "${DRAFT_SERVICE_NAMES[$idx]}=http://127.0.0.1:${DRAFT_PORT_LIST[$idx]}/metrics")
        done
    fi
    [[ "$START_SERVICES" == true ]] || args+=(--shared)
    [[ -z "$INDEX_FILE" ]] || args+=(--subset)
    "$CLIENT_PYTHON" "$CODE_ROOT/scripts/bench_tokens.py" "${args[@]}"
}

start_services() {
    local idx
    if [[ "$RUN_MODE" != direct ]]; then
        start_sandbox
    fi
    print_gpu_snapshot
    start_vllm main "$MAIN_GPUS" "$MAIN_MODEL_PATH" "$MAIN_PORT" \
        "$MAIN_TP" "$MAIN_GPU_MEMORY_UTILIZATION" "$MAIN_MAX_MODEL_LEN" \
        "$MAIN_MAX_NUM_BATCHED_TOKENS" "$MAIN_EXTRA_ARGS"
    if [[ "$START_DRAFT" == true ]]; then
        for idx in "${!DRAFT_PORT_LIST[@]}"; do
            start_vllm "${DRAFT_SERVICE_NAMES[$idx]}" \
                "${DRAFT_GPU_GROUP_LIST[$idx]}" "$DRAFT_MODEL_PATH" \
                "${DRAFT_PORT_LIST[$idx]}" "$DRAFT_TP" \
                "$DRAFT_GPU_MEMORY_UTILIZATION" "$DRAFT_MAX_MODEL_LEN" \
                "$DRAFT_MAX_NUM_BATCHED_TOKENS" "$DRAFT_EXTRA_ARGS"
        done
    fi
}

export_step_budget() {
    # Adapters retain their legacy names, but the launcher has one source of truth.
    export SPECTOOL_MAX_STEPS="$MAX_STEPS"
    export SPECTOOL_MATHVISION_MAX_STEPS="$MAX_STEPS"
    export SPECEYE_MAX_STEPS="$MAX_STEPS"
    export MMSEARCH_MAX_STEPS="$MAX_STEPS"
}

run_eval() {
    local lf pf eval_pid status draft_urls
    lf="$(log_file eval)"
    pf="$(pid_file eval)"
    log "Running eval driver=${SPECAGENT_EVAL_DRIVER:-vlmeval} mode=$RUN_MODE bench=$BENCH model=$MODEL"
    draft_urls="$(draft_base_urls)"
    (
        source "$CONDA_INIT"
        specagent_activate_client
        # The PyArrow shim is only for vLLM workers; EvalKit must use real PyArrow.
        unset PYTHONPATH
        cd "$SPECTOOL_ROOT"
        export LMUData="${LMUData:-$RESOURCE_DIR/LMUData}"
        export MODEL MODE API_NPROC EVAL_RETRY BENCH WORK_DIR RUN_MODE
        export SPECTOOL_RAW_TRAJ_DIR="$WORK_DIR/raw_trajectories"
        export_step_budget
        export SPECTOOL_OPENAI_MAX_RETRIES SPECTOOL_API_TIMEOUT SPECTOOL_API_RETRY
        export MAX_IMAGE_HISTORY IMAGE_MAX_SIZE TOOL_IMAGE_MAX_SIZE
        export IMAGE_MAX_PIXELS IMAGE_RESIZE_FACTOR IMAGE_MIN_PIXELS IMAGE_RESIZE_POLICY
        export IMAGE_FORMAT IMAGE_TRANSPORT LOCAL_MEDIA_ROOT
        export SPECTOOL_QJ_MAX_IMAGES="$QJ_MAX_IMAGES"
        export SPECTOOL_QJ_IMAGE_MAX_SIZE="$QJ_IMAGE_MAX_SIZE"
        export SPECTOOL_SANDBOX_SHARED_FS
        export SPECTOOL_QUALITY_JUDGE_ENABLED="$QUALITY_JUDGE_ENABLED"
        export SPECTOOL_QUALITY_JUDGE_THRESHOLD="$QUALITY_JUDGE_THRESHOLD"
        export SPECTOOL_FAST_ANSWER_ENABLED="$FAST_ANSWER_ENABLED"
        export SPECTOOL_FAST_ANSWER_HANDOFF_GRACE_SECONDS="$FAST_ANSWER_HANDOFF_GRACE_SECONDS"
        export SPECTOOL_ANSWERABILITY_THRESHOLD="$ANSWERABILITY_THRESHOLD"
        export SPECTOOL_FALLBACK_REUSE_VALIDATED_EVIDENCE="$FALLBACK_REUSE_VALIDATED_EVIDENCE"
        export SPECTOOL_SAMPLE_WALL_BUDGET_SECONDS="$SAMPLE_WALL_BUDGET_SECONDS"
        export SPECTOOL_VLLM_BASE="http://127.0.0.1:${EVAL_MAIN_PORT}/v1"
        export SPECTOOL_VLLM_MODEL="$EVAL_MAIN_MODEL"
        export SPECTOOL_VLLM_KEY="${SPECTOOL_VLLM_KEY:-EMPTY}"
        export SPECTOOL_SANDBOX_URL="http://127.0.0.1:${SANDBOX_PORT}/run_jupyter"
        if [[ -n "$INDEX_FILE" ]]; then
            export VLMEVAL_INDEX_FILE="$INDEX_FILE"
        else
            unset VLMEVAL_INDEX_FILE
        fi
        export SPECEYE_TOP_LOGPROBS SPECEYE_CONFIDENCE_THRESHOLD SPECEYE_BATCH_SIZE
        export SPECEYE_FALLBACK_NPROC="$API_NPROC"
        export SPECEYE_ROUTER_MAX_TOKENS SPECEYE_DIRECT_MAX_TOKENS
        export SPECEYE_MAIN_MAX_TOKENS
        export SPECEYE_CONFIDENCE_MODE="${SPECEYE_CONFIDENCE_MODE:-min}"
        export SPECEYE_CONFIDENCE_ALPHA="${SPECEYE_CONFIDENCE_ALPHA:-0.6}"
        if [[ "$RUN_MODE" == "spectool" || "$RUN_MODE" == "speceyes" ]]; then
            export SPECTOOL_DRAFT_BASE="$draft_urls"
            export SPECTOOL_DRAFT_MODEL="$DRAFT_MODEL_PATH"
            export SPECTOOL_DRAFT_KEY="${SPECTOOL_DRAFT_KEY:-EMPTY}"
        fi
        if [[ "${SPECAGENT_EVAL_DRIVER:-vlmeval}" == specreason || "${SPECAGENT_EVAL_DRIVER:-vlmeval}" == dreamr ]]; then
            local -a spec_args
            spec_args=(--execute --data "$BENCH" --work-dir "$WORK_DIR"
                --main-model "$MAIN_MODEL_PATH" --draft-model "$DRAFT_MODEL_PATH"
                --main-url "http://127.0.0.1:${EVAL_MAIN_PORT}/v1"
                --draft-url "http://127.0.0.1:${DRAFT_PORT}/v1"
                --api-nproc "$API_NPROC" --max-turns "$MAX_STEPS"
                --max-tool-calls "$MAX_STEPS" --max-tokens "${SPECTOOL_MAX_TOKENS:-4096}"
                --threshold "${SPECREASON_THRESHOLD:-7}")
            [[ -z "$INDEX_FILE" ]] || spec_args+=(--index-file "$INDEX_FILE")
            if [[ "$SPECAGENT_EVAL_DRIVER" == dreamr ]]; then
                spec_args+=(--method dreamr --accept-threshold "${DREAMR_ACCEPT_THRESHOLD:-0.7}"
                    --first-main-turns "${DREAMR_FIRST_MAIN_TURNS:-0}")
                exec setsid bash "$CODE_ROOT/baselines/dreamr/run_dreamr.sh" "${spec_args[@]}"
            fi
            exec setsid bash "$CODE_ROOT/baselines/specreason/run_specreason.sh" "${spec_args[@]}"
        fi
        exec setsid bash scripts/eval.sh
    ) > >(tee "$lf") 2>&1 &
    eval_pid="$!"
    record_process_group "$pf" "$eval_pid"
    if wait "$eval_pid"; then
        status=0
    else
        status=$?
    fi
    stop_pid_file "$pf"
    return "$status"
}

print_summary() {
    local draft_url sandbox_url decision_gates
    sandbox_url="http://127.0.0.1:${SANDBOX_PORT}/run_jupyter"
    decision_gates="QJ=${QUALITY_JUDGE_THRESHOLD} answerability=${ANSWERABILITY_THRESHOLD}"
    if [[ "${SPECAGENT_EVAL_DRIVER:-}" == dreamr ]]; then
        decision_gates="DREAM-R accept=${DREAMR_ACCEPT_THRESHOLD:-0.7} (independent controller)"
    elif [[ "${SPECAGENT_EVAL_DRIVER:-}" == specreason ]]; then
        decision_gates="SpecReason threshold=${SPECREASON_THRESHOLD:-7} (independent controller)"
    fi
    if [[ "$RUN_MODE" == direct ]]; then
        sandbox_url="<disabled: direct answer>"
        decision_gates="<disabled: one ordinary Main request>"
    fi
    if [[ "$START_DRAFT" == true ]]; then
        draft_url="$(draft_base_urls)"
    else
        draft_url="<disabled>"
    fi
    printf '%s\n' \
        "Run ID      : $RUN_ID" \
        "Run mode    : $RUN_MODE" \
        "Eval driver : ${SPECAGENT_EVAL_DRIVER:-vlmeval}" \
        "Run dir     : $RUN_DIR" \
        "Predictions : $WORK_DIR" \
        "Bench tokens: $RUN_DIR/token_usage.json" \
        "Main URL    : http://127.0.0.1:${MAIN_PORT}/v1" \
        "Main layout : TP=${MAIN_TP} GPUs=${MAIN_GPUS:-<resident>}" \
        "Draft URL(s): $draft_url" \
        "Draft layout: replicas=${DRAFT_REPLICAS} TP=${DRAFT_TP} GPUs=${DRAFT_GPUS}" \
        "Max steps   : $MAX_STEPS (agents only; direct always uses one request)" \
        "API limits  : main timeout=${SPECTOOL_API_TIMEOUT}s draft timeout=${SPECTOOL_DRAFT_TIMEOUT:-200}s eval attempts=$EVAL_RETRY SDK retries=$SPECTOOL_OPENAI_MAX_RETRIES" \
        "Sandbox URL : $sandbox_url" \
        "Main MM budget: $MAIN_MAX_NUM_BATCHED_TOKENS tokens" \
        "Tool image max: $TOOL_IMAGE_MAX_SIZE px" \
        "Image media : $IMAGE_TRANSPORT ${LOCAL_MEDIA_ROOT:-<embedded>}" \
        "Decision gates: $decision_gates" \
        "Fast-answer handoff grace: ${FAST_ANSWER_HANDOFF_GRACE_SECONDS}s" \
        "Fallback validated-evidence reuse: $FALLBACK_REUSE_VALIDATED_EVIDENCE" \
        "SpecEyes batch: ${SPECEYE_BATCH_SIZE} (used only in speceyes mode)" \
        "Sample wall budget: ${SAMPLE_WALL_BUDGET_SECONDS}s" \
        "Services    : $( [[ "$REUSE_SERVICES" == true ]] && echo "reused (resident)" || { [[ "$START_SERVICES" == true ]] && echo "started by this run" || echo "external (--no-services)"; } )" \
        "Index file  : ${INDEX_FILE:-<full dataset>}" \
        "Eval log    : $(log_file eval)"
}

main() {
    parse_args "$@"
    resolve_mode
    trap cleanup_services_on_exit EXIT
    trap 'exit 130' INT
    trap 'exit 143' TERM
    setup_paths
    if [[ "$REUSE_SERVICES" == true ]]; then
        check_reused_services
    fi
    if [[ "$KILL_EXISTING" == true ]]; then
        stop_recorded_services
    fi
    if [[ "$START_SERVICES" == true ]]; then
        resolve_ports
    fi
    print_summary
    if [[ "$START_SERVICES" == true ]]; then
        start_services
    fi
    if [[ "$RUN_EVAL" == true ]]; then
        TOKEN_ACCOUNTING_STARTED=true
        bench_token_snapshot before >"$RUN_DIR/logs/tokens.log" 2>&1 \
            || die "Cannot start token accounting; see $RUN_DIR/logs/tokens.log"
        run_eval
    else
        release_port_reservations
        trap - EXIT INT TERM
        log "Services started; eval skipped by --services-only"
    fi
}

if [[ "${SPECTOOL_START_TEST_SOURCE_ONLY:-0}" != "1" ]]; then
    main "$@"
fi
