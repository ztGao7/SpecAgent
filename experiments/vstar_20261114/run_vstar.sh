#!/usr/bin/env bash
# Full VStarBench (191) experiments of 2026-11-14. Wraps the repository's start_test.sh; reads the root .env.
#
#   bash experiments/vstar_20261114/run_vstar.sh services            # start 27B + 9B + sandbox once and keep them up
#   bash experiments/vstar_20261114/run_vstar.sh <variant> [...]     # run variants one after another on those services
#
# Variants
#   baseline        Agent-27B tool baseline
#   async_final     asynchronous Spectool, final config (mid fast prompt, lenient verifier, GOOD/BAD step judge)
#   sync_fast_on    step-synchronous protocol, fast answer on
#   sync_fast_off   step-synchronous protocol, fast answer off
#   no_fast | no_qj | no_fast_no_qj   2x2 ablations of async_final (no_qj = no step judge and no answer verifier)
# Knobs (environment, optional)
#   QBAD=1     consecutive BAD step judgements before Main takes over (default 2; 1 = first BAD -> fallback)
#   GRACE=1    seconds the async loop waits for an in-flight fast answer when Draft's first tool step arrives (0-10)
#   MAIN_GPUS=0,1 DRAFT_GPUS=2,3 MAIN_TP=2 DRAFT_TP=2 MAIN_PORT DRAFT_PORT SANDBOX_PORT NPROC=4 OUT_ROOT MEDIA_PARENT
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd); REPO=$(cd "$HERE/../.." && pwd)
source "$REPO/scripts/common.sh"
P=$HERE/prompts
MAIN_GPUS=${MAIN_GPUS:-0,1}; DRAFT_GPUS=${DRAFT_GPUS:-2,3}; MAIN_TP=${MAIN_TP:-2}; DRAFT_TP=${DRAFT_TP:-2}
MAIN_PORT=${MAIN_PORT:-55980}; DRAFT_PORT=${DRAFT_PORT:-55981}; SANDBOX_PORT=${SANDBOX_PORT:-55982}
NPROC=${NPROC:-4}; QBAD=${QBAD:-2}; GRACE=${GRACE:-1}
OUT_ROOT=${OUT_ROOT:-$SPECAGENT_RESULTS_ROOT/vstar_20261114}
MEDIA_PARENT=${MEDIA_PARENT:-$SPECAGENT_WORK_ROOT/media/vstar_20261114}
# Service settings of the reference runs: 0.9 / 0.8 GPU memory, 8 in-flight sequences, prefix caching on both models.
export MAIN_GPU_MEMORY_UTILIZATION=${MAIN_GPU_MEMORY_UTILIZATION:-0.9} DRAFT_GPU_MEMORY_UTILIZATION=${DRAFT_GPU_MEMORY_UTILIZATION:-0.8}
mkdir -p "$MEDIA_PARENT" "$OUT_ROOT"
SERVICE_ARGS=(--main-port $MAIN_PORT --draft-port $DRAFT_PORT --sandbox-port $SANDBOX_PORT --main-gpus $MAIN_GPUS --draft-gpus $DRAFT_GPUS
  --main-tp $MAIN_TP --draft-tp $DRAFT_TP --draft-replicas 1 --image-transport file --media-parent "$MEDIA_PARENT")
COMMON=(--reuse-services --bench vstar --mode infer --api-nproc $NPROC "${SERVICE_ARGS[@]}" --max-steps 8 --max-image-history 5
  --image-max-size 0 --image-max-pixels 12845056 --image-resize-factor 32 --tool-image-max-size 1536)
SPEC=(--spectool --qj-image-max-size 0 --quality-judge --quality-threshold 0.7 --answerability-threshold 0.9
  --fast-answer-handoff-grace $GRACE --no-fallback-reuse-validated-evidence)
FINAL_ENV=(SPECTOOL_QJ_ONLY=1 SPECTOOL_IMAGE_FIRST=assess SPECTOOL_JUDGE_ANCHOR=fast SPECTOOL_QJ_PROMPT_FILE=$P/qj_prompt_target.txt
  SPECTOOL_FAST_HANDOFF_DETACH=1 SPECTOOL_FAST_PROMPT_FILE=$P/fast_system_mid.txt SPECTOOL_FAST_INSTR=lenient
  SPECTOOL_REVIEW_SYSTEM_FILE=$P/review_system.txt SPECTOOL_REVIEW_INSTRUCTION_FILE=$P/review_instruction.txt
  SPECTOOL_MAX_CONSECUTIVE_QUALITY_BAD=$QBAD)
UNSET=(-u SPECTOOL_SYNC -u SPECTOOL_ABL_NO_STEP_JUDGE -u SPECTOOL_ABL_NO_VERIFIER -u SPECTOOL_FAST_ACCEPT_AT_HANDOFF)
run() {  # run <name> <extra env...> -- <start_test.sh args...>
  local name=$1; shift; local envs=(); while [[ $1 != -- ]]; do envs+=("$1"); shift; done; shift
  local tag=${name}$([[ $QBAD != 2 ]] && echo _qbad$QBAD)$([[ $GRACE != 1 ]] && echo _grace$GRACE)_$(date +%m%d_%H%M)
  local rc=0
  echo "[$(date '+%F %T')] start $tag -> $OUT_ROOT/$tag"
  ( cd "$REPO" && env "${UNSET[@]}" "${envs[@]}" bash start_test.sh "$@" --run-id "$tag" --work-dir "$OUT_ROOT/$tag" ) > "$OUT_ROOT/$tag.out" 2>&1 || rc=$?
  echo "[$(date '+%F %T')] done $tag exit=$rc"
}
[[ $# -ge 1 ]] || { sed -n 2,19p "$0"; exit 1; }
for v in "$@"; do case $v in
  services) ( cd "$REPO" && bash start_test.sh --spectool --services-only "${SERVICE_ARGS[@]}" ) ;;
  baseline) run baseline -- --baseline "${COMMON[@]}" ;;
  async_final) run async_final "${FINAL_ENV[@]}" -- "${SPEC[@]}" "${COMMON[@]}" --fast-answer ;;
  sync_fast_on) run sync_fast_on "${FINAL_ENV[@]}" SPECTOOL_SYNC=1 -- "${SPEC[@]}" "${COMMON[@]}" --fast-answer ;;
  sync_fast_off) run sync_fast_off "${FINAL_ENV[@]}" SPECTOOL_SYNC=1 -- "${SPEC[@]}" "${COMMON[@]}" --no-fast-answer ;;
  no_fast) run no_fast "${FINAL_ENV[@]}" -- "${SPEC[@]}" "${COMMON[@]}" --no-fast-answer ;;
  no_qj) run no_qj "${FINAL_ENV[@]}" SPECTOOL_ABL_NO_STEP_JUDGE=1 SPECTOOL_ABL_NO_VERIFIER=1 -- "${SPEC[@]}" "${COMMON[@]}" --fast-answer ;;
  no_fast_no_qj) run no_fast_no_qj "${FINAL_ENV[@]}" SPECTOOL_ABL_NO_STEP_JUDGE=1 SPECTOOL_ABL_NO_VERIFIER=1 -- "${SPEC[@]}" "${COMMON[@]}" --no-fast-answer ;;
  *) echo "unknown variant $v"; exit 1 ;;
esac; done
