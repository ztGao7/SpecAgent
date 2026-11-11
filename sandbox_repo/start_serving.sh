#!/usr/bin/env bash
set -Eeuo pipefail

SANDBOX_PORT="${SANDBOX_PORT:-18901}"
SANDBOX_REDIS_PORT="${SANDBOX_REDIS_PORT:-26379}"
SANDBOX_WORKERS="${SANDBOX_WORKERS:-8}"

mkdir -p uvilog

# The bundled client has a fixed default port. The SIF application directory is
# read-only, so write an override module to tmpfs and import it before the copy
# bundled in the image.
sandbox_app_dir="$(pwd)"
redis_override_dir="${TMPDIR:-/tmp}/jupyter_sandbox_${SANDBOX_PORT}"
mkdir -p "$redis_override_dir"
export SANDBOX_REDIS_PORT
cp redis_client.py "$redis_override_dir/redis_client.py"
export PYTHONPATH="$redis_override_dir:$sandbox_app_dir${PYTHONPATH:+:$PYTHONPATH}"

# Disable IPython's on-disk history sqlite db. The history-db code path has
# crashed kernels under disk I/O errors (sandbox_18902.log, 2026-06-21): the
# corrupt-db recovery handler in this IPython version itself raises
# AttributeError instead of recovering, so kernels die before kernel_info
# replies. Disabling history removes the dependency on that disk entirely.
ipython_override_dir="${TMPDIR:-/tmp}/jupyter_sandbox_ipython_${SANDBOX_PORT}"
mkdir -p "$ipython_override_dir/profile_default"
cat > "$ipython_override_dir/profile_default/ipython_kernel_config.py" <<'EOF'
c.HistoryManager.enabled = False
c.HistoryManager.hist_file = ':memory:'
EOF
export IPYTHONDIR="$ipython_override_dir"

redis-server ./redis.conf \
    --port "$SANDBOX_REDIS_PORT" \
    --requirepass "${SANDBOX_REDIS_PASSWORD:-}" \
    --pidfile "$(pwd)/uvilog/redis_${SANDBOX_REDIS_PORT}.pid" \
    --dir "$(pwd)/uvilog" \
    --dbfilename "sandbox_${SANDBOX_REDIS_PORT}.rdb" \
    --save "" &
redis_pid=$!

cd "$redis_override_dir"
uvicorn fast_api_server_v2:app \
    --host 127.0.0.1 \
    --port "$SANDBOX_PORT" \
    --workers "$SANDBOX_WORKERS" &
uvicorn_pid=$!

cleanup() {
    kill "$uvicorn_pid" "$redis_pid" 2>/dev/null || true
    wait "$uvicorn_pid" "$redis_pid" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

wait -n "$redis_pid" "$uvicorn_pid"
