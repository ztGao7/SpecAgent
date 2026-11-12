"""Exercise real occupied sockets, parallel port reservations and launcher flags."""
import importlib.util
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('service_ports', ROOT / 'scripts/service_ports.py')
ports = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ports)


def clean_env(tmp_path):
    env = {key: value for key, value in os.environ.items() if not key.startswith(
        ('MAIN_', 'DRAFT_', 'SANDBOX_', 'SPECAGENT_', 'CLIENT_', 'SERVER_', 'CONDA_', 'SPECTOOL_'))}
    env.update(SPECAGENT_ENV_FILE=str(tmp_path / 'empty.env'), CLIENT_PYTHON=sys.executable,
               SPECTOOL_START_TEST_SOURCE_ONLY='1', SPECAGENT_WORK_ROOT=str(tmp_path / 'work'))
    (tmp_path / 'empty.env').touch()
    return env


def free_base():
    with socket.socket() as listener:
        listener.bind(('127.0.0.1', 0))
        port = listener.getsockname()[1]
    # Leave enough headroom for fallback attempts on macOS ephemeral ports.
    return min(port, 50000)


def close_locks(locks):
    for fd in locks:
        os.close(fd)


@pytest.mark.parametrize('listening', [False, True])
def test_skip_bound_and_listening_ports(listening):
    with socket.socket() as listener:
        listener.bind(('127.0.0.1', 0))
        if listening:
            listener.listen()
        start = listener.getsockname()[1]
        selected, locks = ports.reserve([('main', start)], 1, 20)
        try:
            assert selected['main'] != start
            assert ports.port_available(selected['main'])
        finally:
            close_locks(locks)


def test_same_port_requested_for_all_services_and_another_experiment():
    start = free_base()
    request = [(name, start) for name in ['main', 'draft', 'sandbox', 'redis']]
    first, locks = ports.reserve(request, 1, 50)
    second_locks = []
    try:
        second, second_locks = ports.reserve(request, 1, 50)
        assert len(set(first.values()) | set(second.values())) == 8
    finally:
        close_locks(second_locks)
        close_locks(locks)
    again, locks = ports.reserve(request, 1, 50)
    try:
        assert again == first
    finally:
        close_locks(locks)


def test_exhaustion_releases_partially_reserved_ports():
    start = free_base()
    with pytest.raises(ValueError, match='No free draft port'):
        ports.reserve([('main', start), ('draft', start)], 1, 1)
    selected, locks = ports.reserve([('retry', start)], 1, 1)
    close_locks(locks)
    assert selected['retry'] == start


@pytest.mark.parametrize('port', [0, -1, 65536])
def test_port_range_is_validated(port):
    with pytest.raises(ValueError, match='out of range'):
        ports.reserve([('main', port)], 100, 50)


def test_ipv6_listener_is_busy():
    try:
        listener = socket.socket(socket.AF_INET6)
        listener.setsockopt(socket.IPPROTO_IPV6, socket.IPV6_V6ONLY, 1)
        listener.bind(('::1', 0))
        listener.listen()
    except OSError:
        pytest.skip('IPv6 unavailable on this host')
    try:
        assert not ports.port_available(listener.getsockname()[1])
    finally:
        listener.close()


def test_root_launcher_skips_an_existing_model_api_and_propagates_ports(tmp_path):
    # A real /v1/models HTTP endpoint must be skipped, not silently reused.
    from http.server import BaseHTTPRequestHandler, HTTPServer
    import threading

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b'{"data":[{"id":"other-experiment"}]}')

        def log_message(self, *_):
            pass

    existing = HTTPServer(('127.0.0.1', 0), Handler)
    thread = threading.Thread(target=existing.serve_forever, daemon=True)
    thread.start()
    start = existing.server_address[1]
    run = tmp_path / 'run'
    env = clean_env(tmp_path)
    # Use real main/argument/port/summary/cleanup code; only replace heavyweight
    # model and sandbox boundaries. No GPU or Conda environment is required.
    code = r'''
source "$1/start_test.sh"
setup_paths() { mkdir -p "$RUN_DIR/logs" "$RUN_DIR/pids"; }
start_services() { :; }
bench_token_snapshot() { :; }
run_eval() {
    printf 'resolved=%s,%s,%s,%s\n' "$MAIN_PORT" "$DRAFT_PORT" "$SANDBOX_PORT" "$SANDBOX_REDIS_PORT"
    printf 'broker=%s\n' "$PORT_RESERVATION_PID"
    printf 'draft_url=%s\n' "$(draft_base_urls)"
    printf 'eval_port=%s\n' "$EVAL_MAIN_PORT"
}
main --spectool --main-gpus 4 --draft-gpus 5 --main-port "$2" --draft-port "$2" --sandbox-port "$2" --redis-port "$2" --port-offset 1
'''
    try:
        result = subprocess.run(['bash', '-c', code, '_', str(ROOT), str(start)],
                                env=dict(env, RUN_DIR=str(run)), capture_output=True, text=True, timeout=20)
        assert result.returncode == 0, result.stderr
        fields = dict(line.split('=', 1) for line in result.stdout.splitlines()
                      if line.startswith(('resolved=', 'broker=', 'draft_url=', 'eval_port=')))
        selected = list(map(int, fields['resolved'].split(',')))
        assert len(set(selected)) == 4
        assert start not in selected
        assert fields['draft_url'] == f'http://127.0.0.1:{selected[1]}/v1'
        assert int(fields['eval_port']) == selected[0]
        with pytest.raises(ProcessLookupError):
            os.kill(int(fields['broker']), 0)
        assert existing.fileno() >= 0
    finally:
        existing.shutdown()
        existing.server_close()
        thread.join()


def test_parallel_launchers_hold_disjoint_ports_until_exit(tmp_path):
    start = free_base()
    env = clean_env(tmp_path)
    code = r'''
source "$1/start_test.sh"
setup_paths() { mkdir -p "$RUN_DIR/logs" "$RUN_DIR/pids"; }
start_services() { :; }
bench_token_snapshot() { :; }
run_eval() {
    printf '%s %s %s %s\n' "$MAIN_PORT" "$DRAFT_PORT" "$SANDBOX_PORT" "$SANDBOX_REDIS_PORT" > "$RUN_DIR/selected"
    for ((attempt=0; attempt<100; attempt++)); do
        [[ ! -f "$RUN_DIR/release" ]] || return 0
        sleep 0.1
    done
    return 1
}
main --spectool --main-gpus 4 --draft-gpus 5 --main-port "$2" --draft-port "$2" --sandbox-port "$2" --redis-port "$2" --port-offset 1
'''
    runs = [tmp_path / f'run-{index}' for index in range(2)]
    processes = [subprocess.Popen(['bash', '-c', code, '_', str(ROOT), str(start)],
                                  env=dict(env, RUN_DIR=str(run)), stdout=subprocess.PIPE,
                                  stderr=subprocess.PIPE, text=True) for run in runs]
    try:
        deadline = time.monotonic() + 8
        while not all((run / 'selected').exists() for run in runs) and time.monotonic() < deadline:
            time.sleep(0.05)
        assert all((run / 'selected').exists() for run in runs)
        selections = [set((run / 'selected').read_text().split()) for run in runs]
        assert len(selections[0] | selections[1]) == 8
        for run in runs:
            (run / 'release').touch()
        for process in processes:
            stdout, stderr = process.communicate(timeout=10)
            assert process.returncode == 0, stdout + stderr
    finally:
        for process in processes:
            if process.poll() is None:
                process.terminate()
                process.communicate(timeout=10)


@pytest.mark.parametrize('method,flags', [
    ('spectool', ['--main-gpus', '4', '--draft-gpus', '5']),
    ('baseline', ['--main-gpus', '4']),
    ('direct', ['--main-gpus', '4']),
    ('draft-only', ['--draft-gpus', '5']),
])
def test_gpu_cli_is_forwarded_to_actual_launcher(method, flags, tmp_path):
    result = subprocess.run(['bash', str(ROOT / 'scripts/run_test.sh'), '--method', method,
                             '--dry-run', *flags, '--main-port', '40001'],
                            env=clean_env(tmp_path), capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    command = result.stdout.splitlines()[-1]
    assert '--main-port 40001' in command
    for index in range(0, len(flags), 2):
        assert f'{flags[index]} {flags[index + 1]}' in command


def test_real_run_requires_gpu_flags_before_loading_dependencies(tmp_path):
    result = subprocess.run(['bash', str(ROOT / 'scripts/run_test.sh'), '--method', 'spectool'],
                            env=clean_env(tmp_path), capture_output=True, text=True)
    assert result.returncode != 0
    assert 'Specify --main-gpus' in result.stderr
    assert not (tmp_path / 'work').exists()


def test_resident_services_need_no_gpu_allocation(tmp_path):
    code = r'''
source "$1/start_test.sh"
parse_args --reuse-services --spectool
resolve_mode
printf 'start=%s main=%s draft=%s\n' "$START_SERVICES" "$MAIN_GPUS" "$DRAFT_GPUS"
'''
    result = subprocess.run(['bash', '-c', code, '_', str(ROOT)],
                            env=clean_env(tmp_path), capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert 'start=false main= draft=' in result.stdout


@pytest.mark.parametrize('flags,message', [
    (['--baseline', '--main-gpus', '0,1'], 'Main GPU count must match'),
    (['--baseline', '--main-gpus', '0,0', '--main-tp', '2'], 'assigned more than once'),
    (['--speceyes', '--main-gpus', '0', '--draft-gpus', '0'], 'assigned to both Main and Draft'),
])
def test_gpu_allocation_errors_fail_before_deployment(flags, message, tmp_path):
    code = 'source "$1/start_test.sh"; shift; parse_args "$@"; resolve_mode'
    result = subprocess.run(['bash', '-c', code, '_', str(ROOT), *flags],
                            env=clean_env(tmp_path), capture_output=True, text=True)
    assert result.returncode != 0
    assert message in result.stderr


def test_templates_contain_no_gpu_or_port_assignments():
    for template in [ROOT / '.env.example', ROOT / 'configs/env.example']:
        keys = {line.split('=', 1)[0] for line in template.read_text().splitlines()
                if line and not line.startswith('#')}
        assert not keys.intersection({'MAIN_GPUS', 'DRAFT_GPUS', 'MAIN_TP', 'DRAFT_TP',
                                      'MAIN_PORT', 'DRAFT_PORT', 'SANDBOX_PORT', 'SANDBOX_REDIS_PORT'})
