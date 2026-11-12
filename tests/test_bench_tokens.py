"""Whole-bench accounting and a real HTTP metrics / launcher integration."""
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('bench_tokens', ROOT / 'scripts/bench_tokens.py')
tokens = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tokens)


def metrics(prompt=0, generation=0):
    return f'''# TYPE vllm:prompt_tokens_total counter
vllm:prompt_tokens_total{{model_name="model",engine="0"}} {prompt}
vllm:generation_tokens_total{{model_name="model",engine="0"}} {generation}
vllm:prompt_tokens_created{{model_name="model",engine="0"}} 100
vllm:num_requests_running{{model_name="model",engine="0"}} 0
vllm:num_requests_waiting{{model_name="model",engine="0"}} 0
'''


def test_counts_all_engine_labels_and_main_draft_services():
    before = {name: tokens.parse_metrics(metrics()) for name in ['main', 'draft_1', 'draft_2']}
    after = {name: tokens.parse_metrics(metrics(100, 20)) for name in before}
    summary = tokens.summarize(before, after)
    assert summary['input_tokens'] == 300
    assert summary['output_tokens'] == 60
    assert summary['bench_attribution_complete']
    text = metrics(10, 2) + 'vllm:prompt_tokens_total{model_name="model",engine="1"} 15\n'
    parsed = tokens.parse_metrics(text)
    assert sum(parsed['counters']['vllm:prompt_tokens_total'].values()) == 25


@pytest.mark.parametrize('failure', ['missing', 'reset', 'restart', 'busy', 'labels'])
def test_incomplete_measurements_are_null_not_zero(failure):
    before = {'main': tokens.parse_metrics(metrics(10, 3))}
    after = {'main': tokens.parse_metrics(metrics(20, 5))}
    if failure == 'missing':
        after['main'] = {'error': 'unavailable'}
    elif failure == 'reset':
        after['main']['counters']['vllm:prompt_tokens_total'] = dict.fromkeys(
            after['main']['counters']['vllm:prompt_tokens_total'], 1)
    elif failure == 'restart':
        after['main']['created'] = {'restart': 200}
    elif failure == 'busy':
        after['main']['idle'] = False
    else:
        after['main']['counters']['vllm:prompt_tokens_total']['new_engine'] = 10
    summary = tokens.summarize(before, after)
    assert not summary['counts_available']
    assert summary['input_tokens'] is None and summary['output_tokens'] is None
    assert summary['errors']


def test_shared_server_totals_are_not_claimed_as_exclusive_bench_usage():
    summary = tokens.summarize({'main': tokens.parse_metrics(metrics())},
                               {'main': tokens.parse_metrics(metrics(50, 5))}, shared=True)
    assert summary['input_tokens'] == 50
    assert summary['counts_available']
    assert not summary['bench_attribution_complete']


@pytest.mark.parametrize('text', ['# no counters', metrics().replace(' 0\n', ' NaN\n'), metrics(1.5, 3)])
def test_missing_or_invalid_metrics_are_rejected(text):
    with pytest.raises(ValueError):
        tokens.parse_metrics(text)


def isolated_env(tmp_path):
    env = {key: value for key, value in os.environ.items() if not key.startswith(
        ('SPECAGENT_', 'SPECTOOL_', 'MAIN_', 'DRAFT_', 'CLIENT_', 'SERVER_', 'CONDA_', 'SANDBOX_'))
        and key not in {'WORK_DIR', 'RUN_DIR', 'RUN_ID', 'CODE_ROOT', 'CONDA_INIT', 'MODEL', 'INDEX_FILE'}}
    config = tmp_path / 'empty.env'
    config.touch()
    env.update(SPECAGENT_ENV_FILE=str(config), CLIENT_PYTHON=sys.executable,
               SPECAGENT_WORK_ROOT=str(tmp_path / 'data'), SPECTOOL_START_TEST_SOURCE_ONLY='1')
    return env


@pytest.mark.parametrize('exit_code', [0, 7])
def test_launcher_keeps_one_directory_and_token_totals_even_on_failure(tmp_path, exit_code):
    state = {'prompt': 100, 'output': 10}

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path == '/evaluate':
                state.update(prompt=850, output=65)
                body = b'ok'
            else:
                body = metrics(state['prompt'], state['output']).encode()
            self.send_response(200)
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *_):
            pass

    server = HTTPServer(('127.0.0.1', 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    run = tmp_path / 'one round with spaces'
    model = tmp_path / 'model'
    model.mkdir()
    code = r'''
source "$1/start_test.sh"
run_eval() {
    curl --noproxy '*' -fsS "http://127.0.0.1:$MAIN_PORT/evaluate" > "$RUN_DIR/logs/eval.log"
    mkdir -p "$WORK_DIR"
    printf 'sample prediction' > "$WORK_DIR/result.txt"
    return "$TEST_EXIT_CODE"
}
main --direct --no-services --main-port "$2" --main-model "$3" --output-dir "$4" --run-id named_round
'''
    try:
        result = subprocess.run(['bash', '-c', code, '_', str(ROOT), str(server.server_port), str(model), str(run)],
                                env=dict(isolated_env(tmp_path), TEST_EXIT_CODE=str(exit_code)),
                                text=True, capture_output=True, timeout=15)
        assert result.returncode == exit_code, result.stdout + result.stderr
        summary = json.loads((run / 'token_usage.json').read_text())
        assert summary['input_tokens'] == 750 and summary['output_tokens'] == 55
        assert summary['benchmark_completed'] == (exit_code == 0)
        assert summary['counts_available']
        assert (run / 'logs/eval.log').read_text() == 'ok'
        assert (run / 'predictions/result.txt').exists()
        assert (run / 'metrics/before/main.prom').exists()
        assert (run / 'metrics/after/main.prom').exists()
        info = json.loads((run / 'run_info.json').read_text())
        assert info['status'] == ('completed' if exit_code == 0 else 'failed')
        assert not (tmp_path / 'data/runs/named_round').exists()
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def test_default_and_explicit_paths_are_same_for_all_run_files(tmp_path):
    code = r'''
source "$1/start_test.sh"
parse_args --direct --main-gpus 4 --run-id example
resolve_mode
printf '%s\n' "$RUN_DIR" "$WORK_DIR" "$LOG_DIR"
'''
    for custom_results in [None, tmp_path / 'configured results']:
        env = isolated_env(tmp_path)
        if custom_results is not None:
            env['SPECAGENT_RESULTS_ROOT'] = str(custom_results)
        result = subprocess.run(['bash', '-c', code, '_', str(ROOT)], env=env,
                                capture_output=True, text=True)
        assert result.returncode == 0, result.stderr
        run = (custom_results or tmp_path / 'data/runs') / 'example'
        assert result.stdout.splitlines() == [str(run), str(run / 'predictions'), str(run / 'logs')]
        assert not run.exists()


def test_quick_output_dir_plan_has_no_side_effects(tmp_path):
    run = tmp_path / 'chosen output'
    result = subprocess.run(['bash', str(ROOT / 'scripts/run_test.sh'), '--method', 'spectool', '--full',
                             '--main-gpus', '4', '--draft-gpus', '5', '--output-dir', str(run), '--dry-run'],
                            env=isolated_env(tmp_path), capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert str(run) in result.stdout
    assert '--output-dir' in result.stdout.splitlines()[-1]
    assert '--index-file' not in result.stdout
    assert not run.exists()


def test_previous_run_is_not_overwritten_or_stopped(tmp_path):
    run, model = tmp_path / 'previous', tmp_path / 'model'
    run.mkdir()
    model.mkdir()
    (run / 'run_info.json').write_text('{"status":"running"}')
    code = r'''
source "$1/start_test.sh"
stop_pid_file() { printf 'UNEXPECTED_STOP\n'; }
cleanup_vllm_ipc_dirs() { printf 'UNEXPECTED_IPC_CLEANUP\n'; }
main --direct --main-gpus 4 --main-model "$2" --output-dir "$3"
'''
    result = subprocess.run(['bash', '-c', code, '_', str(ROOT), str(model), str(run)],
                            env=isolated_env(tmp_path), capture_output=True, text=True)
    assert result.returncode != 0
    assert 'already contains a previous run' in result.stderr
    assert 'UNEXPECTED_STOP' not in result.stdout
    assert 'UNEXPECTED_IPC_CLEANUP' not in result.stdout
    assert (run / 'run_info.json').read_text() == '{"status":"running"}'


def test_full_bench_clears_inherited_subset_before_starting_launcher(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / 'scripts'))
    spec = importlib.util.spec_from_file_location('quick_run_test', ROOT / 'scripts/run_test.py')
    quick = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(quick)
    model = tmp_path / 'model'
    model.mkdir()
    (model / 'config.json').write_text('{}')
    data = tmp_path / 'data'
    data.mkdir()
    (data / 'VStarBench.tsv').write_text('index\tquestion\n1\tQ\n')
    for key, value in {'CLIENT_PYTHON': sys.executable, 'SERVER_PYTHON': sys.executable,
                       'LMUData': str(data), 'MAIN_MODEL_PATH': str(model),
                       'INDEX_FILE': str(tmp_path / 'stale-indices.json')}.items():
        monkeypatch.setenv(key, value)
    calls = []
    monkeypatch.setattr(quick.subprocess, 'run', lambda command, **kwargs: calls.append((command, kwargs)))
    monkeypatch.setattr(sys, 'argv', ['run_test.py', '--method', 'direct', '--full',
                                    '--main-gpus', '4', '--output-dir', str(tmp_path / 'run')])
    quick.main()
    command, options = calls[-1]
    assert command[1] == str(ROOT / 'start_test.sh')
    assert '--index-file' not in command
    assert options['env']['INDEX_FILE'] == ''
