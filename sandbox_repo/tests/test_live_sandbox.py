"""Real Redis/HTTP/Jupyter tests on private ports, with no model or GPU calls."""
import concurrent.futures
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time
import uuid

import psutil
import pytest
import redis
import requests
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parent / 'spectool'))
from spectool_agent import sandbox
from spectool_agent.tools import ToolRegistry, register_default_tools
from spectool_agent.types import ToolCall


def free_port():
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', 0))
        return sock.getsockname()[1]


@pytest.fixture(scope='module')
def server(tmp_path_factory):
    root = tmp_path_factory.mktemp('sandbox_runtime_live')
    redis_port, api_port = free_port(), free_port()
    while api_port == redis_port:
        api_port = free_port()
    logs, processes = [], []
    try:
        redis_log = (root / 'redis.log').open('w')
        logs.append(redis_log)
        processes.append(subprocess.Popen([
            'redis-server', '--bind', '127.0.0.1', '--port', str(redis_port),
            '--save', '', '--appendonly', 'no', '--dir', str(root)],
            stdout=redis_log, stderr=subprocess.STDOUT, start_new_session=True))
        conn = redis.Redis(host='127.0.0.1', port=redis_port, socket_timeout=1, protocol=2)
        for _ in range(100):
            try:
                if conn.ping():
                    break
            except redis.exceptions.ConnectionError:
                time.sleep(.05)
        else:
            pytest.fail('private Redis failed to start')
        api_log = (root / 'server.log').open('w')
        logs.append(api_log)
        env = dict(os.environ, SANDBOX_SMOKE_REDIS_PORT=str(redis_port),
                   TMPDIR=str(root), PYTHONPATH=str(ROOT))
        processes.append(subprocess.Popen([
            sys.executable, '-m', 'uvicorn', 'smoke_app:app', '--host', '127.0.0.1',
            '--port', str(api_port), '--workers', '2'], cwd=str(ROOT / 'tests'),
            env=env, stdout=api_log, stderr=subprocess.STDOUT, start_new_session=True))
        base = f'http://127.0.0.1:{api_port}'
        for _ in range(200):
            try:
                if requests.get(base + '/docs', timeout=.5).ok:
                    break
            except requests.RequestException:
                pass
            time.sleep(.05)
        else:
            pytest.fail('private Sandbox failed to start')
        print(f'LIVE_LOG_DIR={root}', flush=True)
        yield base, conn, root
    finally:
        # These PIDs and their descendants were created only by this fixture.
        for process in reversed(processes):
            descendants = []
            if process.poll() is None:
                descendants = psutil.Process(process.pid).children(recursive=True)
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait(timeout=5)
            for child in descendants:
                if child.is_running():
                    try:
                        child.kill()
                    except psutil.NoSuchProcess:
                        pass
        for log in logs:
            log.close()


def send(server, session, code, timeout=20):
    response = requests.post(server[0] + '/run_jupyter', json={
        'session_id': session, 'request_id': uuid.uuid4().hex,
        'code': code, 'timeout': timeout}, timeout=timeout+15)
    response.raise_for_status()
    return response.json()


def success(result):
    assert result['status'] == 'success', result
    assert result['state_committed'], result
    return result['output']['stdout'].strip()


def test_real_state_persistence_and_cross_worker_serialization(server):
    session = 'serial-' + uuid.uuid4().hex
    success(send(server, session, 'counter = 0'))
    code = 'import time\nold = counter\ntime.sleep(.15)\ncounter = old + 1\nprint(counter)'
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(lambda _: send(server, session, code), range(4)))
    assert sorted(int(success(r)) for r in results) == [1, 2, 3, 4]
    assert int(success(send(server, session, 'print(counter)'))) == 4
    intervals = sorted((r['timing']['execute_start_unix'],
                        r['timing']['execute_start_unix'] + r['timing']['execute_s']) for r in results)
    assert all(left[1] <= right[0] for left, right in zip(intervals, intervals[1:]))
    print('SERIAL_WORKER_PIDS=' + str(sorted({r['timing']['worker_pid'] for r in results})))


def test_real_distinct_sessions_have_overlapping_execution(server):
    def request(_):
        return send(server, 'parallel-' + uuid.uuid4().hex,
                    'import time\ntime.sleep(.8)\nprint("ok")')
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(request, range(4)))
    assert all(success(r) == 'ok' for r in results)
    events = []
    for result in results:
        t = result['timing']
        events.extend([(t['execute_start_unix'], 1),
                       (t['execute_start_unix'] + t['execute_s'], -1)])
    active = peak = 0
    for _, delta in sorted(events):
        active += delta
        peak = max(peak, active)
    assert peak >= 2
    print(f'LIVE_EXECUTION_PEAK={peak}')


@pytest.mark.parametrize('continuous', [False, True])
def test_real_deadline_kills_kernel_and_preserves_last_committed_state(server, continuous):
    session = 'timeout-' + uuid.uuid4().hex
    success(send(server, session, 'value = 7'))
    marker = server[2] / (session + '.pid')
    code = f'import os, time\nopen({str(marker)!r}, "w").write(str(os.getpid()))\nvalue = 99\n'
    code += ('while True:\n print("progress", flush=True)\n time.sleep(.02)' if continuous
             else 'while True:\n time.sleep(1)')
    started = time.monotonic()
    result = send(server, session, code, timeout=4)
    elapsed = time.monotonic() - started
    assert result['status'] == 'error' and 'Deadline' in result['error'], result
    assert not result['state_committed'] and result['session_usable']
    assert elapsed < 10 and marker.exists()
    pid = int(marker.read_text())
    assert not psutil.pid_exists(pid), f'timed-out kernel {pid} survived'
    assert success(send(server, session, 'print(value)')) == '7'
    print(f'LIVE_DEADLINE continuous={continuous} elapsed={elapsed:.3f} kernel_gone=True')


def test_real_snapshot_exception_cleans_kernel_without_replay(server):
    session = 'snapshot-' + uuid.uuid4().hex
    success(send(server, session, 'value = 7'))
    marker = server[2] / (session + '.pid')
    countfile = server[2] / (session + '.count')
    code = ('# inject_snapshot_failure\nimport os\n'
            f'open({str(marker)!r}, "w").write(str(os.getpid()))\n'
            f'open({str(countfile)!r}, "a").write("one\n")\nvalue = 99')
    # Escape the newline inside the generated test string literal.
    code = code.replace('write("one\n")', 'write("one\\n")')
    result = send(server, session, code)
    assert result['status'] == 'error' and 'snapshot failure' in result['error']
    assert not psutil.pid_exists(int(marker.read_text()))
    assert countfile.read_text() == 'one\n'
    assert success(send(server, session, 'print(value)')) == '7'


@pytest.mark.parametrize('shared', ['0', '1'])
def test_real_client_originals_crops_and_corrective_python_error(server, monkeypatch, shared):
    monkeypatch.setenv('SPECTOOL_SANDBOX_SHARED_FS', shared)
    monkeypatch.setattr(sandbox, '_sandbox_urls', [server[0] + '/run_jupyter'])
    originals = [Image.new('RGB', (12, 9), 'red'), Image.new('RGB', (8, 8), 'blue')]
    registry = ToolRegistry()
    register_default_tools(registry)
    registry.set_input_images(originals)
    try:
        observation, images = registry.dispatch(ToolCall('sandbox', {'code':
            'print(image_1.size, image_2.size)\nvalue = 42\n'
            'import matplotlib.pyplot as plt\nplt.imshow(image_1.crop((0,0,4,4)))\nplt.show()'}), originals[0])
        assert '(12, 9)' in observation and '(8, 8)' in observation
        assert images and '沙箱错误' not in observation, observation
        session_id = registry._previous_context['session_id']
        observation, _ = registry.dispatch(ToolCall('sandbox', {'code':
            'print(value)\nraise ValueError("correctable")'}), originals[0])
        assert '42' in observation and 'correctable' in observation
        observation, _ = registry.dispatch(ToolCall('sandbox', {'code':
            'print(value + 1, image_1.getpixel((0,0)))'}), originals[0])
        assert '43' in observation and '(255, 0, 0)' in observation, observation
        assert registry._previous_context['session_id'] == session_id
    finally:
        registry.cleanup()
