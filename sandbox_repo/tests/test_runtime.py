import asyncio
import importlib
import json
import queue
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import local_jupyter_session as js
import request_runtime as runtime


class FakeLock:
    def __init__(self, conn, name, timeout, blocking_timeout, **kwargs):
        self.conn, self.name, self.wait = conn, name, blocking_timeout
        self.local = SimpleNamespace(token=f'token-{id(self)}'.encode())
        with conn.guard:
            self.lock = conn.locks.setdefault(name, threading.Lock())

    def acquire(self):
        acquired = self.lock.acquire(timeout=max(0, self.wait))
        if acquired:
            self.conn.values[self.name] = self.local.token
        return acquired

    def release(self):
        self.conn.values.pop(self.name, None)
        self.lock.release()


class FakeRedis:
    def __init__(self):
        self.values, self.locks = {}, {}
        self.guard = threading.Lock()
        self.get_error = self.commit_error = None

    def lock(self, name, **kwargs):
        return FakeLock(self, name, **kwargs)

    def get(self, key):
        if self.get_error:
            raise self.get_error
        return self.values.get(key)

    def exists(self, key):
        return key in self.values

    def eval(self, script, nkeys, *args):
        keys, values = args[:nkeys], args[nkeys:]
        if self.values.get(keys[0]) != values[0]:
            return 0
        if script == runtime.CLEAR:
            self.values.pop(keys[1], None)
            self.values[keys[2]] = '1'
            return 1
        if self.commit_error:
            raise self.commit_error
        if time.time() * 1000 > values[3]:
            return -1
        self.values[keys[1]] = values[1]
        return 1


@pytest.fixture
def client():
    return SimpleNamespace(conn=FakeRedis())


@pytest.fixture
def sessions(monkeypatch):
    created = []

    class Session:
        def __init__(self, **kwargs):
            self.closed = False
            self.state = 'old'
            self.deadline = kwargs['deadline']
            created.append(self)

        def load(self, state):
            self.state = state

        def execute(self, code, **kwargs):
            self.state = code
            return {'stdout': code, 'stderr': '', 'images': []}

        def dump(self):
            return self.state

        def close(self):
            self.closed = True

    monkeypatch.setattr(runtime, 'LocalJupyterSession', Session)
    return Session, created


def execute(client, code='new', timeout=1, session='sample'):
    return asyncio.run(runtime.run_request(client, session, code, timeout))


def test_success_and_python_error_preserve_state_and_cleanup(client, sessions):
    Session, created = sessions
    client.conn.values['sample'] = b'previous'
    result = execute(client)
    assert result['status'] == 'success' and result['state_committed']
    assert client.conn.values['sample'] == 'new' and created[0].closed
    Session.execute = lambda self, *a, **k: {'stdout': '', 'stderr': 'NameError: missing',
                                            'execution_status': 'error', 'images': []}
    result = execute(client)
    assert result['status'] == 'success'  # Execution error remains visible in output.
    assert 'NameError' in result['output']['stderr'] and created[-1].closed


@pytest.mark.parametrize('failure', ['load', 'execute', 'dump'])
def test_failure_closes_kernel_and_does_not_replace_state(client, sessions, failure):
    Session, created = sessions
    client.conn.values['sample'] = b'previous'
    def fail(*args, **kwargs):
        raise ValueError('injected failure')
    setattr(Session, failure, fail)
    result = execute(client)
    assert result['status'] == 'error' and not result['state_committed']
    assert result['session_usable'] and created[0].closed
    assert client.conn.values['sample'] == b'previous'
    assert not client.conn.values.get(runtime.session_keys('sample')[0])


def test_redis_read_failure_never_starts_kernel(client, sessions):
    client.conn.get_error = RuntimeError('redis unavailable')
    assert execute(client)['status'] == 'error'
    assert not sessions[1]


def test_uncertain_commit_quarantines_and_closes(client, sessions):
    client.conn.commit_error = TimeoutError('reply lost')
    result = execute(client)
    assert not result['session_usable'] and sessions[1][0].closed


def test_expired_lease_never_commits(client, sessions):
    Session, created = sessions
    def lose_lease(self):
        client.conn.values.pop(runtime.session_keys('sample')[0], None)
        return 'new'
    Session.dump = lose_lease
    result = execute(client)
    assert result['status'] == 'error' and not result['session_usable']
    assert 'sample' not in client.conn.values and created[0].closed


def test_cleanup_failure_does_not_skip_lock_release(client, sessions):
    def fail(self):
        raise RuntimeError('cleanup failed')
    sessions[0].close = fail
    result = execute(client)
    assert result['status'] == 'success' and not result['session_usable']
    assert 'cleanup_error' in result['timing']
    assert runtime.session_keys('sample')[0] not in client.conn.values


def test_distinct_sessions_parallel_same_session_serial(client, sessions):
    Session, created = sessions
    active, maximum, active_by_session = 0, 0, set()
    guard = threading.Lock()
    def work(self, code, **kwargs):
        nonlocal active, maximum
        with guard:
            assert code not in active_by_session
            active_by_session.add(code)
            active += 1
            maximum = max(maximum, active)
        time.sleep(.04)
        with guard:
            active -= 1
            active_by_session.remove(code)
        self.state = code
        return {'stdout': code, 'stderr': '', 'images': []}
    Session.execute = work
    async def run():
        return await asyncio.gather(*(runtime.run_request(client, name, name, 2)
                                      for name in ['a', 'a', 'b', 'c']))
    results = asyncio.run(run())
    assert maximum >= 2 and all(r['status'] == 'success' for r in results)
    assert all(s.closed for s in created)


def test_waiting_session_deadline_does_not_execute(client, sessions):
    lock = client.conn.lock(runtime.session_keys('sample')[0], timeout=1, blocking_timeout=.1)
    assert lock.acquire()
    try:
        result = execute(client, timeout=.03)
        assert result['status'] == 'error' and not sessions[1]
        assert result['execution_time'] < .2
    finally:
        lock.release()


def test_clear_uses_same_lock_and_tombstone(client, sessions):
    assert execute(client)['status'] == 'success'
    assert runtime.clear_request(client, 'sample')['status'] == 'success'
    assert 'sample' not in client.conn.values
    result = execute(client)
    assert result['status'] == 'error' and not result['session_usable']
    assert len(sessions[1]) == 1


def bare_session(client):
    session = js.LocalJupyterSession.__new__(js.LocalJupyterSession)
    session._client = client
    session._default_timeout = .04
    session._deadline = None
    session._owns_kernel = False
    session._km = None
    session._closed = False
    return session


def test_output_messages_do_not_reset_deadline():
    class Client:
        def execute(self, *a, **k):
            return 'request'
        def get_iopub_msg(self, timeout):
            time.sleep(min(.008, timeout))
            return {'parent_header': {'msg_id': 'request'}, 'msg_type': 'stream',
                    'content': {'name': 'stdout', 'text': 'progress'}}
        def get_shell_msg(self, **kwargs):
            pytest.fail('must not wait a second full timeout after output deadline')
    session = bare_session(Client())
    started = time.monotonic()
    with pytest.raises(js.SandboxDeadlineExceeded):
        session.execute('print progress', timeout=.04)
    assert time.monotonic() - started < .15


def test_shell_wait_shares_remaining_deadline():
    client = MagicMock()
    client.execute.return_value = 'id'
    def iopub(timeout):
        time.sleep(.025)
        return {'parent_header': {'msg_id': 'id'}, 'msg_type': 'status',
                'content': {'execution_state': 'idle'}}
    client.get_iopub_msg.side_effect = iopub
    client.get_shell_msg.side_effect = queue.Empty
    with pytest.raises(js.SandboxDeadlineExceeded):
        bare_session(client).execute('pass', timeout=.04)
    assert 0 < client.get_shell_msg.call_args.kwargs['timeout'] < .03


def test_partial_kernel_startup_is_closed(monkeypatch):
    manager = MagicMock()
    manager.blocking_client.return_value.wait_for_ready.side_effect = RuntimeError('startup failed')
    monkeypatch.setattr('jupyter_client.KernelManager', lambda: manager)
    with pytest.raises(RuntimeError, match='startup failed'):
        js.LocalJupyterSession(timeout=1)
    manager.shutdown_kernel.assert_called_once_with(now=True)
    manager.blocking_client.return_value.stop_channels.assert_called_once()


def test_kernel_shutdown_failure_is_visible_and_close_remains_retryable():
    session = bare_session(MagicMock())
    session._owns_kernel = True
    session._km = MagicMock()
    session._km.shutdown_kernel.side_effect = [RuntimeError('shutdown failed'), None]
    with pytest.raises(RuntimeError, match='shutdown failed'):
        session.close()
    assert not session._closed
    session.close()
    assert session._closed and session._km.shutdown_kernel.call_count == 2
    session.close()
    assert session._km.shutdown_kernel.call_count == 2


@pytest.mark.parametrize('timeout', [0, -1, float('nan'), float('inf')])
def test_invalid_timeout_is_rejected(timeout):
    with pytest.raises(ValueError):
        js.LocalJupyterSession(timeout=timeout)


def test_api_capacity_queue_and_disconnect_hold_slot(client, sessions, monkeypatch):
    api = importlib.import_module('fast_api_server_v2')
    Session, created = sessions
    running, release = threading.Event(), threading.Event()
    original = Session.execute
    def blocking(self, code, **kwargs):
        running.set()
        assert release.wait(1)
        return original(self, code, **kwargs)
    Session.execute = blocking
    class Request:
        async def json(self):
            return {'session_id': 'a', 'code': 'work', 'timeout': .08}
    async def run():
        monkeypatch.setattr(api, 'RC', client)
        monkeypatch.setattr(api, 'SEM', asyncio.Semaphore(1))
        first = asyncio.create_task(api.query(Request()))
        while not running.is_set():
            await asyncio.sleep(.001)
        first.cancel()
        with pytest.raises(asyncio.CancelledError):
            await first
        second = await api.query(Request())
        assert 'capacity' in second['error']
        assert len(created) == 1
        release.set()
        await asyncio.gather(*list(api.ACTIVE_REQUESTS))
        assert api.SEM._value == 1
    try:
        asyncio.run(run())
    finally:
        release.set()
