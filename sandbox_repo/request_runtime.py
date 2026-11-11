"""Bounded execution, cross-worker session ordering, and state commit fencing."""
import asyncio
import contextlib
import hashlib
import json
import logging
import os
import threading
import time
import uuid

from local_jupyter_session import LocalJupyterSession, SandboxDeadlineExceeded, remaining

LOG = logging.getLogger("uvicorn.error")
STATE_TTL = 6 * 3600
# Keep the session lease during forced kernel shutdown and bounded Redis IO.
CLEANUP_GRACE = 20

COMMIT = """
if redis.call('GET', KEYS[1]) ~= ARGV[1] then return 0 end
local now = redis.call('TIME')
if tonumber(now[1])*1000 + math.floor(tonumber(now[2])/1000) > tonumber(ARGV[4]) then
    return -1
end
redis.call('SET', KEYS[2], ARGV[2], 'EX', ARGV[3])
return 1
"""
CLEAR = """
if redis.call('GET', KEYS[1]) ~= ARGV[1] then return 0 end
redis.call('DEL', KEYS[2])
redis.call('SET', KEYS[3], '1', 'EX', ARGV[2])
return 1
"""


def session_keys(session_id):
    digest = hashlib.sha256(session_id.encode()).hexdigest()
    return 'spectool:lock:' + digest, 'spectool:closed:' + digest


def _phase(metrics, name, fn, *args, **kwargs):
    started = time.monotonic()
    metrics[name + '_start_unix'] = time.time()
    try:
        return fn(*args, **kwargs)
    finally:
        metrics[name + '_s'] = time.monotonic() - started


def execute_request(client, session_id, code, timeout, deadline, request_id, started):
    metrics = {'request_id': request_id, 'session_hash': session_keys(session_id)[0][-16:],
               'worker_pid': os.getpid(), 'thread_id': threading.get_ident(),
               'code_sha256': hashlib.sha256(code.encode()).hexdigest(),
               'timeout_s': timeout, 'admission_wait_s': time.monotonic() - started}
    result = {'status': 'error', 'output': {}, 'request_id': request_id,
              'state_committed': False, 'session_usable': True, 'protocol_version': 2}
    session, lock, acquired = None, None, False
    commit_started = False
    try:
        conn = client.conn
        lock_name, closed_name = session_keys(session_id)
        lock = conn.lock(lock_name, timeout=remaining(deadline) + CLEANUP_GRACE,
                         blocking_timeout=remaining(deadline), sleep=.02)
        acquired = _phase(metrics, 'session_wait', lock.acquire)
        if not acquired:
            raise SandboxDeadlineExceeded("Deadline exceeded waiting for this sandbox session")
        remaining(deadline)
        if conn.exists(closed_name):
            result['session_usable'] = False
            raise RuntimeError("Sandbox session was closed; start a fresh session")
        state = _phase(metrics, 'state_read', conn.get, session_id)
        remaining(deadline)
        session = _phase(metrics, 'kernel_start', LocalJupyterSession,
                         timeout=remaining(deadline), deadline=deadline)
        if state:
            state = state.decode('utf-8') if isinstance(state, bytes) else state
            _phase(metrics, 'restore', session.load, state)
        remaining(deadline)
        result['output'] = _phase(metrics, 'execute', session.execute, code,
                                  timeout=remaining(deadline))
        state = _phase(metrics, 'snapshot', session.dump)
        remaining(deadline)
        # An expired owner must never overwrite a later request's state.
        deadline_ms = int((time.time() + remaining(deadline)) * 1000)
        commit_started = True
        committed = _phase(metrics, 'commit', conn.eval, COMMIT, 2, lock_name, session_id,
                           lock.local.token, state, STATE_TTL, deadline_ms)
        if committed != 1:
            commit_started = False  # Redis confirmed that no write was made.
            if committed == -1:
                raise SandboxDeadlineExceeded("Deadline exceeded before state commit")
            result['session_usable'] = False
            raise RuntimeError("Sandbox session lease was lost before state commit")
        result['state_committed'] = True
        result['status'] = 'success'
    except Exception as exc:
        result['error'] = f'{type(exc).__name__}: {exc}'
        if commit_started and not result['state_committed']:
            # A lost Redis reply could mean the write happened. Do not pretend
            # this was a rollback or allow automatic replay on that session.
            result['session_usable'] = False
        metrics['error_type'] = type(exc).__name__
    finally:
        if session is not None:
            try:
                _phase(metrics, 'cleanup', session.close)
            except Exception as exc:
                metrics['cleanup_error'] = type(exc).__name__
                result['session_usable'] = False
                LOG.exception('Sandbox kernel cleanup failed for %s', request_id)
        if acquired:
            with contextlib.suppress(Exception):
                lock.release()
        metrics.update(status=result['status'], state_committed=result['state_committed'],
                       total_s=time.monotonic() - started)
        result['timing'] = metrics
        result['execution_time'] = metrics['total_s']
        LOG.info('[sandbox_request] %s', json.dumps(metrics, sort_keys=True))
    return result


async def run_request(client, session_id, code, timeout, *, deadline=None,
                      request_id=None, started=None):
    started = time.monotonic() if started is None else started
    deadline = started + timeout if deadline is None else deadline
    return await asyncio.to_thread(execute_request, client, session_id, code, timeout,
                                   deadline, request_id or uuid.uuid4().hex, started)


def clear_request(client, session_id, timeout=10):
    deadline = time.monotonic() + timeout
    conn = client.conn
    lock_name, closed_name = session_keys(session_id)
    lock = conn.lock(lock_name, timeout=timeout + CLEANUP_GRACE,
                     blocking_timeout=remaining(deadline), sleep=.02)
    if not lock.acquire():
        return {'status': 'error', 'error': 'Session is still executing; not cleared'}
    try:
        remaining(deadline)
        cleared = conn.eval(CLEAR, 3, lock_name, session_id, closed_name,
                            lock.local.token, STATE_TTL)
        return {'status': 'success' if cleared == 1 else 'error'}
    finally:
        with contextlib.suppress(Exception):
            lock.release()
