"""Private-port test app. Never used by the production launcher."""
import os
from types import SimpleNamespace

import redis
from redis.backoff import NoBackoff
from redis.retry import Retry

import fast_api_server_v2 as api
import request_runtime


class FaultSession(request_runtime.LocalJupyterSession):
    def execute(self, code, **kwargs):
        if code.startswith('# inject_snapshot_failure\n'):
            self._test_fail_snapshot = True
        return super().execute(code, **kwargs)

    def dump(self):
        if getattr(self, '_test_fail_snapshot', False):
            raise ValueError('injected snapshot failure after actual kernel execution')
        return super().dump()


request_runtime.LocalJupyterSession = FaultSession
api.RC = SimpleNamespace(conn=redis.Redis(
    host='127.0.0.1', port=int(os.environ['SANDBOX_SMOKE_REDIS_PORT']),
    socket_timeout=5, socket_connect_timeout=5, protocol=2, retry=Retry(NoBackoff(), 0)))
app = api.app
