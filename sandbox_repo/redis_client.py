import os
import redis
import logging
from redis.backoff import NoBackoff
from redis.retry import Retry

REDIS_HOST = os.environ.get('SANDBOX_REDIS_HOST', '127.0.0.1')
REDIS_PORT = int(os.environ.get('SANDBOX_REDIS_PORT', '26379'))
REDIS_PSWD = os.environ.get('SANDBOX_REDIS_PASSWORD') or None

class Singleton(type):
    """
    An metaclass for singleton purpose. Every singleton class should inherit from this class by 'metaclass=Singleton'.
    """
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]


class RedisClient(metaclass=Singleton):
    def __init__(self):
        self._conn = None
        self.pool = redis.ConnectionPool(protocol=2,
            host=REDIS_HOST, 
            port=REDIS_PORT,
            password=REDIS_PSWD,
            socket_connect_timeout=5,
            socket_timeout=5,
            retry=Retry(NoBackoff(), 0),
            retry_on_timeout=False,
        )

    @property
    def conn(self):
        if not self._conn:
            self._conn = self.getConnection()
        return self._conn

    def getConnection(self):
        conn = redis.Redis(
            connection_pool=self.pool, 
            decode_responses=True,
            # charset='UTF-8',
            encoding='UTF-8',
        )
        conn.set('charset', 'utf-8')
        # current_app.logger.info(f'Redis connection with {REDIS_HOST}:{REDIS_PORT} built')
        return conn
