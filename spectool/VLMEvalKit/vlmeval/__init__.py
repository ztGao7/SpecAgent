import ssl
import warnings

# Ignore pkg_resources warning due to jieba depends on it.
warnings.filterwarnings("ignore", category=UserWarning, message="pkg_resources is deprecated")

# Temporarily bypass SSL certificate verification to download files from oss.
ssl._create_default_https_context = ssl._create_unverified_context


def load_env():
    import logging
    import os
    from pathlib import Path
    logging.basicConfig(
        format='[%(asctime)s] %(levelname)s - %(filename)s: %(funcName)s - %(lineno)d: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S')
    pth = Path(__file__).parent.parent / '.env'
    if not pth.exists():
        # Public launchers export the root .env. A second, vendored .env is optional.
        logging.debug(f'No optional EvalKit .env at {pth}; using exported configuration.')
        return

    from dotenv import dotenv_values
    values = dotenv_values(str(pth))
    for k, v in values.items():
        if v is not None and len(v):
            os.environ.setdefault(k, v)
    logging.info(f'Optional EvalKit configuration loaded from {pth}')


load_env()

try:
    import torch  # noqa: F401
except ImportError:
    pass

# from .api import *  # noqa: F401, F403, E402
# from .config import *  # noqa: F401, F403, E402
# from .dataset import *  # noqa: F401, F403, E402
# from .smp import *  # noqa: F401, F403, E402
from .tools import cli  # noqa: F401, E402

# from .utils import *  # noqa: F401, F403, E402
# from .vlm import *  # noqa: F401, F403, E402

__version__ = '0.2rc1'
