import os
import sys
import types

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_HERE, ".."))
sys.path.insert(0, _REPO_ROOT)


def _install_vlmeval_stub():
    """Inject minimal vlmeval / vlmeval.api / vlmeval.api.base stubs into
    sys.modules so the wrapper at VLMEvalKit/vlmeval/api/spectool_agent.py
    can be imported in tests without pulling vlmeval/api/__init__.py and
    its heavy ML dependency chain (torch, transformers, datasets, ...).

    `vlmeval.api` is given a __path__ pointing at the real
    VLMEvalKit/vlmeval/api directory so that
    `from vlmeval.api.spectool_agent import SpectoolAgentAPI` resolves to
    the actual wrapper source file.
    """
    if "vlmeval" in sys.modules:
        return

    vlmeval_api_dir = os.path.join(
        _REPO_ROOT, "VLMEvalKit", "vlmeval", "api"
    )

    vlmeval_mod = types.ModuleType("vlmeval")
    vlmeval_mod.__path__ = [os.path.join(_REPO_ROOT, "VLMEvalKit", "vlmeval")]

    vlmeval_api_mod = types.ModuleType("vlmeval.api")
    vlmeval_api_mod.__path__ = [vlmeval_api_dir]

    vlmeval_api_base_mod = types.ModuleType("vlmeval.api.base")

    class _BaseAPIStub:
        allowed_types = ["text", "image", "video"]
        INTERLEAVE = True
        INSTALL_REQ = False
        is_api = True

        def __init__(
            self,
            retry=10,
            wait=1,
            system_prompt=None,
            verbose=True,
            fail_msg="Failed to obtain answer via API.",
            **kwargs,
        ):
            self.retry = retry
            self.wait = wait
            self.system_prompt = system_prompt
            self.verbose = verbose
            self.fail_msg = fail_msg

    vlmeval_api_base_mod.BaseAPI = _BaseAPIStub

    sys.modules["vlmeval"] = vlmeval_mod
    sys.modules["vlmeval.api"] = vlmeval_api_mod
    sys.modules["vlmeval.api.base"] = vlmeval_api_base_mod


_install_vlmeval_stub()
