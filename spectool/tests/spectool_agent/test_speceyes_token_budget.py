"""SpecEyes fallback shares Spectool's ordinary request token budget."""
import ast
from functools import partial
import os
from pathlib import Path
import subprocess
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from spectool_agent.speceyes import SpecEyesOrchestrator
from spectool_agent.types import DEFAULT_MAX_TOKENS
from vlmeval.api.spectool_speceyes import SpecEyesSpectoolAPI


ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(autouse=True)
def clean_environment(monkeypatch):
    for name in (
        "SPECEYE_MAIN_MAX_TOKENS", "SPECTOOL_MAX_TOKENS",
        "SPECTOOL_MAX_TOKENS_CAP", "SPECEYE_ROUTER_MAX_TOKENS",
        "SPECEYE_DIRECT_MAX_TOKENS", "SPECEYE_MAX_STEPS",
        "SPECTOOL_MAX_STEPS", "SPECTOOL_ENABLE_THINKING",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("SPECTOOL_AGENT_ROOT", str(ROOT))


def registered_builder():
    tree = ast.parse((ROOT / "VLMEvalKit/vlmeval/config.py").read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.Dict):
            for key, value in zip(node.keys, node.values):
                if isinstance(key, ast.Constant) and key.value == "SpecEyes-Qwen3.5":
                    return eval(compile(ast.Expression(value), "config.py", "eval"), {
                        "partial": partial, "os": os,
                        "api": SimpleNamespace(SpecEyesSpectoolAPI=SpecEyesSpectoolAPI),
                    })
    raise AssertionError("Missing SpecEyes registration")


@pytest.mark.parametrize("specific,shared,explicit,expected", [
    (None, None, None, DEFAULT_MAX_TOKENS),
    ("", "", None, DEFAULT_MAX_TOKENS),
    (None, "3072", None, 3072),
    ("2048", "3072", None, 2048),
    (None, None, 1024, 1024),
    ("2048", "3072", 1024, 2048),
])
def test_budget_reaches_main_request(monkeypatch, specific, shared, explicit, expected):
    for name, value in (("SPECEYE_MAIN_MAX_TOKENS", specific),
                        ("SPECTOOL_MAX_TOKENS", shared)):
        if value is not None:
            monkeypatch.setenv(name, value)
    client = MagicMock()
    message = SimpleNamespace(content="<answer>A</answer>", tool_calls=[], model_dump=lambda: {})
    client.chat.completions.create.return_value = SimpleNamespace(choices=[
        SimpleNamespace(message=message, finish_reason="stop", logprobs=None)])
    monkeypatch.setattr("spectool_agent.model.OpenAI", MagicMock(return_value=client))
    builder = registered_builder()
    assert "max_tokens" not in builder.keywords
    api = builder(**({"max_tokens": explicit} if explicit is not None else {}))
    api.main_model.generate([{"role": "user", "content": "Question"}])
    assert api.cfg.max_tokens == expected
    assert client.chat.completions.create.call_args.kwargs["max_tokens"] == expected
    orchestrator = SpecEyesOrchestrator(api.main_model, api.draft_models[0], api.cfg)
    assert orchestrator.router_max_tokens == 10
    assert orchestrator.direct_max_tokens == 20


@pytest.mark.parametrize("override", [None, "2048"])
def test_launcher_does_not_inject_legacy_budget(override):
    env = dict(os.environ, SPECTOOL_START_TEST_SOURCE_ONLY="1")
    if override is not None:
        env["SPECEYE_MAIN_MAX_TOKENS"] = override
    result = subprocess.run([
        "bash", "-c",
        'source "$1"; export SPECEYE_MAIN_MAX_TOKENS; printf "%s" "${SPECEYE_MAIN_MAX_TOKENS:-}"',
        "bash", str(ROOT.parent / "start_test.sh"),
    ], env=env, check=True, capture_output=True, text=True)
    assert result.stdout == (override or "")
