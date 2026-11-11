"""Exercise launcher/registry/wrapper/request parameter propagation offline."""
import argparse
import ast
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from functools import partial
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from PIL import Image

from spectool_agent.draft_model import DraftModel
from spectool_agent.model import RequestCancellation, VLLMModel
from spectool_agent.pool import DraftPool
from spectool_agent.speculative_agent import CancellableTask, SpeculativeAgentLoop
from spectool_agent.types import AgentConfig, DEFAULT_MAX_TOKENS, Trajectory
from vlmeval.api.spectool_speculative import SpeculativeSpectoolAgentAPI

ROOT = Path(__file__).resolve().parents[2]
VLM_ROOT = ROOT / "VLMEvalKit"
MODELS = ["Agent-Qwen3.5-27B", "Agent-Qwen3.5-9B", "Spectool-Qwen3.5"]


@pytest.fixture(autouse=True)
def runtime_environment(monkeypatch):
    for name in (
        "SPECTOOL_MAX_TOKENS", "SPECTOOL_MAX_TOKENS_CAP",
        "SPECTOOL_DRAFT_MAX_TOKENS_CAP", "SPECTOOL_FORCED_MAX_TOKENS_CAP",
        "SPECTOOL_API_TIMEOUT", "SPECTOOL_API_RETRY", "SPECTOOL_DRAFT_TIMEOUT",
        "SPECTOOL_OPENAI_MAX_RETRIES", "SPECTOOL_FALLBACK_ONLY",
        "MAX_TOKENS", "EVAL_RETRY", "SPECTOOL_ENABLE_THINKING",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("SPECTOOL_AGENT_ROOT", str(ROOT))


def response(content="<answer>A</answer>"):
    message = SimpleNamespace(content=content, tool_calls=[], model_dump=lambda: {})
    return SimpleNamespace(choices=[SimpleNamespace(
        message=message, finish_reason="stop", logprobs=None)])


@pytest.fixture
def clients(monkeypatch):
    main, draft = MagicMock(), MagicMock()
    main.chat.completions.create.return_value = response()
    draft.chat.completions.create.return_value = response()
    main_factory, draft_factory = MagicMock(return_value=main), MagicMock(return_value=draft)
    monkeypatch.setattr("spectool_agent.model.OpenAI", main_factory)
    monkeypatch.setattr("spectool_agent.draft_model.OpenAI", draft_factory)
    return SimpleNamespace(main=main, draft=draft, main_factory=main_factory, draft_factory=draft_factory)


def registered_builder(name):
    # Evaluate the real registration without importing unrelated ML adapters.
    tree = ast.parse((VLM_ROOT / "vlmeval/config.py").read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.Dict):
            for key, value in zip(node.keys, node.values):
                if isinstance(key, ast.Constant) and key.value == name:
                    return eval(compile(ast.Expression(value), "config.py", "eval"), {
                        "partial": partial, "os": os, "SPECTOOL_AGENT_MAX_STEPS": 15,
                        "api": SimpleNamespace(SpeculativeSpectoolAgentAPI=SpeculativeSpectoolAgentAPI),
                    })
    raise AssertionError(f"Registration missing: {name}")


def cli_helpers():
    names = {"_StoreExplicitGenerationArg", "_apply_spectool_generation_args",
             "parse_args", "parse_reuse_aux_arg"}
    tree = ast.parse((VLM_ROOT / "run.py").read_text())
    nodes = [node for node in tree.body if getattr(node, "name", None) in names]
    namespace = {"argparse": argparse, "os": os, "logger": MagicMock()}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), "run.py", "exec"), namespace)
    return namespace


def parse_cli(monkeypatch, *args):
    helpers = cli_helpers()
    monkeypatch.setattr(sys, "argv", ["run.py", *args])
    return helpers, helpers["parse_args"]()


@pytest.mark.parametrize("name", MODELS)
@pytest.mark.parametrize("tokens", [512, 3072, 8192])
def test_registered_cli_reaches_actual_generation(monkeypatch, clients, name, tokens):
    monkeypatch.setenv("SPECTOOL_MAX_TOKENS", "256")
    monkeypatch.setenv("SPECTOOL_API_TIMEOUT", "99")
    helpers, args = parse_cli(monkeypatch, "--max-tokens", str(tokens), "--timeout=17.5")
    builder = registered_builder(name)
    helpers["_apply_spectool_generation_args"](builder, args)
    api = builder()
    api.main_model.generate([{"role": "user", "content": "Q"}])
    request = clients.main.chat.completions.create.call_args.kwargs
    assert request["max_tokens"] == tokens
    assert request["timeout"] == 17.5
    assert api.runtime_config["main_max_tokens"] == tokens
    if name == "Spectool-Qwen3.5":
        captured = {}
        async def complete(kwargs, cancel):
            captured.update(kwargs)
            return response()
        api.draft_model._create_cancellable_completion = complete
        loop = SpeculativeAgentLoop(api.main_model, api.draft_model, MagicMock(), api.cfg)
        try:
            loop._draft_loop([{"role": "user", "content": "Q"}], Image.new("RGB", (2, 2)),
                             DraftPool(), CancellableTask(), Trajectory())
            assert captured["max_tokens"] == tokens
            assert captured["timeout"] == 200
            assert captured["extra_body"] == {"repetition_penalty": 1.1}
        finally:
            loop.shutdown()


@pytest.mark.parametrize("name", MODELS)
def test_absent_cli_keeps_registration_environment(monkeypatch, clients, name):
    monkeypatch.setenv("SPECTOOL_MAX_TOKENS", "768")
    monkeypatch.setenv("SPECTOOL_API_TIMEOUT", "27")
    helpers, args = parse_cli(monkeypatch)
    assert args.max_tokens == 32768 and args.timeout == 1800
    builder = registered_builder(name)
    helpers["_apply_spectool_generation_args"](builder, args)
    api = builder()
    assert (api.cfg.max_tokens, api.cfg.timeout) == (768, 27)


@pytest.mark.parametrize("name", MODELS)
def test_registered_default_comes_from_request_config(monkeypatch, clients, name):
    helpers, args = parse_cli(monkeypatch)
    builder = registered_builder(name)
    assert "max_tokens" not in builder.keywords
    helpers["_apply_spectool_generation_args"](builder, args)
    api = builder()
    api.main_model.generate([{"role": "user", "content": "Q"}])
    assert DEFAULT_MAX_TOKENS == 3072
    assert api.cfg.max_tokens == DEFAULT_MAX_TOKENS
    assert clients.main.chat.completions.create.call_args.kwargs["max_tokens"] == DEFAULT_MAX_TOKENS
    if api.draft_model is not None:
        api.draft_model.generate_full([])
        assert clients.draft.chat.completions.create.call_args.kwargs["max_tokens"] == DEFAULT_MAX_TOKENS


def test_other_model_registrations_are_unchanged(monkeypatch):
    helpers, args = parse_cli(monkeypatch, "--max-tokens=512", "--timeout", "17")
    builder = partial(dict, max_tokens=9, timeout=10)
    helpers["_apply_spectool_generation_args"](builder, args)
    assert builder.keywords == {"max_tokens": 9, "timeout": 10}


@pytest.mark.parametrize("env_retry,cli_retry,expected", [(None, None, 1), ("3", None, 3), ("3", "2", 2)])
def test_registered_retry_precedence(monkeypatch, clients, env_retry, cli_retry, expected):
    if env_retry is not None:
        monkeypatch.setenv("SPECTOOL_API_RETRY", env_retry)
    cli = ["--retry", cli_retry] if cli_retry is not None else []
    helpers, args = parse_cli(monkeypatch, *cli)
    builder = registered_builder("Spectool-Qwen3.5")
    helpers["_apply_spectool_generation_args"](builder, args)
    assert builder().retry == expected


@pytest.mark.parametrize("explicit", [False, True])
def test_wrapper_environment_and_explicit_precedence(monkeypatch, clients, explicit):
    monkeypatch.setenv("SPECTOOL_MAX_TOKENS", "512")
    monkeypatch.setenv("SPECTOOL_API_TIMEOUT", "17")
    monkeypatch.setenv("SPECTOOL_API_RETRY", "2")
    monkeypatch.setenv("SPECTOOL_DRAFT_TIMEOUT", "19")
    kwargs = dict(max_tokens=700, timeout=23, retry=4, draft_timeout=31) if explicit else {}
    api = SpeculativeSpectoolAgentAPI(model="fake", api_base="http://unused/v1", **kwargs)
    expected = (700, 23, 4, 31) if explicit else (512, 17, 2, 19)
    assert (api.cfg.max_tokens, api.cfg.timeout, api.retry, api.draft_model._timeout) == expected


@pytest.mark.parametrize("name,value", [("max_tokens", 0), ("timeout", 0),
                                       ("timeout", float("nan")), ("retry", 0)])
def test_invalid_runtime_parameters_fail_early(clients, name, value):
    with pytest.raises(ValueError):
        SpeculativeSpectoolAgentAPI(model="fake", api_base="http://unused/v1", **{name: value})
    clients.main_factory.assert_not_called()


@pytest.mark.parametrize("retries", [0, 2])
@pytest.mark.parametrize("role", ["main", "draft"])
def test_sync_and_async_sdk_retries_match(monkeypatch, clients, retries, role):
    import asyncio
    monkeypatch.setenv("SPECTOOL_OPENAI_MAX_RETRIES", str(retries))
    async_client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=AsyncMock(return_value=response()))),
        close=AsyncMock())
    factory = MagicMock(return_value=async_client)
    module = "spectool_agent.model" if role == "main" else "spectool_agent.draft_model"
    monkeypatch.setattr(f"{module}.AsyncOpenAI", factory)
    model = (VLLMModel(AgentConfig(api_base="http://unused/v1", model_name="fake"))
             if role == "main" else DraftModel("http://unused/v1", "fake"))
    sync_factory = clients.main_factory if role == "main" else clients.draft_factory
    assert sync_factory.call_args.kwargs["max_retries"] == retries
    asyncio.run(model._create_cancellable_completion({"model": "fake"}, RequestCancellation()))
    assert factory.call_args.kwargs["max_retries"] == retries
    async_client.close.assert_awaited_once()


def test_explicit_role_caps_remain_visible(monkeypatch, clients, capsys):
    monkeypatch.setenv("SPECTOOL_MAX_TOKENS_CAP", "600")
    monkeypatch.setenv("SPECTOOL_DRAFT_MAX_TOKENS_CAP", "700")
    api = SpeculativeSpectoolAgentAPI(model="fake", api_base="http://unused/v1", max_tokens=8192)
    api.main_model.generate([{"role": "user", "content": "Q"}])
    api.draft_model.generate_full([], max_tokens=api.cfg.max_tokens)
    assert clients.main.chat.completions.create.call_args.kwargs["max_tokens"] == 600
    assert clients.draft.chat.completions.create.call_args.kwargs["max_tokens"] == 700
    logged = json.loads(capsys.readouterr().out.split("[spectool-config] ")[-1])
    assert logged["requested_max_tokens"] == 8192
    assert (logged["main_max_tokens"], logged["draft_max_tokens"]) == (600, 700)


def test_judge_and_forced_budgets_unchanged(clients):
    model = VLLMModel(AgentConfig(api_base="http://unused/v1", model_name="fake", max_tokens=8192))
    clients.main.chat.completions.create.return_value = response("GOOD")
    model.judge_good_bad([])
    assert clients.main.chat.completions.create.call_args.kwargs["max_tokens"] == 4
    model.assess_draft_step([], choices=["A", "B"])
    assert clients.main.chat.completions.create.call_args.kwargs["max_tokens"] == 128
    model.generate_forced([], choices=["A", "B"])
    assert clients.main.chat.completions.create.call_args.kwargs["max_tokens"] == 4096


def test_runtime_config_is_saved_with_result(monkeypatch, clients):
    api = SpeculativeSpectoolAgentAPI(model="fake", api_base="http://unused/v1",
        fallback_only=True, max_tokens=512)
    monkeypatch.setattr(api, "_parse_vlmeval_inputs", lambda _: ([Image.new("RGB", (2, 2))], "Q"))
    monkeypatch.setattr(SpeculativeAgentLoop, "run", lambda *a, **k: Trajectory(final_answer="A"))
    code, answer, extra = api.generate_inner([{"type": "text", "value": "Q"}], dataset="VStarBench")
    assert (code, answer) == (0, "A")
    assert extra["runtime_config"]["main_max_tokens"] == 512
    assert extra["runtime_config"]["draft_max_tokens"] is None


def test_eval_script_does_not_override_request_token_budget(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    (repo / "scripts").mkdir(parents=True)
    (repo / "VLMEvalKit").mkdir()
    script = repo / "scripts/eval.sh"
    shutil.copy2(ROOT / "scripts/eval.sh", script)
    fake_python = tmp_path / "fake_python"
    fake_python.write_text(f"#!{sys.executable}\n" + """
import json, os, sys
if sys.argv[1] == 'run.py':
    print('CAPTURE=' + json.dumps({'argv': sys.argv[2:], 'env': {
        key: os.environ.get(key) for key in ['SPECTOOL_MAX_TOKENS', 'SPECTOOL_API_TIMEOUT', 'SPECTOOL_API_RETRY']}}))
""")
    fake_python.chmod(0o755)
    fake_curl = tmp_path / "curl"
    fake_curl.write_text("#!/bin/sh\nexit 0\n")
    fake_curl.chmod(0o755)
    env = {**os.environ, "PATH": f"{tmp_path}:{os.environ['PATH']}", "VLMEVAL_PY": str(fake_python),
           "BENCH": "mmstar", "MAX_TOKENS": "512", "SPECTOOL_API_TIMEOUT": "17.5",
           "SPECTOOL_API_RETRY": "3", "EVAL_RETRY": "2", "LMUData": str(tmp_path / "data")}
    result = subprocess.run(["bash", str(script)], env=env, capture_output=True, text=True, check=True)
    captured = json.loads(next(line[8:] for line in result.stdout.splitlines() if line.startswith("CAPTURE=")))
    args = captured["argv"]
    assert "--max-tokens" not in args
    assert args[args.index("--timeout") + 1] == "17.5"
    assert args[args.index("--retry") + 1] == "2"
    assert captured["env"] == {"SPECTOOL_MAX_TOKENS": None, "SPECTOOL_API_TIMEOUT": "17.5",
                               "SPECTOOL_API_RETRY": "2"}


def test_start_script_does_not_set_token_budget():
    launcher = ROOT.parent / "start_test.sh"
    env = {**os.environ, "SPECTOOL_START_TEST_SOURCE_ONLY": "1",
           "SPECTOOL_API_RETRY": "3"}
    command = 'source "$1"\n'
    command += 'printf "BUDGET=%s ATTEMPTS=%s\\n" "${MAX_TOKENS-unset}" "$EVAL_RETRY"\n'
    result = subprocess.run(["bash", "-c", command, "_", str(launcher)],
                            env=env, text=True, capture_output=True, check=True)
    assert "BUDGET=unset ATTEMPTS=3" in result.stdout


def test_start_script_rejects_removed_option():
    env = {**os.environ, "SPECTOOL_START_TEST_SOURCE_ONLY": "1"}
    result = subprocess.run(
        ["bash", "-c", 'source "$1"\nparse_args --max-tokens 512', "_", str(ROOT.parent / "start_test.sh")],
        env=env, text=True, capture_output=True)
    assert result.returncode != 0
    assert "--max-tokens" in result.stderr
