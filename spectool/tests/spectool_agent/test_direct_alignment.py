"""Direct VQA must share Main settings without entering the agent/tool loop."""
import ast
import json
import os
from functools import partial
from pathlib import Path
import subprocess
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import httpx
from openai import OpenAI
from PIL import Image

from spectool_agent.model import pil_to_data_url
from spectool_agent.prompts import (
    DIRECT_SYSTEM_PROMPT, FINAL_ANSWER_INSTRUCTION, SYSTEM_PROMPT,
    VISUAL_REASONING_ROLE,
)
from vlmeval.api.spectool_speculative import SpeculativeSpectoolAgentAPI

ROOT = Path(__file__).resolve().parents[2]
LAUNCHER = Path(os.environ.get("SPECTOOL_TEST_LAUNCHER", ROOT.parent / "start_test.sh"))


def builder(name):
    tree = ast.parse((ROOT / "VLMEvalKit/vlmeval/config.py").read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.Dict):
            for key, value in zip(node.keys, node.values):
                if isinstance(key, ast.Constant) and key.value == name:
                    return eval(compile(ast.Expression(value), "config.py", "eval"), {
                        "partial": partial, "os": os, "SPECTOOL_AGENT_MAX_STEPS": 8,
                        "api": SimpleNamespace(SpeculativeSpectoolAgentAPI=SpeculativeSpectoolAgentAPI),
                    })
    raise AssertionError(name)


@pytest.fixture
def setup(monkeypatch, tmp_path):
    for key in list(os.environ):
        if key.startswith("SPECTOOL_"):
            monkeypatch.delenv(key)
    monkeypatch.setenv("SPECTOOL_AGENT_ROOT", str(ROOT))
    monkeypatch.setenv("SPECTOOL_API_TIMEOUT", "180")
    monkeypatch.setenv("SPECTOOL_API_RETRY", "1")
    main, draft = MagicMock(), MagicMock()
    main.chat.completions.create.return_value = response("<answer>B</answer>")
    monkeypatch.setattr("spectool_agent.model.OpenAI", lambda **kwargs: main)
    monkeypatch.setattr("spectool_agent.draft_model.OpenAI", draft)
    tools = MagicMock(side_effect=AssertionError("Direct must not register tools"))
    monkeypatch.setattr("spectool_agent.tools.register_default_tools", tools)
    loop = MagicMock(side_effect=AssertionError("Direct must not enter the loop"))
    monkeypatch.setattr("spectool_agent.speculative_agent.SpeculativeAgentLoop", loop)
    path = tmp_path / "input.png"
    Image.new("RGB", (160, 128), "red").save(path)
    inputs = [{"type": "image", "value": str(path)},
              {"type": "text", "value": "Which color?\nA. Blue\nB. Red"}]
    return SimpleNamespace(main=main, draft=draft, tools=tools, inputs=inputs, path=path)


def response(text):
    message = SimpleNamespace(content=text, tool_calls=[], model_dump=lambda: {})
    return SimpleNamespace(choices=[SimpleNamespace(message=message)])


@pytest.mark.parametrize("thinking", ["0", "1"])
def test_main_request_settings_and_images_match(setup, monkeypatch, thinking):
    monkeypatch.setenv("SPECTOOL_ENABLE_THINKING", thinking)
    direct = builder("Qwen3.5-27B-vLLM")()
    baseline = builder("Agent-Qwen3.5-27B")()
    baseline.main_model.generate([{"role": "user", "content": "Q"}], tools=[{"type": "function"}])
    baseline_request = setup.main.chat.completions.create.call_args.kwargs
    setup.main.reset_mock()
    code, answer, record = direct.generate_inner(setup.inputs, dataset="VStarBench", sample_id="1")
    request = setup.main.chat.completions.create.call_args.kwargs
    assert (code, answer) == (0, "B")
    for key in ("model", "temperature", "max_tokens", "timeout", "extra_body"):
        assert request[key] == baseline_request[key]
    assert request["max_tokens"] == 4096
    assert request["extra_body"]["chat_template_kwargs"]["enable_thinking"] == (thinking == "1")
    assert request["extra_body"]["repetition_penalty"] == 1.1
    assert "tools" not in request and "tool_choice" not in request
    assert request["messages"][0]["content"] == DIRECT_SYSTEM_PROMPT
    content = request["messages"][1]["content"]
    assert content[0]["image_url"]["url"] == pil_to_data_url(Image.open(setup.path).convert("RGB"), max_size=0)
    assert content[-1]["text"] == setup.inputs[-1]["value"]
    assert record["fallback_mode"] is False and record["tool_call_count"] == 0
    assert record["total_time"] >= record["total_inference_time"] >= 0
    assert direct.runtime_config["draft_max_tokens"] is None
    setup.draft.assert_not_called()
    setup.tools.assert_not_called()
    setup.main.chat.completions.create.assert_called_once()


@pytest.mark.parametrize("text,expected", [
    # This fixture is an A/B MCQ: numeric output is invalid. Exact matching
    # letter + option text is safely canonicalized without another request.
    ("A", None), ("<answer></answer>", None), ("<answer>42</answer>", None),
    ("<answer>B. Red</answer>", "B"), ("<answer>B</answer>", "B"),
    ('<tool_call>{"name":"sandbox","arguments":{"code":"1"}}</tool_call><answer>A</answer>', None),
])
def test_strict_parser_no_hidden_fallback(setup, text, expected):
    setup.main.chat.completions.create.return_value = response(text)
    api = builder("Qwen3.5-27B-vLLM")()
    code, answer, record = api.generate_inner(setup.inputs, dataset="VStarBench")
    if expected is None:
        assert code == 1
        assert json.loads(record)["fallback_mode"] is False
    else:
        assert (code, answer) == (0, expected)
    setup.main.chat.completions.create.assert_called_once()


def test_timeout_is_visible_without_agent_retry(setup):
    setup.main.chat.completions.create.side_effect = TimeoutError("test timeout")
    api = builder("Qwen3.5-27B-vLLM")()
    code, _, raw = api.generate_inner(setup.inputs)
    record = json.loads(raw)
    assert code == 1 and record["stop_reason"] == "exception"
    assert record["error"] == "test timeout"
    setup.main.chat.completions.create.assert_called_once()


def test_multi_image_order_and_limit(setup):
    api = builder("Qwen3.5-27B-vLLM")()
    code, _, _ = api.generate_inner([setup.inputs[0]] * 2 + setup.inputs[1:])
    content = setup.main.chat.completions.create.call_args.kwargs["messages"][1]["content"]
    assert code == 0
    assert [item["text"] for item in content if item["type"] == "text"] == [
        "[Input image 1]", "[Input image 2]", setup.inputs[-1]["value"]]
    setup.main.reset_mock()
    code, _, raw = api.generate_inner([setup.inputs[0]] * 16 + setup.inputs[1:])
    assert code == 1 and "at most 15" in json.loads(raw)["error"]
    setup.main.chat.completions.create.assert_not_called()


def test_prompts_reuse_only_applicable_rules():
    assert SYSTEM_PROMPT.startswith(VISUAL_REASONING_ROLE)
    assert DIRECT_SYSTEM_PROMPT.startswith(VISUAL_REASONING_ROLE)
    assert SYSTEM_PROMPT.endswith(FINAL_ANSWER_INSTRUCTION)
    assert DIRECT_SYSTEM_PROMPT.endswith(FINAL_ANSWER_INSTRUCTION)
    assert "sandbox(code" not in DIRECT_SYSTEM_PROMPT
    assert "web_search(" not in DIRECT_SYSTEM_PROMPT


def test_legacy_endpoint_and_runtime_overrides(setup, monkeypatch):
    monkeypatch.setenv("SPECTOOL_VLLM_BASE", "http://localhost:8123/v1/chat/completions")
    monkeypatch.setenv("SPECTOOL_MAX_TOKENS", "8192")
    api = builder("Qwen3.5-27B-vLLM")(max_tokens=3072, timeout=12, retry=2)
    assert api.cfg.api_base == "http://localhost:8123/v1"
    assert api.cfg.max_tokens == 3072 and api.cfg.timeout == 12
    assert api.runtime_config["eval_attempts"] == 2


def test_real_sdk_and_evalkit_adapter_without_network(setup):
    requests = []

    def serve(request):
        requests.append(request)
        return httpx.Response(200, json={
            "id": "offline-test", "object": "chat.completion", "created": 0,
            "model": "Qwen3.5-27B", "choices": [{"index": 0, "finish_reason": "stop",
                "message": {"role": "assistant", "content": "<answer>B</answer>"}}],
        })

    api = builder("Qwen3.5-27B-vLLM")()
    client = OpenAI(base_url="http://offline.test/v1", api_key="EMPTY", max_retries=0,
                    http_client=httpx.Client(transport=httpx.MockTransport(serve)))
    api.main_model.client = client
    try:
        code, answer, record = api.generate_inner(setup.inputs, dataset="VStarBench")
    finally:
        client.close()
    assert (code, answer) == (0, "B")
    assert record["stop_reason"] == "direct_answer"
    assert len(requests) == 1 and requests[0].url.path == "/v1/chat/completions"
    body = json.loads(requests[0].content)
    assert body["max_tokens"] == 4096
    assert body["chat_template_kwargs"]["enable_thinking"] is False
    assert body["repetition_penalty"] == 1.1
    assert "tools" not in body and "logprobs" not in body


def test_direct_launcher_skips_sandbox_and_draft():
    script = r'''
source "$1"
parse_args --direct --bench hrbench4k --api-nproc 4
resolve_mode
[[ "$MODEL" == Qwen3.5-27B-vLLM && "$START_DRAFT" == false ]]
[[ "$MAIN_MAX_MODEL_LEN" == 160000 && "$API_NPROC" == 4 ]]
[[ -z "$(draft_base_urls)" ]]
start_sandbox() { echo UNEXPECTED_SANDBOX; return 1; }
print_gpu_snapshot() { :; }
start_vllm() { echo "VLLM=$1 TP=$5"; }
start_services
probe_http() { return 1; }
choose_port() { [[ "$1" == main ]] || return 1; printf '%s' "$2"; }
resolve_ports
'''
    result = subprocess.run(["bash", "-c", script, "_", str(LAUNCHER)],
                            env={**os.environ, "SPECTOOL_START_TEST_SOURCE_ONLY": "1"},
                            text=True, capture_output=True)
    assert result.returncode == 0, result.stderr
    assert "VLLM=main TP=2" in result.stdout
    assert "UNEXPECTED" not in result.stdout
