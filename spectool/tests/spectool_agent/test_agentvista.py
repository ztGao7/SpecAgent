"""AgentVista profile: tools (Serper search/scrape, official image search, sandbox aliases),
prompts, fallback fork, TSV builder and the official-prompt judge.  Network is mocked.

Needs the official AgentVista checkout: AGENTVISTA_REPO (skipped otherwise).
"""
import json
import os
import subprocess
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from PIL import Image

REPO = os.environ.get("AGENTVISTA_REPO", "")
needs_repo = pytest.mark.skipif(not os.path.isfile(os.path.join(REPO, "tools", "web_search.py")),
                                reason="set AGENTVISTA_REPO to the official AgentVista checkout")


class _Resp:
    def __init__(self, payload=None, status=200, text=None):
        self._p, self.status_code = payload, status
        self.text = text if text is not None else json.dumps(payload)

    def json(self):
        return self._p

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(self.status_code)


def _names(schemas):
    return [s["function"]["name"] for s in schemas]


def test_profile_and_prompts():
    from spectool_agent.benchmarks import AGENTVISTA_PROFILE, DEFAULT_PROFILE, resolve_benchmark_profile
    from spectool_agent.prompts import get_prompt_set
    from spectool_agent.tools import get_tool_schemas
    assert resolve_benchmark_profile(dataset="AgentVista") == AGENTVISTA_PROFILE
    assert resolve_benchmark_profile(dataset="VStarBench") == DEFAULT_PROFILE
    ps = get_prompt_set(dataset="AgentVista")
    assert ps.profile == AGENTVISTA_PROFILE and "web_search" in ps.system_prompt
    assert ps.draft_system_prompt.startswith(ps.system_prompt) and not ps.normalize_mcq_answers
    assert _names(get_tool_schemas(dataset="AgentVista")) == ["web_search", "image_search", "visit", "code_interpreter"]
    assert _names(get_tool_schemas(dataset="VStarBench")) == ["sandbox"]


@pytest.fixture
def registry(monkeypatch, tmp_path):
    from spectool_agent.tools import ToolRegistry, register_default_tools
    monkeypatch.setenv("SERPAPI_KEY", "test-key")
    monkeypatch.setenv("SPECTOOL_VLLM_BASE", "http://main:1/v1")
    monkeypatch.setenv("SPECTOOL_VLLM_MODEL", "main-27b")
    reg = ToolRegistry()
    schemas = register_default_tools(reg, dataset="AgentVista")
    reg.set_input_images([Image.new("RGB", (8, 8), "red"), Image.new("RGB", (8, 8), "blue")])
    yield reg, schemas
    reg.cleanup()


@needs_repo
def test_web_search_uses_serper(registry, monkeypatch):
    import requests
    reg, _ = registry
    calls = []

    def post(url, *a, **kw):
        calls.append((url, kw))
        return _Resp({"organic": [{"title": "T", "link": "https://e.x", "snippet": "S"}]})
    monkeypatch.setattr(requests, "post", post)
    from spectool_agent.types import ToolCall
    obs, imgs = reg.dispatch(ToolCall("web_search", {"query": "eiffel"}), None)
    assert calls[0][0] == "https://google.serper.dev/search" and calls[0][1]["headers"]["X-API-KEY"] == "test-key"
    assert "[T](https://e.x)" in obs and imgs == []


@needs_repo
def test_visit_serper_scrape_then_official_summary_without_thinking(registry, monkeypatch):
    import spectool_agent.agentvista_tools as av
    reg, _ = registry
    calls = []

    def post(url, *a, **kw):
        calls.append((url, kw.get("json")))
        if url == "https://scrape.serper.dev":
            return _Resp({"markdown": "# Page\nThe tower is 330 m tall."})
        return _Resp({"choices": [{"message": {"content": '{"evidence": "330 m", "summary": "330 m tall"}'}}]})
    monkeypatch.setattr(av, "_ORIGINAL_POST", post)
    from spectool_agent.types import ToolCall
    obs, _ = reg.dispatch(ToolCall("visit", {"url": "https://e.x/a", "goal": "height"}), None)
    assert calls[0][0] == "https://scrape.serper.dev" and calls[0][1]["url"] == "https://e.x/a"
    summary_url, payload = calls[1]
    assert summary_url == "http://main:1/v1/chat/completions" and payload["model"] == "main-27b"
    assert payload["chat_template_kwargs"] == {"enable_thinking": False} and payload["temperature"] == 0.3
    assert "330 m" in obs and "https://e.x/a" in obs


@needs_repo
def test_code_interpreter_aliases_and_tool_images(registry, monkeypatch):
    from spectool_agent.tools import ToolRegistry
    from spectool_agent.types import ToolCall
    reg, _ = registry
    seen = {}

    def fake_sandbox(self, image, code):
        seen["code"] = code
        return "done", [Image.new("RGB", (4, 4), "green")]
    monkeypatch.setattr(ToolRegistry, "_sandbox", fake_sandbox)
    obs, imgs = reg.dispatch(ToolCall("code_interpreter", {"code": "print(1)"}), None)
    assert seen["code"].splitlines()[:2] == ["original_image = image_1", "original_image_1 = image_2"]
    assert "tool_image_1" in obs and len(imgs) == 1 and os.path.isfile(reg._av_tool_images[0])
    assert reg.is_stateful_tool("code_interpreter") and not reg.is_stateful_tool("web_search")
    assert reg.validate_call(ToolCall("code_interpreter", {"code": " "})) is not None


@needs_repo
def test_image_refs_resolve_to_local_files(registry):
    import spectool_agent.agentvista_tools as av
    reg, _ = registry
    reg._av_tool_images = ["/tmp/t1.png"]
    assert av._resolve_image_ref(reg, "original_image") == reg._input_image_paths[0]
    assert av._resolve_image_ref(reg, "original_image_1") == reg._input_image_paths[1]
    assert av._resolve_image_ref(reg, "tool_image_1.png") == "/tmp/t1.png"
    assert av._resolve_image_ref(reg, "https://x/y.jpg") == "https://x/y.jpg"
    with pytest.raises(ValueError):
        av._resolve_image_ref(reg, "tool_image_9")


@needs_repo
def test_fork_rebinds_agentvista_tools_to_child(registry):
    reg, _ = registry
    child = reg.fork_clean_sandbox([Image.new("RGB", (8, 8))])
    try:
        assert child._tools["code_interpreter"].__self__ is child
        assert child._av_tools is reg._av_tools and child._av_tool_images == []
    finally:
        child.cleanup()


def test_tsv_builder_and_judge(tmp_path, monkeypatch):
    src = tmp_path / "prepared"; (src / "images").mkdir(parents=True)
    for name in ("agentvista_000_01.png", "agentvista_001_01.png", "agentvista_001_02.png"):
        Image.new("RGB", (4, 4)).save(src / "images" / name)
    rows = [{"question_id": "agentvista_000", "problem": "<image>\nWhat city?", "images": ["agentvista_000_01.png"],
             "solution": "Paris", "domain": "geography", "subdomain": "g"},
            {"question_id": "agentvista_001", "problem": "<image><image>\nHow many?", "images":
             ["agentvista_001_01.png", "agentvista_001_02.png"], "solution": "3", "domain": "commerce", "subdomain": "c"}]
    (src / "official_eval.json").write_text(json.dumps(rows))
    lmu = tmp_path / "LMUData"
    script = Path(__file__).resolve().parents[2] / "scripts" / "build_agentvista_tsv.py"
    subprocess.run([sys.executable, str(script)], check=True, env={**os.environ, "SRC_DIR": str(src), "LMUData": str(lmu)})
    import pandas as pd
    tsv = pd.read_csv(lmu / "AgentVista.tsv", sep="\t")
    assert list(tsv["question"]) == ["What city?", "How many?"] and list(tsv["n_images"]) == [1, 2]
    assert json.loads(tsv["image_path"][1])[1].endswith("agentvista_001_02.png")

    vk = Path(__file__).resolve().parents[2] / "VLMEvalKit"
    sys.path.insert(0, str(vk))
    try:
        from vlmeval.dataset.agentvista import judge_one
    except Exception as exc:  # VLMEvalKit's heavy imports may be missing on a CPU box
        pytest.skip(f"vlmeval not importable here: {exc}")
    client = MagicMock()
    client.chat.completions.create.return_value.choices = [MagicMock(message=MagicMock(content="Score: 1\nExplanation: same"))]
    assert judge_one("What city?", "Paris", "paris", client, "deepseek")[0] == 1
    client.chat.completions.create.side_effect = RuntimeError("down")
    monkeypatch.setattr("time.sleep", lambda s: None)
    assert judge_one("q", "a", "b", client, "deepseek")[0] is None
