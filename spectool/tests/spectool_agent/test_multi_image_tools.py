from unittest.mock import MagicMock

from PIL import Image

from spectool_agent.pool import DraftPool
from spectool_agent.sandbox import SandboxResult, _shared_init_code
from spectool_agent.speculative_agent import (
    SpeculativeAgentLoop, _truncate_image_history)
from spectool_agent.tools import ToolRegistry
from spectool_agent.types import AgentConfig, PoolEntry, ToolCall


def test_tool_registry_passes_all_input_images_to_sandbox(monkeypatch):
    captured = {}

    def fake_execute(**kwargs):
        captured.update(kwargs)
        return SandboxResult(success=True, output="ok", execution_context={})

    monkeypatch.setattr("spectool_agent.tools._sandbox.execute", fake_execute)
    registry = ToolRegistry()
    registry.register("sandbox", registry._sandbox)
    registry.set_input_images([
        Image.new("RGB", (8, 8), (255, 0, 0)),
        Image.new("RGB", (6, 10), (0, 0, 255)),
    ])
    try:
        observation, result_images = registry.dispatch(
            ToolCall("sandbox", {"code": "print(image_2.size)"}),
            Image.new("RGB", (8, 8)),
        )
        assert observation == "ok"
        assert result_images == []
        assert len(captured["input_image_paths"]) == 2
        assert captured["input_image_path"] == captured["input_image_paths"][0]
    finally:
        registry.cleanup()


def test_shared_sandbox_bootstrap_exposes_all_images(tmp_path):
    first = tmp_path / "first.png"
    second = tmp_path / "second.png"
    Image.new("RGB", (8, 8), (255, 0, 0)).save(first)
    Image.new("RGB", (6, 10), (0, 0, 255)).save(second)
    namespace = {}

    exec(_shared_init_code(
        [str(first), str(second)], str(tmp_path), [str(first), str(second)]
    ), namespace)

    assert namespace["image_1"].size == (8, 8)
    assert namespace["image_2"].size == (6, 10)
    assert namespace["image_paths"] == [str(first), str(second)]


def test_speculative_history_preserves_all_original_images():
    original_parts = [
        {"type": "image_url", "image_url": {"url": "original-1"}},
        {"type": "image_url", "image_url": {"url": "original-2"}},
    ]
    messages = [
        {"role": "system", "content": "system"},
        {"role": "user", "content": original_parts},
        {"role": "tool", "content": [
            {"type": "image_url", "image_url": {"url": "tool-1"}},
            {"type": "image_url", "image_url": {"url": "tool-2"}},
        ]},
    ]

    _truncate_image_history(messages, max_images=3)

    assert messages[1]["content"] == original_parts
    retained_tool_images = [
        part for part in messages[2]["content"]
        if part.get("type") == "image_url"
    ]
    assert len(retained_tool_images) == 1


def test_speculative_main_contexts_receive_all_original_images():
    images = [
        Image.new("RGB", (8, 8), (255, 0, 0)),
        Image.new("RGB", (6, 10), (0, 0, 255)),
    ]
    loop = SpeculativeAgentLoop(
        MagicMock(), MagicMock(), ToolRegistry(),
        AgentConfig(api_base="http://fake", model_name="fake"),
    )
    entry = PoolEntry(step_idx=0, tool_observation="evidence")
    pool = DraftPool()
    pool.add(entry)

    def original_image_count(messages):
        return sum(
            1 for part in messages[1]["content"]
            if part.get("type") == "image_url"
        )

    try:
        shared = loop._shared_prompt_context(images, "Compare them")
        review = loop._build_main_context(
            images, "Compare them", pool, 0, shared_ctx=shared)
        fast = loop._build_fast_answer_context(
            images, "Compare them", shared_ctx=shared)
        assessment = loop._build_step_assessment_prefix(
            images, "Compare them", entry, shared_ctx=shared)

        assert original_image_count(review) == 2
        assert original_image_count(fast) == 2
        assert original_image_count(assessment) == 2
    finally:
        loop.shutdown()
