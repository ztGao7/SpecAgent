"""Tests for SpeculativeAgentLoop using mocked draft model.

The agent no longer runs main model inference. It monitors the draft
pool and accepts the first answer the draft produces.
"""
from unittest.mock import MagicMock

from PIL import Image

from spectool_agent.speculative_agent import (
    SpeculativeAgentLoop, _fast_answer_handoff_grace_seconds,
)
from spectool_agent.tools import ToolRegistry, register_default_tools
from spectool_agent.types import AgentConfig, ToolCall


def _make_draft_loop(
    draft_responses: list[str],
    max_steps: int = 5,
) -> SpeculativeAgentLoop:
    cfg = AgentConfig(api_base="http://fake", model_name="m",
                      max_steps=max_steps)

    main_model = MagicMock()  # kept for backward-compat init, not used
    draft_model = MagicMock()
    draft_model.generate_full.side_effect = draft_responses

    tools = ToolRegistry()
    register_default_tools(tools)

    return SpeculativeAgentLoop(main_model, draft_model, tools, cfg)


def test_draft_answers_directly():
    """Draft emits <answer> on first call — agent accepts it."""
    loop = _make_draft_loop(["<answer>B</answer>"])
    img = Image.new("RGB", (40, 40), (0, 0, 0))
    traj = loop.run(img, "What letter?")

    assert traj.final_answer == "B"
    assert traj.stop_reason == "draft_answer"
    assert traj.total_time >= 0.0

    loop.shutdown()


def test_draft_tool_then_answer():
    """Draft does a tool call then answers on the next step."""
    loop = _make_draft_loop([
        '<tool_call>{"name": "sandbox", "arguments": {"code": "print(1)"}}</tool_call>',
        '<answer>red</answer>',
    ])
    img = Image.new("RGB", (40, 40), (255, 0, 0))
    traj = loop.run(img, "What color?")

    assert traj.final_answer == "red"
    assert traj.stop_reason == "draft_answer"

    loop.shutdown()


def test_draft_fallback_answer():
    """Draft emits a bare MCQ letter without <answer> tags."""
    loop = _make_draft_loop(["A"])
    img = Image.new("RGB", (10, 10), (0, 0, 0))
    traj = loop.run(img, "What letter?")

    assert traj.final_answer == "A"
    assert traj.stop_reason == "draft_answer"

    loop.shutdown()


def test_draft_no_answer():
    """Draft runs out of steps with no answer — agent returns None."""
    loop = _make_draft_loop([
        '<tool_call>{"name": "sandbox", "arguments": {"code": "print(1)"}}</tool_call>',
    ] * 5, max_steps=5)
    img = Image.new("RGB", (10, 10), (0, 0, 0))
    traj = loop.run(img, "q")

    assert traj.final_answer is None
    assert traj.stop_reason == "no_draft_answer"

    loop.shutdown()


def test_timing_populated():
    """Timing fields are populated even with mocked draft."""
    loop = _make_draft_loop(["<answer>Z</answer>"])
    traj = loop.run(Image.new("RGB", (10, 10)), "q")

    assert traj.total_time >= 0.0
    assert traj.total_inference_time == 0.0  # no main model inference

    loop.shutdown()


def test_cancellation_stops_draft():
    """Agent cancels draft once answer is accepted."""
    loop = _make_draft_loop(["<answer>done</answer>"])
    traj = loop.run(Image.new("RGB", (10, 10)), "q")

    assert traj.final_answer == "done"
    # Draft should have been cancelled after answer
    loop.shutdown()


def test_fast_answer_handoff_grace_is_bounded(monkeypatch):
    monkeypatch.setenv("SPECTOOL_FAST_ANSWER_HANDOFF_GRACE_SECONDS", "3.5")
    assert _fast_answer_handoff_grace_seconds() == 3.5

    monkeypatch.setenv("SPECTOOL_FAST_ANSWER_HANDOFF_GRACE_SECONDS", "99")
    assert _fast_answer_handoff_grace_seconds() == 10.0

    monkeypatch.setenv("SPECTOOL_FAST_ANSWER_HANDOFF_GRACE_SECONDS", "bad")
    assert _fast_answer_handoff_grace_seconds() == 1.0
