"""Regression tests for the single-entry QJ/answerability context."""

from inspect import signature
from unittest.mock import MagicMock

from PIL import Image

from spectool_agent.speculative_agent import SpeculativeAgentLoop
from spectool_agent.types import AgentConfig, PoolEntry, ToolCall


def _loop(monkeypatch):
    monkeypatch.setenv("SPECTOOL_FAST_ANSWER_ENABLED", "true")
    monkeypatch.setenv("SPECTOOL_QJ_MAX_IMAGES", "4")
    loop = SpeculativeAgentLoop(
        MagicMock(), MagicMock(), MagicMock(),
        AgentConfig(api_base="http://fake", model_name="fake",
                    quality_judge_enabled=True, quality_judge_threshold=.75,
                    answerability_threshold=.9))
    loop.main_model.assess_draft_step.return_value = {
        "score": .99, "label": "GOOD", "quality_valid": True,
        "answer": "A", "answerable": True, "answerability_score": .99,
        "raw": "GOOD|<answer>A</answer><verified>1</verified>"}
    return loop


def _entry(index, image_count=1):
    return PoolEntry(
        step_idx=index, tool_call=ToolCall("sandbox", {"code": "print(1)"}),
        tool_observation=f"clue {index}",
        tool_result_images=[Image.new("RGB", (2, 2), (index, item, 0))
                            for item in range(image_count)])


def _text(messages):
    return "\n".join(
        part["text"]
        for message in messages if isinstance(message["content"], list)
        for part in message["content"] if part["type"] == "text")


def test_assessment_accepts_only_the_current_entry(monkeypatch):
    loop = _loop(monkeypatch)
    try:
        current = _entry(1, image_count=5)
        quality, answer = loop._assess_tool_entry(
            Image.new("RGB", (2, 2)), "Question\nA. one\nB. two", current)
        messages = loop.main_model.assess_draft_step.call_args.args[0]
        prompt = _text(messages)

        # One original image leaves capacity for exactly three tool images.
        image_count = sum(
            part["type"] == "image_url"
            for message in messages if isinstance(message["content"], list)
            for part in message["content"])
        assert image_count == 4
        assert "clue 1" in prompt
        assert "PRIOR_QJ" not in prompt
        assert "INCOMPLETE EVIDENCE" not in prompt
        assert quality["passed"] is True
        assert answer["answer"] == "A"
    finally:
        loop.shutdown()


def test_assessment_api_cannot_receive_history_entries():
    assert "evidence_entries" not in signature(
        SpeculativeAgentLoop._assess_tool_entry).parameters
