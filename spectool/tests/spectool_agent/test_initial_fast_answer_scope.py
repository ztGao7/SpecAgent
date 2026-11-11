"""The initial shortcut stays narrow without changing later answer paths."""

import threading
from unittest.mock import MagicMock

import pytest
from PIL import Image

from spectool_agent.prompt import (
    FAST_ANSWER_SYSTEM_PROMPT,
    answer_verification_instruction,
    fast_answer_instruction,
)
from spectool_agent.prompts import get_prompt_set
from spectool_agent.speculative_agent import SpeculativeAgentLoop
from spectool_agent.types import AgentConfig, PoolEntry


@pytest.mark.parametrize("choices", [None, ["A", "B"]])
def test_initial_scope_preserves_images_question_and_protocol(choices):
    loop = SpeculativeAgentLoop.__new__(SpeculativeAgentLoop)
    loop.prompt_set = get_prompt_set()
    question = "What color is the sign?"
    if choices:
        question += "\nA. red\nB. blue"
    images = [Image.new("RGB", (8, 8)), Image.new("RGB", (16, 16))]
    shared = loop._shared_prompt_context(images, question)
    messages = loop._build_fast_answer_context(images, question, shared_ctx=shared)

    # The opening is the shared fallback prefix; the fast role follows it.
    assert messages[0]["content"] == loop.prompt_set.system_prompt
    assert messages[1]["content"][:-1] == shared["image_content"]
    assert messages[1]["content"][-1]["text"] == shared["question_text"]
    role = messages[2]["content"][0]["text"]
    assert FAST_ANSWER_SYSTEM_PROMPT in role
    assert "without a candidate answer" in role
    assert messages[3]["content"][0]["text"] == fast_answer_instruction(choices)


@pytest.mark.parametrize("choices", [None, ["A", "B"], ["A", "B", "C", "D"]])
def test_initial_instruction_preserves_historical_output_example_order(choices):
    original = answer_verification_instruction(choices)
    initial = fast_answer_instruction(choices)
    answer = (
        "X, where X is one of " + ", ".join(choices)
        if choices else "YOUR COMPLETE ANSWER"
    )
    abstain = "NA|<verified>0</verified>"
    supported = (
        "NA|<evidence>brief observable evidence</evidence>"
        f"<answer>{answer}</answer><verified>1</verified>"
    )
    assert abstain + "\n" + supported in original
    assert initial == original


def test_initial_system_restores_the_balanced_scope():
    assert "A short visual scan" in FAST_ANSWER_SYSTEM_PROMPT
    assert "a short chain of reasoning" in FAST_ANSWER_SYSTEM_PROMPT
    assert "prolonged or exhaustive search" in FAST_ANSWER_SYSTEM_PROMPT
    assert "If you need tools to collect or verify further evidence, abstain." in FAST_ANSWER_SYSTEM_PROMPT


def test_quality_instruction_keeps_its_original_example_order():
    instruction = answer_verification_instruction(["A", "B"], quality_enabled=True)
    assert "GOOD|<verified>0</verified>\nGOOD|<evidence>" in instruction


def test_initial_abstention_leaves_draft_and_verification_running(monkeypatch):
    monkeypatch.setenv("SPECTOOL_FAST_ANSWER_ENABLED", "1")
    loop = SpeculativeAgentLoop(
        MagicMock(), MagicMock(), MagicMock(),
        AgentConfig(api_base="http://fake", model_name="fake",
                    quality_judge_threshold=0.7, answerability_threshold=0.8))
    fast_started = threading.Event()

    def draft(messages, image, pool, cancel, traj):
        assert fast_started.wait(2)
        pool.add(PoolEntry(step_idx=0, answer="B"))

    def fast(*args):
        fast_started.set()
        return {"answer": None, "answerable": False,
                "answerability_score": None,
                "answerability_reason": "insufficient",
                "raw": "NA|<verified>0</verified>"}

    loop._draft_loop = draft
    loop._run_fast_answer_gate = fast
    loop._run_main_review = MagicMock(return_value={
        "candidate_answer": "B", "supported": True, "score": 0.95, "raw": "GOOD"})
    loop._run_fallback_once = MagicMock(side_effect=AssertionError("unexpected fallback"))
    try:
        traj = loop.run(Image.new("RGB", (8, 8)), "How many?\nA. three\nB. four")
        assert traj.final_answer == "B"
        assert traj.stop_reason != "fast_main_answer"
        assert any(step.verdict_mode == "initial_answerability" for step in traj.steps)
        loop._run_main_review.assert_called_once()
        loop._run_fallback_once.assert_not_called()
        assert loop.config.quality_judge_threshold == 0.7
        assert loop.config.answerability_threshold == 0.8
    finally:
        loop.shutdown()
