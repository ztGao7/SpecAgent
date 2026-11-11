import threading
import time
from unittest.mock import MagicMock

import pytest
from PIL import Image

from spectool_agent.model import RequestCancelled
from spectool_agent.pool import DraftPool
from spectool_agent.speculative_agent import SpeculativeAgentLoop
from spectool_agent.types import AgentConfig, PoolEntry, ToolCall, Trajectory


QUESTION = "Which object?\nA. first\nB. second"


def make_loop():
    return SpeculativeAgentLoop(
        MagicMock(), MagicMock(), MagicMock(),
        AgentConfig(api_base="http://fake", model_name="fake",
                    quality_judge_threshold=0.65,
                    answerability_threshold=0.65))


def result(score=0.9, answer="A", answerable=True):
    return {"answer":answer, "answerable":answerable,
            "answerability_score":score, "answerability_reason":"answer",
            "raw":f"NA|1|<answer>{answer}</answer>"}


@pytest.mark.parametrize("score,accepted", [
    (0.649, False), (0.65, True), (0.9, True), (None, False),
    (float("nan"), False), (float("inf"), False), (1.1, False),
])
def test_fast_answer_keeps_exact_threshold_and_rejects_invalid_scores(score, accepted):
    loop = make_loop()
    try:
        traj = Trajectory()
        assert loop._apply_fast_answer_result(
            result(score), QUESTION, DraftPool(), traj) is accepted
        assert (traj.final_answer == "A") is accepted
        assert loop.config.answerability_threshold == 0.65
        assert loop.config.quality_judge_threshold == 0.65
    finally:
        loop.shutdown()


@pytest.mark.parametrize("reply", [result(answer="Z"), result(answer=""),
                                  result(answer=None), result(answerable=False)])
def test_invalid_answer_or_abstention_does_not_bypass_gate(reply):
    loop = make_loop()
    try:
        assert not loop._apply_fast_answer_result(
            reply, QUESTION, DraftPool(), Trajectory())
    finally:
        loop.shutdown()


def test_known_draft_conflict_is_not_overwritten_by_fast_answer():
    loop = make_loop()
    pool = DraftPool()
    pool.add(PoolEntry(step_idx=0, answer="B"))
    traj = Trajectory()
    try:
        assert not loop._apply_fast_answer_result(result(), QUESTION, pool, traj)
        assert traj.final_answer is None
        assert traj.steps[-1].answer == "initial_answerability_draft_answer_conflict"
        traj.final_answer = "B"
        assert not loop._apply_fast_answer_result(result(), QUESTION, pool, traj)
        assert traj.final_answer == "B"
    finally:
        loop.shutdown()


def test_open_answer_keeps_format_and_original_images():
    loop = make_loop()
    try:
        images = [Image.new("RGB", (8, 8)), Image.new("RGB", (16, 16))]
        shared = loop._shared_prompt_context(images, "What is written?")
        messages = loop._build_fast_answer_context(
            images, "What is written?", shared_ctx=shared)
        assert messages[1]["content"][:-1] == shared["image_content"]
        text = messages[-1]["content"][0]["text"]
        assert "YOUR COMPLETE ANSWER" in text
        assert "NA|<verified>0</verified>" in text
        assert "unreadable or occluded" in text
        assert loop._apply_fast_answer_result(
            result(answer="a red sign"), "What is written?", DraftPool(), Trajectory())
    finally:
        loop.shutdown()


@pytest.mark.parametrize("question", [QUESTION, "What is written on the sign?"])
def test_tool_evidence_gate_applies_to_mcq_and_open_answers(question):
    loop = make_loop()
    try:
        messages = loop._build_fast_answer_context(
            Image.new("RGB", (8, 8)), question)
        system = messages[2]["content"][0]["text"]
        instruction = messages[-1]["content"][0]["text"]
        assert "without using tools" in system
        assert "need tools to collect or verify further evidence, abstain" in system
        assert "If you need tools to collect or verify further evidence, abstain" in instruction
        assert "Set verified to 1 only when" in instruction
        assert "use 0 for an unsupported candidate or uncertainty" in instruction
        assert "NA|<verified>0</verified>" in instruction
        assert "Never put the verification decision before the candidate" in instruction
        assert "by itself require abstention" not in instruction
    finally:
        loop.shutdown()


@pytest.mark.parametrize("draft_answer", [None, "A", "B"])
def test_draft_finishing_first_does_not_lose_fast_result(monkeypatch, draft_answer):
    monkeypatch.setenv("SPECTOOL_FAST_ANSWER_ENABLED", "1")
    loop = make_loop()
    fast_started = threading.Event()
    draft_done = threading.Event()

    def draft(messages, image, pool, cancel, traj):
        assert fast_started.wait(1)
        if draft_answer:
            pool.add(PoolEntry(step_idx=0, answer=draft_answer))
        draft_done.set()

    def fast(*args):
        fast_started.set()
        assert draft_done.wait(1)
        time.sleep(0.03)
        return result()

    loop._draft_loop = draft
    loop._run_fast_answer_gate = fast
    loop._run_main_review = MagicMock(return_value={
        "candidate_answer":"B", "supported":True, "score":0.9, "raw":"GOOD"})
    try:
        traj = loop.run(Image.new("RGB", (8, 8)), QUESTION)
        if draft_answer == "B":
            assert traj.final_answer == "B"
            loop._run_main_review.assert_called_once()
        else:
            assert traj.final_answer == "A"
            assert traj.stop_reason == "fast_main_answer"
            loop._run_main_review.assert_not_called()
        assert len([s for s in traj.steps if s.verdict_mode in (
            "fast_main_answer", "initial_answerability")]) == 1
    finally:
        loop.shutdown()


def test_tool_handoff_rejects_fast_result_that_did_not_see_new_evidence(monkeypatch):
    monkeypatch.setenv("SPECTOOL_FAST_ANSWER_ENABLED", "1")
    loop = make_loop()
    tool_ready = threading.Event()

    def draft(messages, image, pool, cancel, traj):
        pool.add(PoolEntry(step_idx=0, tool_call=ToolCall("sandbox", {}),
                           tool_observation="visible detail"))
        tool_ready.set()
        deadline = time.monotonic() + 2
        while not cancel.cancelled and time.monotonic() < deadline:
            time.sleep(0.005)

    def fast(*args):
        assert tool_ready.wait(1)
        time.sleep(0.05)
        return result()

    loop._draft_loop = draft
    loop._run_fast_answer_gate = fast
    loop._assess_tool_entry = MagicMock(return_value=(
        {"passed": True, "result": {"score": .9, "raw": "GOOD"}, "elapsed": 0},
        {"answer": "A", "score": .9, "status": "answer", "elapsed": 0}))
    try:
        traj = loop.run(Image.new("RGB", (8, 8)), QUESTION)
        assert traj.stop_reason == "main_answer_from_draft_evidence"
        loop._assess_tool_entry.assert_called_once()
        assert not any(s.verdict_mode == "fast_main_answer" for s in traj.steps)
        assert any(s.answer == "initial_answerability_stale_evidence" for s in traj.steps)
    finally:
        loop.shutdown()


@pytest.mark.parametrize("late_reply", [False, True])
def test_handoff_is_bounded_and_late_reply_cannot_overwrite_review(monkeypatch, late_reply):
    monkeypatch.setenv("SPECTOOL_FAST_ANSWER_ENABLED", "1")
    monkeypatch.setattr("spectool_agent.speculative_agent._FAST_ANSWER_HANDOFF_GRACE_S", 0.05)
    loop = make_loop()
    started, cancelled, finished = threading.Event(), threading.Event(), threading.Event()

    def draft(messages, image, pool, cancel, traj):
        assert started.wait(1)
        pool.add(PoolEntry(step_idx=0, answer="B"))

    def fast(image, question, traj, token):
        started.set()
        deadline = time.monotonic() + 2
        while not token.cancelled and time.monotonic() < deadline:
            time.sleep(0.002)
        assert token.cancelled
        cancelled.set()
        finished.set()
        if late_reply:
            return result()
        raise RequestCancelled("cancelled on handoff")

    loop._draft_loop = draft
    loop._run_fast_answer_gate = fast
    loop._run_main_review = MagicMock(return_value={
        "candidate_answer":"B", "supported":True, "score":0.9, "raw":"GOOD"})
    begin = time.monotonic()
    try:
        traj = loop.run(Image.new("RGB", (8, 8)), QUESTION)
        assert time.monotonic() - begin < 1
        assert cancelled.wait(0.5)
        assert finished.wait(0.5)
        assert traj.final_answer == "B"
        assert traj.stop_reason == "main_answer_draft_evidence_verified"
        assert not any(s.verdict_mode == "fast_main_answer" for s in traj.steps)
        loop._run_main_review.assert_called_once()
    finally:
        loop.shutdown()
