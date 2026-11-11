import threading
from concurrent.futures import Future
from unittest.mock import MagicMock

import pytest
from PIL import Image

from spectool_agent.pool import DraftPool
from spectool_agent.speculative_agent import (
    CancellableTask,
    FallbackCoordinator,
    SpeculativeAgentLoop,
)
from spectool_agent.types import AgentConfig, PoolEntry, ToolCall, Trajectory


def _loop():
    return SpeculativeAgentLoop(
        MagicMock(), MagicMock(), MagicMock(),
        AgentConfig(
            api_base="http://fake",
            model_name="fake",
            quality_judge_enabled=True,
        ),
    )


def test_qj_request_error_is_unknown_and_is_recorded():
    loop = _loop()
    entry = PoolEntry(
        step_idx=3,
        tool_call=ToolCall("sandbox", {"code": "print(1)"}),
        tool_observation="useful evidence",
    )
    loop.main_model.assess_draft_step.side_effect = RuntimeError("upstream down")
    traj = Trajectory()

    try:
        quality, answer = loop._assess_tool_entry(
            Image.new("RGB", (2, 2)), "Question", entry, traj=traj)
        loop._record_quality_judge(traj, quality)
    finally:
        loop.shutdown()

    assert quality["passed"] is None
    assert quality["result"] is None
    assert quality["status"] == "error"
    assert "RuntimeError: upstream down" in quality["error"]
    assert answer["status"] == "error"
    assert traj.steps[-1].verdict_mode == "quality_judge_error"


def test_qj_error_stays_unverified_and_does_not_reset_bad_streak():
    loop = _loop()
    pool = DraftPool()
    for step_idx in range(3):
        pool.add(PoolEntry(
            step_idx=step_idx,
            tool_call=ToolCall("sandbox", {"code": str(step_idx)}),
            tool_observation="evidence",
        ))

    traj = Trajectory()
    cancel = CancellableTask()
    draft_future = Future()
    fallback_started = threading.Event()
    outcomes = iter([False, None, False])

    def fake_assessment(*args, **kwargs):
        step_idx = kwargs["step_idx"]
        passed = next(outcomes)
        return (
            {
                "passed": passed,
                "result": None,
                "status": "error" if passed is None else "ok",
                "error": "RuntimeError: upstream down" if passed is None else "",
                "elapsed": 0.0,
                "step_idx": step_idx,
            },
            {
                "answer": None,
                "raw": "",
                "status": "error" if passed is None else "insufficient",
                "elapsed": 0.0,
                "step_idx": step_idx,
            },
        )

    loop._assess_tool_entry = fake_assessment

    def fake_fallback(*args, **kwargs):
        fallback_started.set()
        traj.final_answer = "A"
        return True

    loop._run_fallback_once = fake_fallback
    worker = threading.Thread(
        target=loop._review_loop,
        args=(
            Image.new("RGB", (1, 1)),
            "Choose A or B.",
            pool,
            traj,
            cancel,
            draft_future,
            FallbackCoordinator(),
        ),
    )
    worker.start()
    try:
        assert fallback_started.wait(timeout=2)
        worker.join(timeout=2)
        assert not worker.is_alive()
    finally:
        cancel.cancel()
        loop.shutdown()

    assert [entry.quality_passed for entry in pool.get_all()] == [
        False, None, False]
    assert traj.stop_reason == "draft_quality_bad_streak"
    assert draft_future.cancelled()
    assert any(step.verdict_mode == "quality_judge_error" for step in traj.steps)


def test_unknown_qj_entry_cannot_become_takeover_history():
    assistant_turn = {
        "role": "assistant",
        "content": "",
        "tool_calls": [{
            "id": "call-0",
            "type": "function",
            "function": {"name": "sandbox", "arguments": "{}"},
        }],
    }
    entries = [
        PoolEntry(
            step_idx=0,
            tool_call=ToolCall("sandbox", {"code": "0"}),
            tool_call_id="call-0",
            assistant_message=assistant_turn,
            tool_observation="validated evidence",
            quality_passed=True,
        ),
        PoolEntry(
            step_idx=1,
            tool_call=ToolCall("sandbox", {"code": "1"}),
            tool_call_id="call-1",
            assistant_message={**assistant_turn, "tool_calls": [{
                **assistant_turn["tool_calls"][0], "id": "call-1",
            }]},
            tool_observation="unavailable QJ verdict",
            quality_passed=None,
        ),
    ]

    groups = SpeculativeAgentLoop._validated_takeover_groups(entries)

    assert [step_idx for step_idx, _ in groups] == [0]


@pytest.mark.parametrize("observations,quality_results,expected_trigger", [
    (["error", "error", "ok"], [True], None),
    (["error", "error", "error"], [], "tool_error_streak"),
    (["error", "error", "ok", "error", "error", "ok"], [True, True], None),
    (["ok", "error", "ok"], [False, False], "quality_bad_streak"),
])
def test_execution_errors_are_separate_from_qj_bad_streak(
        observations, quality_results, expected_trigger):
    loop = _loop()
    pool = DraftPool()
    for idx, kind in enumerate(observations):
        observation = (
            "stderr:\nAttributeError                            "
            "Traceback (most recent call last)\n"
            "AttributeError: 'PngImageFile' object has no attribute 'read'"
            if kind == "error" else "actual evidence")
        pool.add(PoolEntry(
            step_idx=idx, tool_call=ToolCall("sandbox", {"code": str(idx)}),
            tool_observation=observation))
    traj = Trajectory()
    cancel = CancellableTask()
    draft_future = Future()
    assessed = []
    fallbacks = []

    def assess(image, user_text, entry, **kwargs):
        passed = quality_results[len(assessed)]
        assessed.append(entry.step_idx)
        answer = "B" if passed and len(assessed) == len(quality_results) else None
        return (
            {"passed": passed, "result": None, "elapsed": 0.0,
             "step_idx": entry.step_idx},
            {"answer": answer, "status": "answer" if answer else "insufficient",
             "elapsed": 0.0, "step_idx": entry.step_idx},
        )

    def fallback(*args, **kwargs):
        fallbacks.append(kwargs["trigger"])
        traj.final_answer = "A"
        return True

    loop._assess_tool_entry = assess
    loop._run_fallback_once = fallback
    try:
        loop._review_loop(
            Image.new("RGB", (2, 2)), "Question", pool, traj, cancel,
            draft_future, FallbackCoordinator())
    finally:
        cancel.cancel()
        loop.shutdown()

    assert len(assessed) == len(quality_results)
    assert all(observations[idx] == "ok" for idx in assessed)
    assert fallbacks == ([expected_trigger] if expected_trigger else [])
    if expected_trigger is None:
        assert traj.final_answer == "B"
        assert traj.stop_reason == "main_answer_from_draft_evidence"
    for entry, kind in zip(pool.get_all(), observations):
        if kind == "error":
            assert entry.quality_passed is None
