import threading
from concurrent.futures import Future, ThreadPoolExecutor
from unittest.mock import MagicMock

import pytest
from PIL import Image

from spectool_agent.pool import DraftPool
from spectool_agent.speculative_agent import CancellableTask, FallbackCoordinator, SpeculativeAgentLoop
from spectool_agent.types import AgentConfig, PoolEntry, ToolCall, Trajectory


def make_loop(**kwargs):
    main = MagicMock()
    main.last_tool_calls = []
    main.last_tool_call_ids = []
    main.last_assistant_msg = {"role": "assistant", "content": ""}
    main.generate.return_value = "<answer>B</answer>"
    main.judge_good_bad.return_value = {"score": .99, "raw": "GOOD"}
    return SpeculativeAgentLoop(main, MagicMock(), MagicMock(), AgentConfig(
        api_base="http://fake", model_name="fake", max_steps=6, **kwargs))


def tool_entry(step=0, **kwargs):
    return PoolEntry(step_idx=step, tool_call=ToolCall("sandbox", {
        "code": "display(image.crop((1, 2, 30, 40)))"}),
        tool_observation="actual crop output", **kwargs)


def test_handoff_preserves_actual_evidence_but_not_draft_state_or_answer():
    loop = make_loop()
    pool = DraftPool()
    pool.add(tool_entry(tool_result_images=[Image.new("RGB", (2, 2))],
                        reasoning="SECRET_REASONING", response_text="SECRET_RESPONSE",
                        assistant_message={"role": "assistant", "content": "SECRET_STATE"}))
    pool.add(PoolEntry(step_idx=1, answer="A", reasoning="SECRET_CANDIDATE"))
    traj = Trajectory()
    try:
        loop._run_fallback_agent(Image.new("RGB", (4, 4)), "A or B?", pool, traj,
                                 CancellableTask())
        messages = loop.main_model.generate.call_args.args[0]
        text = repr(messages)
        assert "actual crop output" in text and "(1, 2, 30, 40)" in text
        assert "[EVIDENCE E1" in text and "fresh sandbox" in text
        assert all(word not in text for word in ["SECRET_REASONING", "SECRET_RESPONSE", "SECRET_STATE", "SECRET_CANDIDATE"])
        assert all(m["role"] in {"system", "user"} for m in messages)
        assert traj.fallback_mode == "main_evidence_handoff"
        assert traj.fallback_passive_entries == 1
        assert traj.fallback_validated_steps == 0
        assert traj.fallback_step_budget == 6
        loop.tools.reset_sandbox_context.assert_called_once()
        loop.tools.dispatch.assert_not_called()
    finally:
        loop.shutdown()


@pytest.mark.parametrize("trigger", ["baseline_direct", "quality_bad_streak", "tool_error_streak"])
def test_protected_triggers_keep_clean_restart(trigger):
    loop = make_loop()
    pool = DraftPool()
    pool.add(tool_entry(quality_passed=True))
    traj = Trajectory()
    try:
        state = loop._build_main_takeover_state(Image.new("RGB", (2, 2)), "Q", pool, traj,
            inject_draft=True, qj_failed_entry=None, trigger=trigger)
        assert len(state.messages) == 2
        assert traj.fallback_mode == "main_clean_restart"
        assert state.step_budget == 6
    finally:
        loop.shutdown()


def test_bad_error_and_candidate_only_entries_are_not_handoff_evidence():
    loop = make_loop()
    pool = DraftPool()
    pool.add(tool_entry(quality_passed=False))
    pool.add(PoolEntry(step_idx=1, tool_call=ToolCall("sandbox", {}), tool_observation="Error: failed"))
    pool.add(PoolEntry(step_idx=2, answer="A", reasoning="secret"))
    traj = Trajectory()
    try:
        state = loop._build_main_takeover_state(Image.new("RGB", (2, 2)), "Q", pool, traj,
            inject_draft=True, qj_failed_entry=None, trigger="speculative_exhausted")
        assert len(state.messages) == 2
        assert traj.fallback_mode == "main_clean_restart"
    finally:
        loop.shutdown()


def test_handoff_respects_existing_image_budget_and_marks_omissions():
    loop = make_loop(max_image_history=3)
    pool = DraftPool()
    for i in range(4):
        pool.add(tool_entry(i, tool_result_images=[Image.new("RGB", (2, 2), (i, 0, 0))]))
    traj = Trajectory()
    try:
        state = loop._build_main_takeover_state([Image.new("RGB", (2, 2))] * 2, "Q", pool, traj,
            inject_draft=True, qj_failed_entry=None, trigger="speculative_exhausted")
        images = [p for m in state.messages for p in m["content"] if isinstance(p, dict) and p.get("type") == "image_url"]
        assert len(images) == 3
        assert repr(state.messages).count("image(s) omitted by the image budget") == 3
        assert traj.fallback_passive_entries == 4
    finally:
        loop.shutdown()


@pytest.mark.parametrize("stale,answer", [(False, "A"), (False, None), (True, "A")])
def test_candidate_waits_for_main_then_reuses_or_reviews_once(stale, answer):
    loop = make_loop()
    pool = DraftPool()
    pool.add(tool_entry())
    draft = Future()
    draft.set_running_or_notify_cancel()
    started, release = threading.Event(), threading.Event()
    observed_cancel = []
    calls = []

    def assess(*args, **kwargs):
        calls.append(1)
        started.set()
        assert release.wait(2)
        observed_cancel.append(kwargs["request_cancel"].cancelled)
        return ({"passed": True, "result": None},
                {"answer": answer, "candidate_answer": answer, "status": "answer" if answer else "insufficient"})

    loop._assess_tool_entry = assess
    traj = Trajectory()
    with ThreadPoolExecutor(max_workers=1) as runner:
        job = runner.submit(loop._review_loop, Image.new("RGB", (2, 2)), "A or B?",
            pool, traj, CancellableTask(), draft, FallbackCoordinator())
        try:
            assert started.wait(2)
            if stale:
                pool.add(tool_entry(1))
            pool.add(PoolEntry(step_idx=2, answer="B"))
            draft.set_result(None)
            release.set()
            job.result(timeout=2)
            assert calls == [1] and observed_cancel == [False]
            if answer and not stale:
                assert traj.final_answer == "A"
                loop.main_model.judge_good_bad.assert_not_called()
            else:
                assert traj.final_answer == "B"
                loop.main_model.judge_good_bad.assert_called_once()
            if stale:
                assert any(s.answer == "stale_evidence" for s in traj.steps)
            assert any("assessed_version=1" in s.reasoning for s in traj.steps)
        finally:
            release.set()
            loop.shutdown()


def test_exhausted_draft_last_tool_is_assessed_exactly_once():
    loop = make_loop()
    pool = DraftPool()
    pool.add(tool_entry())
    done = Future()
    done.set_result(None)
    loop._assess_tool_entry = MagicMock(return_value=(
        {"passed": True, "result": None}, {"answer": "A", "status": "answer"}))
    traj = Trajectory()
    cancel = CancellableTask()
    cancel.cancel()
    try:
        loop._review_loop(Image.new("RGB", (2, 2)), "Q", pool, traj, cancel, done, FallbackCoordinator())
        assert traj.final_answer == "A"
        loop._assess_tool_entry.assert_called_once()
        loop.main_model.judge_good_bad.assert_not_called()
    finally:
        loop.shutdown()


def test_terminal_answer_has_exactly_one_winner():
    loop = make_loop()
    traj = Trajectory()
    barrier = threading.Barrier(2)
    def commit(answer):
        barrier.wait()
        return loop._try_set_final_answer(traj, answer, answer)
    try:
        with ThreadPoolExecutor(max_workers=2) as runner:
            futures = [runner.submit(commit, answer) for answer in ["A", "B"]]
            assert sorted(f.result() for f in futures) == [False, True]
        assert traj.final_answer == traj.stop_reason
    finally:
        loop.shutdown()
