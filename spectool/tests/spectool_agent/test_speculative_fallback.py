import threading
import time
from concurrent.futures import Future
from unittest.mock import MagicMock

import pytest
from PIL import Image

from spectool_agent.pool import DraftPool
from spectool_agent.prompts import (SPECTOOL_REVIEW_SYSTEM_PROMPT,
                                    SYSTEM_PROMPT)
from spectool_agent.speculative_agent import (
    CancellableTask,
    FallbackCoordinator,
    SpeculativeAgentLoop,
    _POOL_REASONING_CHARS,
)
from spectool_agent.types import AgentConfig, PoolEntry, ToolCall, Trajectory


def test_main_review_uses_candidate_evidence_verifier_prompt():
    config = AgentConfig(api_base="http://fake", model_name="fake")
    loop = SpeculativeAgentLoop(
        MagicMock(), MagicMock(), MagicMock(), config)
    pool = DraftPool()
    pool.add(PoolEntry(step_idx=0, response_text="draft evidence"))
    pool.add(PoolEntry(step_idx=1, answer="B", reasoning="draft candidate"))
    try:
        messages = loop._build_main_context(
            Image.new("RGB", (2, 2)), "Choose A or B.", pool, 0)
    finally:
        loop.shutdown()

    # The verifier shares the fallback opening; its role follows the question.
    assert messages[0] == {"role": "system", "content": SYSTEM_PROMPT}
    role = messages[2]["content"][0]["text"]
    assert SPECTOOL_REVIEW_SYSTEM_PROMPT in role
    assert "untrusted Draft tool outputs" in role.replace("\n", " ")
    assert "not a problem-solving request" in role
    assert "DRAFT CANDIDATE UNDER VERIFICATION" in (
        messages[-2]["content"][0]["text"])
    assert "Do not solve the question" in messages[-1]["content"][0]["text"]
    assert "GOOD or BAD only" in messages[-1]["content"][0]["text"]


def test_review_uses_original_and_tool_evidence_without_resolving():
    loop = SpeculativeAgentLoop(
        MagicMock(), MagicMock(), MagicMock(),
        AgentConfig(api_base="http://fake", model_name="fake"))
    pool = DraftPool()
    pool.add(PoolEntry(
        step_idx=0,
        tool_call=ToolCall(tool_name="crop", params={"box": [0, 0, 2, 2]}),
        tool_result_images=[Image.new("RGB", (2, 2))],
        tool_observation="A close view of the target.",
    ))
    pool.add(PoolEntry(step_idx=1, answer="B"))
    try:
        messages = loop._build_main_context(
            Image.new("RGB", (4, 4)), "Choose A or B.", pool, 2)
    finally:
        loop.shutdown()

    system = " ".join(messages[2]["content"][0]["text"].split())
    instruction = messages[-1]["content"][0]["text"]
    assert "original images are evidence in their own right" in system
    assert "A crop need not establish the whole answer by itself" in system
    assert "applying the question's reference frame" in system
    assert "do not restart the task" in system
    assert "claims, confidence, and candidate are not evidence" in system
    assert "Return BAD if evidence contradicts it" in system
    assert "additional evidence or unsupported assumptions" in instruction
    assert "original image is supplied only" not in system
    assert "would require new reasoning" not in instruction
    assert "supplied visual and tool evidence" in (
        messages[-2]["content"][0]["text"])
    assert sum(
        part.get("type") == "image_url"
        for msg in messages if isinstance(msg.get("content"), list)
        for part in msg["content"]
    ) == 2


@pytest.mark.parametrize(
    ("score", "passes"),
    [(0.69, False), (0.70, True), (0.70001, True)],
)
def test_original_only_candidate_final_review_uses_quality_judge_threshold(
        score, passes):
    main = MagicMock()
    main.judge_good_bad.return_value = {
        "score": score, "label": "GOOD", "raw": "GOOD"}
    tools = MagicMock()
    loop = SpeculativeAgentLoop(
        main, MagicMock(), tools,
        AgentConfig(api_base="http://fake", model_name="fake",
                    quality_judge_threshold=0.7,
                    answerability_threshold=0.95))
    pool = DraftPool()
    pool.add(PoolEntry(step_idx=0, answer="B"))
    traj = Trajectory()
    try:
        review = loop._run_main_review(
            Image.new("RGB", (2, 2)), "Question", pool, 1, traj)
        accepted = loop._apply_review_result(
            review, pool, traj, None, "Question", 1,
            stop_reason="main_answer_draft_evidence_verified")
    finally:
        loop.shutdown()

    assert accepted is passes
    assert traj.final_answer == ("B" if passes else None)
    assert traj.steps[-1].judge_score == score
    main.judge_good_bad.assert_called_once()
    main.generate.assert_not_called()
    tools.dispatch.assert_not_called()
    messages = main.judge_good_bad.call_args.args[0]
    assert "No tool call is required when the original images already support" in (
        " ".join(messages[2]["content"][0]["text"].split()))


def test_main_review_accepts_only_the_verified_draft_candidate():
    loop = SpeculativeAgentLoop(
        MagicMock(), MagicMock(), MagicMock(),
        AgentConfig(api_base="http://fake", model_name="fake"))
    try:
        traj = Trajectory()
        completed = loop._apply_review_result(
            {"candidate_answer": "A", "supported": True, "score": 0.9,
             "raw": "GOOD"},
            DraftPool(), traj, None, "question", 0,
            stop_reason="main_answer_draft_evidence_verified")
        assert completed
        assert traj.final_answer == "A"
        assert traj.stop_reason == "main_answer_draft_evidence_verified"
        assert traj.steps[-1].verdict_mode == "draft_answer_evidence_verifier"
    finally:
        loop.shutdown()


@pytest.mark.parametrize("length", [0, 10, 1500, 1501, 12000])
def test_candidate_excerpt_preserves_short_text_and_bounds_long_text(length):
    reasoning = "x" * length
    excerpt = SpeculativeAgentLoop._candidate_rationale_excerpt(reasoning)
    assert len(excerpt) <= _POOL_REASONING_CHARS
    if length <= _POOL_REASONING_CHARS:
        assert excerpt == reasoning
    else:
        assert "middle of Draft rationale omitted" in excerpt


def test_candidate_excerpt_keeps_late_correction_without_trusting_it():
    opening = "Initially I thought the object was on the left. "
    correction = (
        "After checking the crop, I corrected that impression: the red chair "
        "is on the right side of the road. The candidate is B, right.")
    reasoning = opening + ("Intermediate exploration. " * 150) + correction
    excerpt = SpeculativeAgentLoop._candidate_rationale_excerpt(reasoning)
    entry = PoolEntry(step_idx=3, answer="B", reasoning=reasoning)
    rendered = SpeculativeAgentLoop._format_pool_entry(entry)

    assert excerpt.startswith(opening)
    assert excerpt.endswith(correction)
    assert len(excerpt) == _POOL_REASONING_CHARS
    assert "Candidate answer (not evidence): B" in rendered
    assert "Draft rationale (unverified)" in rendered
    assert correction in rendered
    assert entry.reasoning == reasoning


def test_rejected_candidate_has_specific_fallback_trigger(monkeypatch):
    monkeypatch.setenv("SPECTOOL_FAST_ANSWER_ENABLED", "0")
    main = MagicMock()
    main.judge_good_bad.return_value = {
        "score": 0.2, "label": "BAD", "raw": "BAD"}
    draft = MagicMock()
    draft.generate_full.return_value = "Observed the target. <answer>B</answer>"
    draft.last_tool_calls = []
    draft.last_tool_call_ids = []
    draft.last_truncated = False
    loop = SpeculativeAgentLoop(
        main, draft, MagicMock(),
        AgentConfig(api_base="http://fake", model_name="fake",
                    quality_judge_threshold=0.65))
    fallback_calls = []

    def fallback(image, user_text, pool, traj, cancel, *, trigger, **kwargs):
        fallback_calls.append((trigger, pool.get_answer()))
        traj.final_answer = "A"
        traj.stop_reason = "main_answer_fallback"
        traj.fallback_trigger = trigger

    loop._run_fallback_agent = fallback
    try:
        traj = loop.run(Image.new("RGB", (2, 2)), "Choose A or B.")
    finally:
        loop.shutdown()

    assert fallback_calls == [("draft_answer_rejected", "B")]
    assert traj.final_answer == "A"
    assert traj.fallback_trigger == "draft_answer_rejected"
    main.judge_good_bad.assert_called_once()
    assert any(step.verdict_mode == "draft_answer_evidence_verifier"
               and step.answer.startswith("draft_answer_evidence_rejected")
               for step in traj.steps)
    assert any(step.verdict_mode == "draft_pool" and step.answer == "B"
               for step in traj.steps)


def test_main_review_uses_constrained_verdict_and_returns_draft_answer():
    main = MagicMock()
    main.judge_good_bad.return_value = {
        "score": 0.96, "label": "GOOD", "raw": "GOOD"}
    loop = SpeculativeAgentLoop(
        main, MagicMock(), MagicMock(),
        AgentConfig(api_base="http://fake", model_name="fake",
                    quality_judge_threshold=0.7))
    pool = DraftPool()
    pool.add(PoolEntry(step_idx=0, response_text="observed clue"))
    pool.add(PoolEntry(step_idx=1, answer="B", reasoning="candidate"))
    try:
        result = loop._run_main_review(
            Image.new("RGB", (2, 2)), "Choose A or B.", pool, 2,
            Trajectory())
    finally:
        loop.shutdown()

    assert result["candidate_answer"] == "B"
    assert result["supported"] is True
    main.judge_good_bad.assert_called_once()
    main.generate.assert_not_called()


def test_fallback_coordinator_runs_one_callback_for_concurrent_callers():
    coordinator = FallbackCoordinator()
    started = threading.Event()
    release = threading.Event()
    calls = []
    results = []

    def callback():
        calls.append(1)
        started.set()
        assert release.wait(timeout=2)

    def invoke():
        results.append(coordinator.run_once(callback))

    first = threading.Thread(target=invoke)
    second = threading.Thread(target=invoke)
    first.start()
    assert started.wait(timeout=2)
    second.start()
    time.sleep(0.05)
    assert calls == [1]

    release.set()
    first.join(timeout=2)
    second.join(timeout=2)

    assert sorted(results) == [False, True]


def test_speculative_run_records_wall_time():
    config = AgentConfig(api_base="http://fake", model_name="fake")
    loop = SpeculativeAgentLoop(
        MagicMock(), MagicMock(), MagicMock(), config)
    loop._draft_loop = lambda *args: None
    loop._review_loop = lambda *args: None

    def fake_fallback(image, user_text, pool, traj, cancel,
                      inject_draft=True, qj_failed_entry=None,
                      trigger="speculative_exhausted"):
        assert trigger == "speculative_exhausted"
        time.sleep(0.01)
        traj.final_answer = "A"

    loop._run_fallback_agent = fake_fallback
    try:
        traj = loop.run(Image.new("RGB", (1, 1)), "Choose A or B.")
    finally:
        loop.shutdown()

    assert traj.final_answer == "A"
    assert traj.total_time >= 0.01


def test_main_takeover_replays_validated_state_and_shares_step_budget():
    main = MagicMock()
    main.last_tool_calls = []
    main.last_tool_call_ids = []
    main.last_assistant_msg = {"role": "assistant", "content": "continue"}
    main.generate.side_effect = [
        "Need one more check.",
        "Still checking.",
        "<answer>B</answer>",
    ]
    config = AgentConfig(
        api_base="http://fake",
        model_name="fake",
        max_steps=5,
        fallback_clean_start=False,
        fallback_shared_step_budget=True,
        fallback_min_steps=1,
    )
    loop = SpeculativeAgentLoop(main, MagicMock(), MagicMock(), config)
    pool = DraftPool()
    for step_idx in range(2):
        call_id = f"call-{step_idx}"
        pool.add(PoolEntry(
            step_idx=step_idx,
            tool_call=ToolCall("sandbox", {"code": str(step_idx)}),
            tool_call_id=call_id,
            assistant_message={
                "role": "assistant",
                "content": "",
                "tool_calls": [{
                    "id": call_id,
                    "type": "function",
                    "function": {
                        "name": "sandbox",
                        "arguments": "{}",
                    },
                }],
            },
            tool_observation=f"validated-{step_idx}",
            quality_passed=True,
        ))
    pool.add(PoolEntry(
        step_idx=2,
        response_text="untrusted failed exploration",
        reasoning="untrusted",
        quality_passed=False,
    ))
    traj = Trajectory(timing_origin=time.perf_counter())

    try:
        loop._run_fallback_agent(
            Image.new("RGB", (8, 8)), "Choose A or B.", pool, traj,
            CancellableTask(), trigger="speculative_exhausted",
        )
    finally:
        loop.shutdown()

    assert traj.final_answer == "B"
    assert main.generate.call_count == 3
    assert traj.fallback_mode == "main_takeover"
    assert traj.fallback_trigger == "speculative_exhausted"
    assert traj.fallback_validated_steps == 2
    assert traj.fallback_passive_entries == 1
    assert traj.fallback_step_budget == 3
    assert traj.fallback_steps_used == 3
    assert not traj.fallback_forced
    messages = main.generate.call_args_list[0].args[0]
    assert sum(
        msg.get("role") == "assistant" and bool(msg.get("tool_calls"))
        for msg in messages
    ) == 2
    assert sum(msg.get("role") == "tool" for msg in messages) == 2
    transition_texts = [
        part.get("text", "")
        for message in messages
        for part in (
            message.get("content", [])
            if isinstance(message.get("content"), list) else []
        )
        if isinstance(part, dict)
    ]
    assert any("Main takeover, not a restart" in text
               for text in transition_texts)
    assert any("at most 3 iterative Main step(s) remain" in text
               for text in transition_texts)


def test_main_takeover_wall_budget_skips_iteration_and_returns_error():
    main = MagicMock()
    main.last_tool_calls = []
    main.last_tool_call_ids = []
    main.last_assistant_msg = {"role": "assistant", "content": ""}
    config = AgentConfig(
        api_base="http://fake",
        model_name="fake",
        max_steps=8,
        sample_wall_time_budget=0.5,
    )
    loop = SpeculativeAgentLoop(main, MagicMock(), MagicMock(), config)
    traj = Trajectory(timing_origin=time.perf_counter() - 1.0)

    try:
        loop._run_fallback_agent(
            Image.new("RGB", (8, 8)), "Choose A or B.", DraftPool(), traj,
            CancellableTask(), trigger="speculative_exhausted",
        )
    finally:
        loop.shutdown()

    main.generate.assert_not_called()
    main.generate_forced.assert_not_called()
    assert traj.final_answer == "error"
    assert traj.fallback_steps_used == 0
    assert not traj.fallback_forced
    assert traj.fallback_budget_exhausted
    assert traj.fallback_wall_elapsed_before >= 0.9


def test_main_takeover_counts_multi_tool_native_turn_as_one_step():
    main = MagicMock()
    main.last_tool_calls = []
    main.last_tool_call_ids = []
    main.last_assistant_msg = {"role": "assistant", "content": ""}
    main.generate.return_value = "<answer>A</answer>"
    config = AgentConfig(
        api_base="http://fake",
        model_name="fake",
        max_steps=3,
        fallback_clean_start=False,
    )
    loop = SpeculativeAgentLoop(main, MagicMock(), MagicMock(), config)
    native_turn = {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {"id": "call-a", "type": "function",
             "function": {"name": "sandbox", "arguments": "{}"}},
            {"id": "call-b", "type": "function",
             "function": {"name": "sandbox", "arguments": "{}"}},
        ],
    }
    pool = DraftPool()
    for call_id in ("call-a", "call-b"):
        pool.add(PoolEntry(
            step_idx=0,
            tool_call=ToolCall("sandbox", {"code": call_id}),
            tool_call_id=call_id,
            assistant_message=native_turn,
            tool_observation=call_id,
            quality_passed=True,
        ))
    traj = Trajectory(timing_origin=time.perf_counter())

    try:
        loop._run_fallback_agent(
            Image.new("RGB", (8, 8)), "Choose A or B.", pool, traj,
            CancellableTask(),
        )
    finally:
        loop.shutdown()

    assert traj.final_answer == "A"
    assert traj.fallback_validated_steps == 1
    assert traj.fallback_step_budget == 2
    messages = main.generate.call_args.args[0]
    assert sum(
        msg.get("role") == "assistant" and bool(msg.get("tool_calls"))
        for msg in messages
    ) == 1
    assert sum(msg.get("role") == "tool" for msg in messages) == 2


def test_first_tool_call_is_judged_and_one_bad_is_tolerated():
    config = AgentConfig(
        api_base="http://fake",
        model_name="fake",
        quality_judge_enabled=True,
    )
    loop = SpeculativeAgentLoop(
        MagicMock(), MagicMock(), MagicMock(), config)
    pool = DraftPool()
    pool.add(PoolEntry(
        step_idx=0,
        tool_call=ToolCall("sandbox", {"code": "1"}),
        tool_observation="useful evidence",
    ))
    traj = Trajectory()
    cancel = CancellableTask()
    draft_future = Future()
    first_judged = threading.Event()
    judge_calls = []

    def fake_assessment(*args, **kwargs):
        judge_calls.append(1)
        first_judged.set()
        return (
            {"passed": False, "result": None, "elapsed": 0.0,
             "step_idx": 0},
            {"answer": None, "raw": "INSUFFICIENT_EVIDENCE",
             "status": "insufficient", "elapsed": 0.0, "step_idx": 0},
        )

    loop._assess_tool_entry = fake_assessment
    loop._run_fallback_once = MagicMock()

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
        assert first_judged.wait(timeout=2)
        time.sleep(0.05)
        assert judge_calls == [1]
        assert pool.get_all()[0].quality_passed is False
        assert not draft_future.cancelled()
        loop._run_fallback_once.assert_not_called()

        traj.final_answer = "A"
        cancel.cancel()
        worker.join(timeout=2)
        assert not worker.is_alive()
    finally:
        loop.shutdown()


def test_two_consecutive_bad_qj_verdicts_start_fallback():
    config = AgentConfig(
        api_base="http://fake",
        model_name="fake",
        quality_judge_enabled=True,
    )
    loop = SpeculativeAgentLoop(
        MagicMock(), MagicMock(), MagicMock(), config)
    pool = DraftPool()
    for step_idx in range(2):
        pool.add(PoolEntry(
            step_idx=step_idx,
            tool_call=ToolCall("sandbox", {"code": str(step_idx)}),
            tool_observation="evidence",
        ))
    traj = Trajectory()
    cancel = CancellableTask()
    draft_future = Future()
    fallback_started = threading.Event()
    judge_calls = []

    def fake_assessment(*args, **kwargs):
        judge_calls.append(1)
        step_idx = len(judge_calls) - 1
        return (
            {"passed": False, "result": None, "elapsed": 0.0,
             "step_idx": step_idx},
            {"answer": None, "raw": "INSUFFICIENT_EVIDENCE",
             "status": "insufficient", "elapsed": 0.0,
             "step_idx": step_idx},
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
        assert judge_calls == [1, 1]
        assert traj.final_answer == "A"
        assert draft_future.cancelled()
    finally:
        loop.shutdown()


def test_good_qj_verdict_resets_consecutive_bad_count():
    config = AgentConfig(
        api_base="http://fake",
        model_name="fake",
        quality_judge_enabled=True,
    )
    loop = SpeculativeAgentLoop(
        MagicMock(), MagicMock(), MagicMock(), config)
    pool = DraftPool()
    for step_idx in range(4):
        pool.add(PoolEntry(
            step_idx=step_idx,
            tool_call=ToolCall("sandbox", {"code": str(step_idx)}),
            tool_observation="evidence",
        ))
    traj = Trajectory()
    cancel = CancellableTask()
    draft_future = Future()
    fallback_started = threading.Event()
    outcomes = iter([False, True, False, False])
    judge_calls = []

    def fake_assessment(*args, **kwargs):
        passed = next(outcomes)
        judge_calls.append(passed)
        step_idx = len(judge_calls) - 1
        return (
            {"passed": passed, "result": None, "elapsed": 0.0,
             "step_idx": step_idx},
            {"answer": None, "raw": "INSUFFICIENT_EVIDENCE",
             "status": "insufficient", "elapsed": 0.0,
             "step_idx": step_idx},
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
        assert judge_calls == [False, True, False, False]
        assert draft_future.cancelled()
    finally:
        loop.shutdown()


def test_qj_and_answerability_contexts_share_the_full_prefix():
    config = AgentConfig(
        api_base="http://fake",
        model_name="fake",
        quality_judge_enabled=True,
    )
    loop = SpeculativeAgentLoop(
        MagicMock(), MagicMock(), MagicMock(), config)
    entry = PoolEntry(
        step_idx=1,
        tool_call=ToolCall("sandbox", {"code": "print(1)"}),
        tool_observation="evidence",
    )
    image = Image.new("RGB", (2, 2))
    prefix = loop._build_step_assessment_prefix(
        image, "Question\nA. first\nB. second", entry)
    quality_messages = loop._build_quality_judge_context(
        image, "Question\nA. first\nB. second", entry,
        assessment_prefix=prefix)
    answer_messages = loop._build_answerability_context(
        image, "Question\nA. first\nB. second", entry,
        assessment_prefix=prefix)
    try:
        assert quality_messages[:-1] == answer_messages[:-1] == prefix
        assert quality_messages[-1] != answer_messages[-1]
        answer_instruction = answer_messages[-1]["content"][0]["text"]
        assert "abstention-first safety gate" in answer_instruction
        assert "default to INSUFFICIENT_EVIDENCE" in answer_instruction
        assert "When in doubt, always abstain" in answer_instruction
        assert "Ignore the Draft's stated answer" in answer_instruction
    finally:
        loop.shutdown()


def test_assessment_context_contains_only_current_tool_evidence():
    config = AgentConfig(api_base="http://fake", model_name="fake")
    loop = SpeculativeAgentLoop(MagicMock(), MagicMock(), MagicMock(), config)
    earlier = PoolEntry(
        step_idx=0,
        tool_call=ToolCall("sandbox", {"code": "print('first')"}),
        tool_observation="first measured clue",
        quality_passed=True,
    )
    latest = PoolEntry(
        step_idx=1,
        tool_call=ToolCall("sandbox", {"code": "print('second')"}),
        tool_observation="second measured clue",
    )
    try:
        messages = loop._build_step_assessment_prefix(
            Image.new("RGB", (2, 2)), "Choose A or B.", latest)
        evidence_text = "\n".join(
            part["text"]
            for message in messages
            for part in (message["content"] if isinstance(message["content"], list) else [])
            if part.get("type") == "text")
        assert "first measured clue" not in evidence_text
        assert "second measured clue" in evidence_text
        assert "PRIOR_QJ" not in evidence_text
    finally:
        loop.shutdown()


def test_fast_answer_prompt_distinguishes_direct_answer_from_tool_needed():
    config = AgentConfig(api_base="http://fake", model_name="fake")
    loop = SpeculativeAgentLoop(
        MagicMock(), MagicMock(), MagicMock(), config)
    try:
        messages = loop._build_fast_answer_context(
            Image.new("RGB", (2, 2)),
            "Where is the object?\nA. left\nB. right")
        instruction = messages[-1]["content"][0]["text"]
        assert "BOTH targets in the same reference frame" in instruction
        assert "ambiguous target identity" in instruction
        assert "Set verified to 1 only when" in instruction
        assert "abstain without a candidate" in instruction
        assert "A small target, a crowded scene" not in instruction
        assert "The mere availability of a crop" not in instruction
        assert "option elimination alone" in instruction
        assert "NA|<verified>0</verified>" in instruction
    finally:
        loop.shutdown()


def test_combined_assessment_separates_quality_from_answer_verification():
    config = AgentConfig(api_base="http://fake", model_name="fake")
    loop = SpeculativeAgentLoop(
        MagicMock(), MagicMock(), MagicMock(), config)
    entry = PoolEntry(
        step_idx=0,
        tool_call=ToolCall("sandbox", {"code": "print(1)"}),
        tool_observation="evidence",
    )
    try:
        messages = loop._build_combined_assessment_context(
            Image.new("RGB", (2, 2)),
            "Question\nA. first\nB. second", entry)
        instruction = messages[-1]["content"][0]["text"]
        assert "does not by itself establish an answer" in instruction
        assert "Either GOOD or BAD may prefix these forms" in instruction
        assert "Never put the verification decision before the candidate" in instruction
    finally:
        loop.shutdown()


def test_qj_and_answerability_use_one_combined_request():
    config = AgentConfig(
        api_base="http://fake",
        model_name="fake",
        quality_judge_enabled=True,
    )
    loop = SpeculativeAgentLoop(
        MagicMock(), MagicMock(), MagicMock(), config)
    entry = PoolEntry(
        step_idx=1,
        tool_call=ToolCall("sandbox", {"code": "print(1)"}),
        tool_observation="evidence",
    )
    loop.main_model.assess_draft_step.return_value = {
        "score": 0.8,
        "label": "GOOD",
        "log_good": -0.2,
        "log_bad": -1.6,
        "answer": None,
        "raw": "GOODINSUFFICIENT_EVIDENCE",
    }
    try:
        quality, answer = loop._assess_tool_entry(
            Image.new("RGB", (2, 2)), "Question", entry)
        loop.main_model.assess_draft_step.assert_called_once()
        assert quality["passed"] is True
        assert answer["status"] == "insufficient"
        assert answer["elapsed"] == 0.0
    finally:
        loop.shutdown()


def test_invalid_combined_qj_retries_with_structured_good_bad():
    config = AgentConfig(
        api_base="http://fake",
        model_name="fake",
        quality_judge_enabled=True,
        quality_judge_threshold=0.7,
    )
    loop = SpeculativeAgentLoop(
        MagicMock(), MagicMock(), MagicMock(), config)
    entry = PoolEntry(
        step_idx=1,
        tool_call=ToolCall("sandbox", {"code": "print(1)"}),
        tool_observation="useful evidence",
    )
    loop.main_model.assess_draft_step.return_value = {
        "score": None,
        "label": "NA",
        "quality_valid": False,
        "log_good": None,
        "log_bad": None,
        "answerable": False,
        "answerability_score": None,
        "answerability_reason": "invalid_format",
        "answer": None,
        "raw": "The crop provides useful evidence.",
    }
    loop.main_model.judge_good_bad.return_value = {
        "score": 0.92,
        "label": "GOOD",
        "log_good": -0.08,
        "log_bad": -2.6,
        "raw": "GOOD",
    }

    try:
        quality, answer = loop._assess_tool_entry(
            Image.new("RGB", (2, 2)), "Question", entry)
        loop.main_model.assess_draft_step.assert_called_once()
        loop.main_model.judge_good_bad.assert_called_once()
        assert quality["passed"] is True
        assert quality["result"]["format_retry"] is True
        assert quality["result"]["score"] == 0.92
        assert answer["status"] == "invalid_format"
    finally:
        loop.shutdown()


def test_combined_bad_quality_rejects_candidate_answer():
    config = AgentConfig(
        api_base="http://fake",
        model_name="fake",
        quality_judge_enabled=True,
        quality_judge_threshold=0.7,
    )
    loop = SpeculativeAgentLoop(
        MagicMock(), MagicMock(), MagicMock(), config)
    entry = PoolEntry(
        step_idx=1,
        tool_call=ToolCall("sandbox", {"code": "print(1)"}),
        tool_observation="misleading evidence",
    )
    loop.main_model.assess_draft_step.return_value = {
        "score": 0.1,
        "label": "BAD",
        "quality_valid": True,
        "log_good": -2.3,
        "log_bad": -0.1,
        "answerable": True,
        "answerability_score": 0.99,
        "answerability_reason": "answer",
        "answer": "B",
        "raw": "BAD|1|<answer>B</answer>",
    }

    try:
        quality, answer = loop._assess_tool_entry(
            Image.new("RGB", (2, 2)), "Question", entry)
        assert quality["passed"] is False
        assert answer["answer"] is None
        assert answer["candidate_answer"] == "B"
        assert answer["status"] == "quality_rejected"
    finally:
        loop.shutdown()


def test_evidence_answer_uses_shared_answerability_threshold():
    config = AgentConfig(
        api_base="http://fake",
        model_name="fake",
        quality_judge_enabled=True,
        quality_judge_threshold=0.7,
        answerability_threshold=0.95,
    )
    loop = SpeculativeAgentLoop(
        MagicMock(), MagicMock(), MagicMock(), config)
    entry = PoolEntry(
        step_idx=1,
        tool_call=ToolCall("sandbox", {"code": "print(1)"}),
        tool_observation="useful evidence",
    )
    loop.main_model.assess_draft_step.return_value = {
        "score": 0.99,
        "label": "GOOD",
        "quality_valid": True,
        "log_good": -0.01,
        "log_bad": -4.0,
        "answerable": True,
        "answerability_score": 0.80,
        "answerability_reason": "answer",
        "answer": "B",
        "raw": "GOOD|1|<answer>B</answer>",
    }
    try:
        _, answer = loop._assess_tool_entry(
            Image.new("RGB", (2, 2)), "Question", entry)
        assert answer["answer"] is None
        assert answer["status"] == "below_answerability_threshold"

        loop.config.answerability_threshold = 0.80
        _, answer = loop._assess_tool_entry(
            Image.new("RGB", (2, 2)), "Question", entry)
        assert answer["answer"] == "B"
        assert answer["status"] == "answer"
    finally:
        loop.shutdown()


def test_answerability_answer_does_not_override_bad_qj():
    config = AgentConfig(
        api_base="http://fake",
        model_name="fake",
        quality_judge_enabled=True,
    )
    loop = SpeculativeAgentLoop(
        MagicMock(), MagicMock(), MagicMock(), config)
    pool = DraftPool()
    pool.add(PoolEntry(
        step_idx=0,
        tool_call=ToolCall("sandbox", {"code": "1"}),
        tool_observation="decisive evidence",
    ))
    traj = Trajectory()
    cancel = CancellableTask()
    draft_future = Future()
    assessed = threading.Event()

    def fake_assessment(*args, **kwargs):
        assessed.set()
        return (
            {"passed": False, "result": None, "elapsed": 0.0,
             "step_idx": 0},
            {"answer": "B", "raw": "<answer>B</answer>",
             "status": "answer", "elapsed": 0.0, "step_idx": 0},
        )

    loop._assess_tool_entry = fake_assessment
    loop._run_fallback_once = MagicMock()
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
        assert assessed.wait(timeout=2)
        time.sleep(0.05)
        assert traj.final_answer is None
        assert pool.get_all()[0].quality_passed is False
        traj.final_answer = "A"
        cancel.cancel()
        worker.join(timeout=2)
        assert not worker.is_alive()
        assert traj.final_answer == "A"
        assert traj.stop_reason != "main_answer_from_draft_evidence"
        assert not draft_future.cancelled()
        loop._run_fallback_once.assert_not_called()
    finally:
        loop.shutdown()


def test_clean_fallback_reuses_only_complete_qj_approved_tool_turns():
    """Opt-in reuse preserves sandbox isolation and excludes untrusted Draft."""
    config = AgentConfig(
        api_base="http://fake",
        model_name="fake",
        max_steps=5,
        fallback_clean_start=True,
        fallback_reuse_validated_evidence=True,
        fallback_shared_step_budget=True,
    )
    loop = SpeculativeAgentLoop(MagicMock(), MagicMock(), MagicMock(), config)
    pool = DraftPool()
    pool.add(PoolEntry(
        step_idx=0,
        tool_call=ToolCall("sandbox", {"code": "validated"}),
        tool_call_id="validated-call",
        assistant_message={
            "role": "assistant",
            "content": "",
            "tool_calls": [{
                "id": "validated-call",
                "type": "function",
                "function": {"name": "sandbox", "arguments": "{}"},
            }],
        },
        tool_observation="validated observation",
        quality_passed=True,
    ))
    pool.add(PoolEntry(
        step_idx=1,
        reasoning="untrusted Draft answer must never be injected",
        answer="B",
        quality_passed=False,
    ))
    traj = Trajectory(timing_origin=time.perf_counter())
    try:
        state = loop._build_main_takeover_state(
            Image.new("RGB", (4, 4)), "Choose A or B.", pool, traj,
            inject_draft=True, qj_failed_entry=None,
            trigger="draft_answer_rejected",
        )
    finally:
        loop.shutdown()

    assert state.validated_step_ids == (0,)
    assert state.passive_entry_count == 0
    assert state.step_budget == 4
    assert traj.fallback_mode == "main_clean_restart_with_validated_evidence"
    assert sum(
        msg.get("role") == "assistant" and bool(msg.get("tool_calls"))
        for msg in state.messages
    ) == 1
    assert sum(msg.get("role") == "tool" for msg in state.messages) == 1
    rendered = "\n".join(
        str(part.get("text", ""))
        for msg in state.messages if isinstance(msg.get("content"), list)
        for part in msg["content"]
    )
    assert "untrusted Draft answer" not in rendered
