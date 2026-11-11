"""Main requests share the fallback prefix; reused turns rebuild sandbox state."""

import time
from unittest.mock import MagicMock

import pytest
from PIL import Image

from spectool_agent.pool import DraftPool
from spectool_agent.prompts import SYSTEM_PROMPT
from spectool_agent.speculative_agent import SpeculativeAgentLoop
from spectool_agent.tools import DEFAULT_TOOL_SCHEMAS, ToolRegistry
from spectool_agent.types import AgentConfig, PoolEntry, ToolCall, Trajectory


QUESTION = "Which object?\nA. first\nB. second"


def _config(**overrides):
    values = dict(api_base="http://fake", model_name="fake",
                  quality_judge_enabled=True, max_steps=5)
    values.update(overrides)
    return AgentConfig(**values)


def _tool_entry(step_idx, tool_name, code, *, passed=True):
    call_id = f"call-{step_idx}"
    return PoolEntry(
        step_idx=step_idx,
        tool_call=ToolCall(tool_name, {"code": code}),
        tool_call_id=call_id,
        assistant_message={
            "role": "assistant", "content": "",
            "tool_calls": [{"id": call_id, "type": "function",
                            "function": {"name": tool_name, "arguments": "{}"}}],
        },
        tool_observation=f"observation {step_idx}",
        quality_passed=passed,
    )


def test_every_main_request_opens_with_the_fallback_prefix():
    loop = SpeculativeAgentLoop(MagicMock(), MagicMock(), MagicMock(), _config())
    image = Image.new("RGB", (8, 8), "red")
    pool = DraftPool()
    tool_entry = _tool_entry(0, "sandbox", "print(1)")
    pool.add(tool_entry)
    pool.add(PoolEntry(step_idx=1, answer="B", reasoning="candidate"))
    try:
        shared = loop._shared_prompt_context(image, QUESTION)
        fallback = loop._build_main_takeover_state(
            image, QUESTION, DraftPool(), Trajectory(timing_origin=time.perf_counter()),
            inject_draft=False, qj_failed_entry=None, trigger="test").messages
        requests = {
            "fast": loop._build_fast_answer_context(image, QUESTION, shared_ctx=shared),
            "assessment": loop._build_combined_assessment_context(
                image, QUESTION, tool_entry, shared_ctx=shared),
            "verifier": loop._build_main_context(image, QUESTION, pool, 2,
                                                 shared_ctx=shared),
        }
    finally:
        loop.shutdown()

    assert fallback[0] == {"role": "system", "content": SYSTEM_PROMPT}
    for name, messages in requests.items():
        assert messages[:2] == fallback[:2], name
        assert "[ROLE FOR THIS REQUEST]" in messages[2]["content"][0]["text"], name


def test_judge_requests_send_the_fallback_tool_schemas():
    main = MagicMock()
    main.judge_good_bad.return_value = {"score": 0.9, "label": "GOOD", "raw": "GOOD"}
    loop = SpeculativeAgentLoop(main, MagicMock(), MagicMock(), _config())
    pool = DraftPool()
    pool.add(PoolEntry(step_idx=0, answer="B"))
    try:
        loop._run_main_review(Image.new("RGB", (2, 2)), QUESTION, pool, 1, Trajectory())
    finally:
        loop.shutdown()

    assert main.judge_good_bad.call_args.kwargs["tools"] is DEFAULT_TOOL_SCHEMAS


@pytest.mark.parametrize("method,kwargs", [
    ("judge_good_bad", {}),
    ("answer_if_sufficient", {"choices": ["A", "B"]}),
    ("assess_draft_step", {"choices": ["A", "B"]}),
])
def test_vllm_judges_render_tools_without_allowing_a_call(monkeypatch, method, kwargs):
    import spectool_agent.model as mod
    from spectool_agent.model import VLLMModel

    fake_client = MagicMock()
    fake_msg = MagicMock(content="GOOD")
    fake_msg.model_dump.return_value = {}
    fake_choice = MagicMock(message=fake_msg, logprobs=None)
    fake_client.chat.completions.create.return_value = MagicMock(choices=[fake_choice])
    monkeypatch.setattr(mod, "OpenAI", lambda **_: fake_client)
    model = VLLMModel(_config())

    getattr(model, method)([], tools=DEFAULT_TOOL_SCHEMAS, **kwargs)
    sent = fake_client.chat.completions.create.call_args.kwargs
    assert sent["tools"] is DEFAULT_TOOL_SCHEMAS
    assert sent["tool_choice"] == "none"

    getattr(model, method)([], **kwargs)
    sent = fake_client.chat.completions.create.call_args.kwargs
    assert "tools" not in sent and "tool_choice" not in sent


@pytest.fixture
def fake_sandbox(monkeypatch):
    """Replace the Jupyter call while keeping ToolRegistry's stateful identity."""
    executed = []

    def sandbox(self, image, code):
        executed.append((id(self), code))
        if code == "needs_hidden_variable":
            return ("沙箱错误: NameError: name 'hidden' is not defined", [])
        return ("(沙箱执行成功，无输出)", [])

    monkeypatch.setattr(ToolRegistry, "_sandbox", sandbox)
    return executed


def _registry():
    registry = ToolRegistry()
    registry.register("sandbox", registry._sandbox)
    registry.register("search", MagicMock(return_value=("web result", [])))
    registry.set_input_images([Image.new("RGB", (4, 4))])
    return registry


def test_reused_turns_are_replayed_and_cut_at_the_first_failed_replay(fake_sandbox):
    loop = SpeculativeAgentLoop(
        MagicMock(), MagicMock(), MagicMock(),
        _config(fallback_reuse_validated_evidence=True))
    pool = DraftPool()
    pool.add(_tool_entry(0, "sandbox", "a = 1"))
    pool.add(_tool_entry(1, "search", "query"))
    pool.add(_tool_entry(2, "sandbox", "needs_hidden_variable"))
    pool.add(_tool_entry(3, "sandbox", "b = a"))
    registry = _registry()
    traj = Trajectory(timing_origin=time.perf_counter())
    try:
        state = loop._build_main_takeover_state(
            Image.new("RGB", (4, 4)), QUESTION, pool, traj,
            inject_draft=False, qj_failed_entry=None, trigger="test",
            replay_tools=registry)
    finally:
        registry.cleanup()
        loop.shutdown()

    assert [code for _, code in fake_sandbox] == ["a = 1", "needs_hidden_variable"]
    registry._tools["search"].assert_not_called()
    assert state.validated_step_ids == (0, 1)
    assert state.step_budget == 3
    assert traj.fallback_replay_dropped_steps == 2
    assert any(event["phase"] == "fallback_state_replay"
               for event in traj.timing_events)


def test_replay_runs_only_when_validated_evidence_is_reused(fake_sandbox):
    loop = SpeculativeAgentLoop(MagicMock(), MagicMock(), MagicMock(), _config())
    pool = DraftPool()
    pool.add(_tool_entry(0, "sandbox", "a = 1"))
    registry = _registry()
    try:
        state = loop._build_main_takeover_state(
            Image.new("RGB", (4, 4)), QUESTION, pool,
            Trajectory(timing_origin=time.perf_counter()),
            inject_draft=False, qj_failed_entry=None, trigger="test",
            replay_tools=registry)
    finally:
        registry.cleanup()
        loop.shutdown()

    assert fake_sandbox == []
    assert state.validated_step_ids == ()


def test_clean_fallback_replays_into_its_own_forked_sandbox(fake_sandbox):
    main = MagicMock()
    main.generate.return_value = "<answer>B</answer>"
    main.last_tool_calls = []
    main.last_tool_call_ids = []
    registry = _registry()
    loop = SpeculativeAgentLoop(
        main, MagicMock(), registry,
        _config(fallback_reuse_validated_evidence=True))
    pool = DraftPool()
    pool.add(_tool_entry(0, "sandbox", "a = 1"))
    traj = Trajectory(timing_origin=time.perf_counter())
    try:
        loop._run_fallback_agent(
            [Image.new("RGB", (4, 4))], QUESTION, pool, traj,
            MagicMock(cancelled=False), inject_draft=False,
            trigger="quality_bad_streak")
    finally:
        registry.cleanup()
        loop.shutdown()

    assert traj.final_answer == "B"
    assert traj.fallback_validated_steps == 1
    [(owner, code)] = fake_sandbox
    assert code == "a = 1"
    assert owner != id(registry)
