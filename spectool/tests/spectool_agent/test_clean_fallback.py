from unittest.mock import MagicMock

from PIL import Image

from spectool_agent.pool import DraftPool
from spectool_agent.speculative_agent import (
    CancellableTask,
    SpeculativeAgentLoop,
)
from spectool_agent.tools import ToolRegistry
from spectool_agent.types import AgentConfig, PoolEntry, ToolCall, Trajectory


def test_clean_fallback_drops_every_draft_message_and_resets_sandbox():
    main = MagicMock()
    main.last_tool_calls = []
    main.last_tool_call_ids = []
    main.last_assistant_msg = {"role": "assistant", "content": ""}
    main.generate.return_value = "<answer>A</answer>"
    tools = ToolRegistry()
    tools.reset_sandbox_context = MagicMock()
    loop = SpeculativeAgentLoop(
        main, MagicMock(), tools,
        AgentConfig(api_base="http://fake", model_name="fake", max_steps=3),
    )
    pool = DraftPool()
    pool.add(PoolEntry(
        step_idx=0,
        tool_call=ToolCall("sandbox", {"code": "leaked_tool_call"}),
        tool_observation="leaked observation",
        assistant_message={"role": "assistant", "content": "leaked assistant"},
        quality_passed=True,
    ))
    pool.add(PoolEntry(
        step_idx=1,
        answer="D",
        reasoning="leaked candidate rationale",
    ))
    traj = Trajectory()

    try:
        loop._run_fallback_agent(
            Image.new("RGB", (8, 8)), "Choose A or B.", pool, traj,
            CancellableTask(), trigger="speculative_exhausted",
        )
    finally:
        loop.shutdown()

    assert traj.final_answer == "A"
    assert traj.fallback_mode == "main_clean_restart"
    assert traj.fallback_validated_steps == 0
    assert traj.fallback_passive_entries == 0
    assert traj.fallback_step_budget == 3
    # The clean branch owns another registry; Draft state must not be reset
    # while its old tool call might still be running.
    tools.reset_sandbox_context.assert_not_called()
    messages = main.generate.call_args.args[0]
    assert [message["role"] for message in messages] == ["system", "user"]
    serialized = repr(messages)
    assert "leaked_tool_call" not in serialized
    assert "leaked observation" not in serialized
    assert "leaked assistant" not in serialized
    assert "leaked candidate rationale" not in serialized


def test_reset_sandbox_context_clears_only_the_persisted_session(monkeypatch):
    tools = ToolRegistry()
    tools._previous_context = {"session_id": "draft-session"}
    cleared = []
    monkeypatch.setattr(
        "spectool_agent.tools._sandbox.clear_session",
        lambda context: cleared.append(context),
    )

    tools.reset_sandbox_context()

    assert cleared == [{"session_id": "draft-session"}]
    assert tools._previous_context is None
