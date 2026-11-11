"""End-to-end smoke test for speculative wrapper using mocked models."""
import json
import os
import sys
import tempfile
from unittest.mock import MagicMock, patch

from PIL import Image


def _add_vlmeval_to_path():
    here = os.path.dirname(os.path.abspath(__file__))
    vlm_path = os.path.abspath(os.path.join(here, "..", "..", "VLMEvalKit"))
    if vlm_path not in sys.path:
        sys.path.insert(0, vlm_path)


_add_vlmeval_to_path()


_SANDBOX_CROP = "crop = image_1.crop((0, 0, 64, 64))\nplt.imshow(crop)\nplt.show()"


def _sandbox_ok_response(stdout=""):
    resp = MagicMock()
    resp.raise_for_status.return_value = None
    resp.json.return_value = {
        "status": "success",
        "output": {"stdout": stdout, "stderr": "", "result": "", "images": []},
    }
    return resp


def _tool_call_json(code: str) -> str:
    escaped = json.dumps(code)
    return (
        '<tool_call>{"name": "sandbox", "arguments": '
        f'{{"code": {escaped}}}}}'
        '</tool_call>'
    )


def test_speculative_wrapper_preserves_multiple_input_images(tmp_path):
    from vlmeval.api.spectool_speculative import SpeculativeSpectoolAgentAPI

    first = tmp_path / "first.png"
    second = tmp_path / "second.png"
    Image.new("RGB", (8, 8), color=(255, 0, 0)).save(first)
    Image.new("RGB", (6, 10), color=(0, 0, 255)).save(second)
    api = SpeculativeSpectoolAgentAPI.__new__(SpeculativeSpectoolAgentAPI)

    images, question = api._parse_vlmeval_inputs([
        {"type": "image", "value": str(first)},
        {"type": "image", "value": str(second)},
        {"type": "text", "value": "Compare the two images."},
    ])

    assert [image.size for image in images] == [(8, 8), (6, 10)]
    assert "image_1、image_2" in question


def test_speculative_end_to_end_with_mocked_models(monkeypatch):
    from vlmeval.api.spectool_speculative import SpeculativeSpectoolAgentAPI

    monkeypatch.setattr(
        "spectool_agent.sandbox.requests.post",
        lambda *args, **kwargs: _sandbox_ok_response(),
    )

    # Draft: tool call then answer (returned as full text with tags)
    scripted_draft = [
        _tool_call_json(_SANDBOX_CROP),
        '<answer>7</answer>',
    ]

    with tempfile.TemporaryDirectory() as tmpdir:
        img_path = os.path.join(tmpdir, "img.jpg")
        Image.new("RGB", (128, 128), color=(200, 100, 50)).save(img_path)

        agent_root = os.path.abspath(os.path.join(
            os.path.dirname(__file__), "..", ".."))

        with patch("spectool_agent.model.OpenAI"):
            with patch("spectool_agent.draft_model.OpenAI"):
                api = SpeculativeSpectoolAgentAPI(
                    model="fake-main",
                    api_base="http://fake/v1",
                    draft_model_name="fake-draft",
                    agent_repo_root=agent_root,
                    retry=1,
                    verbose=False,
                    max_steps=5,
                )

        api.agent.draft_model.generate_full = MagicMock(side_effect=scripted_draft)

        ret_code, answer, log = api.generate_inner(
            [{"type": "image", "value": img_path},
             {"type": "text", "value": "What number?"}],
            dataset="DummyBench",
        )

        assert ret_code == 0
        assert answer == "7"
        assert log["stop_reason"] == "draft_answer"
        assert len(log["trajectory"]) == 0  # no main steps recorded


def test_speculative_no_answer_returns_fail(monkeypatch):
    from vlmeval.api.spectool_speculative import SpeculativeSpectoolAgentAPI

    monkeypatch.setattr(
        "spectool_agent.sandbox.requests.post",
        lambda *args, **kwargs: _sandbox_ok_response(),
    )

    # Draft only does tool calls, never answers
    scripted_draft = [
        _tool_call_json("print(1)"),
    ]

    with tempfile.TemporaryDirectory() as tmpdir:
        img_path = os.path.join(tmpdir, "img.jpg")
        Image.new("RGB", (128, 128)).save(img_path)

        agent_root = os.path.abspath(os.path.join(
            os.path.dirname(__file__), "..", ".."))

        with patch("spectool_agent.model.OpenAI"):
            with patch("spectool_agent.draft_model.OpenAI"):
                api = SpeculativeSpectoolAgentAPI(
                    model="fake-main",
                    api_base="http://fake/v1",
                    draft_model_name="fake-draft",
                    agent_repo_root=agent_root,
                    retry=1,
                    verbose=False,
                    max_steps=1,
                )

        api.agent.draft_model.generate_full = MagicMock(side_effect=scripted_draft * 5)

        ret_code, answer, log = api.generate_inner(
            [{"type": "image", "value": img_path},
             {"type": "text", "value": "q"}],
            dataset="DummyBench",
        )

        assert ret_code == 1
        assert answer == api.fail_msg
        assert "no_draft_answer" in log


def test_qj_fallback_replays_native_tools_with_bounded_images():
    """Fallback must not turn validated native tool calls into user text."""
    from spectool_agent.pool import DraftPool
    from spectool_agent.speculative_agent import CancellableTask, SpeculativeAgentLoop
    from spectool_agent.tools import ToolRegistry
    from spectool_agent.types import AgentConfig, PoolEntry, ToolCall, Trajectory

    main = MagicMock()
    main.last_tool_calls = []
    main.last_tool_call_ids = []
    main.generate.return_value = "<answer>A</answer>"
    cfg = AgentConfig(api_base="http://fake", model_name="fake-main",
                      max_steps=1, max_image_history=2,
                      fallback_clean_start=False)
    loop = SpeculativeAgentLoop(main, MagicMock(), ToolRegistry(), cfg)
    pool = DraftPool()
    native_turn = {
        "role": "assistant",
        "content": "",
        "tool_calls": [{"id": "call-1", "type": "function",
                        "function": {"name": "sandbox", "arguments": "{}"}}],
    }
    pool.add(PoolEntry(
        step_idx=1,
        tool_call=ToolCall("sandbox", {"code": "print(1)"}),
        tool_call_id="call-1",
        assistant_message=native_turn,
        tool_observation="ok",
        tool_result_images=[Image.new("RGB", (8, 8)) for _ in range(20)],
        quality_passed=True,
    ))
    failed = PoolEntry(step_idx=2, tool_call=ToolCall("sandbox", {}))

    traj = Trajectory()
    loop._run_fallback_agent(
        Image.new("RGB", (8, 8)), "Which option?", pool, traj,
        CancellableTask(), inject_draft=False, qj_failed_entry=failed,
        trigger="quality_bad_streak",
    )

    messages = main.generate.call_args.args[0]
    assert any(msg.get("role") == "assistant" and msg.get("tool_calls")
               for msg in messages)
    assert any(msg.get("role") == "tool" and msg.get("tool_call_id") == "call-1"
               for msg in messages)
    image_count = sum(
        1 for msg in messages for part in msg.get("content", [])
        if isinstance(part, dict) and part.get("type") == "image_url")
    assert image_count <= 2
    assert traj.final_answer == "A"
    assert traj.fallback_mode == "main_takeover"
    assert traj.fallback_trigger == "quality_bad_streak"
    assert traj.fallback_validated_steps == 1
    assert traj.fallback_step_budget == 1
    assert traj.fallback_steps_used == 1
    loop.shutdown()
