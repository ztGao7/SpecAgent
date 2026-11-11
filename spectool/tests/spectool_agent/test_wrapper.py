import os
import sys
import json
import tempfile
from unittest.mock import MagicMock, patch

from PIL import Image


def _add_vlmeval_to_path():
    here = os.path.dirname(os.path.abspath(__file__))
    vlm_path = os.path.abspath(os.path.join(here, "..", "..", "VLMEvalKit"))
    if vlm_path not in sys.path:
        sys.path.insert(0, vlm_path)


_add_vlmeval_to_path()


def _make_image_file(tmpdir, color=(123, 45, 67)):
    p = os.path.join(tmpdir, "img.jpg")
    Image.new("RGB", (32, 32), color=color).save(p)
    return p


def test_wrapper_happy_path():
    from vlmeval.api.spectool_agent import SpectoolAgentAPI
    from spectool_agent.types import Trajectory, StepResult, ToolCall

    with tempfile.TemporaryDirectory() as tmpdir:
        img_path = _make_image_file(tmpdir)
        agent_root = os.path.abspath(os.path.join(
            os.path.dirname(__file__), "..", ".."))

        # bypass OpenAI client construction
        with patch("spectool_agent.model.OpenAI") as _mock_openai:
            api = SpectoolAgentAPI(
                model="m", api_base="http://fake/v1",
                agent_repo_root=agent_root, retry=1, verbose=False)

        fake_traj = Trajectory(
            sample_id="s1", final_answer="B", tool_call_count=1,
            stop_reason="answer",
            total_inference_time=1.0, total_tool_time=0.2, total_time=1.3,
            steps=[
                StepResult(step_idx=0, reasoning="...",
                           tool_call=ToolCall("crop", {"box": [0, 0, 1, 1]}),
                           observation="ok", inference_time=0.5, tool_time=0.2),
                StepResult(step_idx=1, reasoning="...", answer="B",
                           inference_time=0.5),
            ])
        inputs = [{"type": "image", "value": img_path},
                  {"type": "text", "value": "What letter?"}]
        with patch("spectool_agent.speculative_agent.SpeculativeAgentLoop.run_main_only",
                   return_value=fake_traj):
            ret_code, answer, log = api.generate_inner(
                inputs, dataset="MMBench_DEV_EN")

        assert ret_code == 0
        assert answer == "B"
        assert isinstance(log, dict)
        assert log["tool_call_count"] == 1
        assert log["stop_reason"] == "answer"
        assert log["total_inference_time"] == 1.0
        assert log["dataset"] == "MMBench_DEV_EN"
        assert len(log["trajectory"]) == 2
        # trajectory entries are JSON-serializable dicts
        import json
        json.dumps(log["trajectory"])


def test_wrapper_no_answer_returns_fail():
    from vlmeval.api.spectool_agent import SpectoolAgentAPI
    from spectool_agent.types import Trajectory

    with tempfile.TemporaryDirectory() as tmpdir:
        img_path = _make_image_file(tmpdir)
        agent_root = os.path.abspath(os.path.join(
            os.path.dirname(__file__), "..", ".."))
        with patch("spectool_agent.model.OpenAI"):
            api = SpectoolAgentAPI(model="m", api_base="http://fake/v1",
                                   agent_repo_root=agent_root, retry=1, verbose=False)
        with patch("spectool_agent.speculative_agent.SpeculativeAgentLoop.run_main_only", return_value=Trajectory(
                stop_reason="max_steps", total_time=2.0)):
            ret_code, answer, log = api.generate_inner(
                [{"type": "image", "value": img_path},
                 {"type": "text", "value": "q"}])
        assert ret_code == 1
        assert "Failed" in answer or answer == api.fail_msg
        assert json.loads(log)["stop_reason"] == "max_steps"


def test_wrapper_no_image_raises_assertion():
    from vlmeval.api.spectool_agent import SpectoolAgentAPI
    agent_root = os.path.abspath(os.path.join(
        os.path.dirname(__file__), "..", ".."))
    with patch("spectool_agent.model.OpenAI"):
        api = SpectoolAgentAPI(model="m", api_base="http://fake/v1",
                               agent_repo_root=agent_root, retry=1, verbose=False)
    ret_code, answer, log = api.generate_inner(
        [{"type": "text", "value": "text only"}])
    assert ret_code == -1
    assert "输入解析错误" in json.loads(log)["error"]


def test_wrapper_preserves_all_input_images():
    from vlmeval.api.spectool_agent import SpectoolAgentAPI

    with tempfile.TemporaryDirectory() as tmpdir:
        first = _make_image_file(tmpdir, color=(255, 0, 0))
        second = os.path.join(tmpdir, "second.jpg")
        Image.new("RGB", (24, 16), color=(0, 0, 255)).save(second)
        agent_root = os.path.abspath(os.path.join(
            os.path.dirname(__file__), "..", ".."))
        with patch("spectool_agent.model.OpenAI"):
            api = SpectoolAgentAPI(
                model="m", api_base="http://fake/v1",
                agent_repo_root=agent_root, retry=1, verbose=False)

        images, question = api._parse_vlmeval_inputs([
            {"type": "image", "value": first},
            {"type": "image", "value": second},
            {"type": "text", "value": "compare"},
        ])

        assert len(images) == 2
        assert images[0].getpixel((0, 0))[0] > images[0].getpixel((0, 0))[2]
        assert images[1].getpixel((0, 0))[2] > images[1].getpixel((0, 0))[0]
        assert "image_1、image_2" in question
