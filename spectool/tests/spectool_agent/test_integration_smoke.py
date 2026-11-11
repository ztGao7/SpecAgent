"""End-to-end smoke test using mocked vLLM responses.

Verifies that a multi-turn tool-call trajectory flows through the wrapper,
through AgentLoop, through the tool registry, and out as a valid
extra_records payload that VLMEvalKit can persist.
"""
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
_SANDBOX_GRAYSCALE = "gray = image_1.convert('L').convert('RGB')\nplt.imshow(gray)\nplt.show()"


def _sandbox_ok_response(stdout=""):
    resp = MagicMock()
    resp.raise_for_status.return_value = None
    resp.json.return_value = {
        "status": "success",
        "output": {"stdout": stdout, "stderr": "", "result": "", "images": []},
    }
    return resp


def test_end_to_end_with_mocked_model(monkeypatch):
    from vlmeval.api.spectool_agent import SpectoolAgentAPI

    monkeypatch.setattr(
        "spectool_agent.sandbox.requests.post",
        lambda *args, **kwargs: _sandbox_ok_response(),
    )

    scripted = [
        ' thinkingneed a closer look response\n'
        '<tool_call>{"name": "sandbox", "arguments": '
        f'{{"code": {json.dumps(_SANDBOX_CROP)}}}}}'
        '</tool_call>',
        '<tool_call>{"name": "sandbox", "arguments": '
        f'{{"code": {json.dumps(_SANDBOX_GRAYSCALE)}}}}}'
        '</tool_call>',
        ' thinkingthat looks like a 7 response\n<answer>7</answer>',
    ]

    with tempfile.TemporaryDirectory() as tmpdir:
        img_path = os.path.join(tmpdir, "img.jpg")
        Image.new("RGB", (128, 128), color=(200, 100, 50)).save(img_path)

        agent_root = os.path.abspath(os.path.join(
            os.path.dirname(__file__), "..", ".."))

        with patch("spectool_agent.model.OpenAI"):
            api = SpectoolAgentAPI(
                model="fake", api_base="http://fake/v1",
                agent_repo_root=agent_root, retry=1, verbose=False,
                max_steps=5)

        api.agent.model.generate = MagicMock(side_effect=scripted)

        ret_code, answer, log = api.generate_inner(
            [{"type": "image", "value": img_path},
             {"type": "text", "value": "What number?"}],
            dataset="DummyBench",
        )

        assert ret_code == 0
        assert answer == "7"
        assert log["tool_call_count"] == 2
        assert log["stop_reason"] == "answer"
        assert log["dataset"] == "DummyBench"
        assert len(log["trajectory"]) == 3
        s = json.dumps(log["trajectory"])
        assert "sandbox" in s
        assert "7" in s
        assert log["total_inference_time"] >= 0
        assert log["total_tool_time"] >= 0
        assert log["total_time"] >= 0
