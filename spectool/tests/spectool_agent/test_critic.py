from unittest.mock import MagicMock

from PIL import Image

from spectool_agent.critic import SemanticCritic
from spectool_agent.types import ToolCall


def _make_critic() -> SemanticCritic:
    return SemanticCritic(api_base="http://fake/v1", model_name="fake-critic")


def _img():
    return Image.new("RGB", (10, 10), (0, 0, 0))


def test_both_none():
    critic = _make_critic()
    accepted, score = critic.verify(None, None, _img())
    assert accepted is True
    assert score == 1.0


def test_draft_none_main_some():
    critic = _make_critic()
    accepted, score = critic.verify(
        None, ToolCall("sandbox", {"code": "x"}), _img())
    assert accepted is False
    assert score == 0.0


def test_draft_some_main_none():
    critic = _make_critic()
    accepted, score = critic.verify(
        ToolCall("sandbox", {"code": "x"}), None, _img())
    assert accepted is False
    assert score == 0.0


def test_exact_match():
    critic = _make_critic()
    tc = ToolCall("sandbox", {"code": "import cv2"})
    accepted, score = critic.verify(tc, tc, _img())
    assert accepted is True
    assert score == 1.0


def test_api_error_falls_back_to_reject(monkeypatch):
    critic = _make_critic()
    client = MagicMock()
    client.chat.completions.create.side_effect = RuntimeError("boom")
    monkeypatch.setattr(critic, "_client", client)

    tc_a = ToolCall("sandbox", {"code": "a"})
    tc_b = ToolCall("sandbox", {"code": "b"})
    accepted, score = critic.verify(tc_a, tc_b, _img())
    assert accepted is False
    assert score == 0.0
