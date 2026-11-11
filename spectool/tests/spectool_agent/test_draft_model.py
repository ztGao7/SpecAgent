from unittest.mock import MagicMock

from spectool_agent.draft_model import DraftModel
from spectool_agent.types import ToolCall


def _make_draft(responses: list[str]) -> DraftModel:
    model = DraftModel(api_base="http://fake/v1", model_name="fake-draft")
    model._generate = MagicMock(side_effect=responses)
    return model


def test_predict_tool_call():
    draft = _make_draft([
        '{"tool_name": "sandbox", "params": {"code": "print(1)"}}',
    ])
    tc, raw = draft.predict_next_tool([{"role": "user", "content": "hi"}])
    assert isinstance(tc, ToolCall)
    assert tc.tool_name == "sandbox"
    assert tc.params == {"code": "print(1)"}
    assert raw == '{"tool_name": "sandbox", "params": {"code": "print(1)"}}'


def test_predict_finished():
    draft = _make_draft(['{"finished": true}'])
    tc, raw = draft.predict_next_tool([{"role": "user", "content": "hi"}])
    assert tc is None
    assert raw == '{"finished": true}'


def test_predict_invalid_json():
    draft = _make_draft(["not json at all"])
    tc, raw = draft.predict_next_tool([{"role": "user", "content": "hi"}])
    assert tc is None
    assert raw == "not json at all"


def test_predict_no_braces():
    draft = _make_draft(["no braces here"])
    tc, raw = draft.predict_next_tool([{"role": "user", "content": "hi"}])
    assert tc is None
    assert raw == "no braces here"


def test_predict_missing_tool_name():
    draft = _make_draft(['{"params": {"code": "x"}}'])
    tc, raw = draft.predict_next_tool([{"role": "user", "content": "hi"}])
    assert tc is None


def test_predict_with_prefix_text():
    draft = _make_draft([
        'some thinking text\n{"tool_name": "sandbox", "params": {}}',
    ])
    tc, raw = draft.predict_next_tool([{"role": "user", "content": "hi"}])
    assert tc is not None
    assert tc.tool_name == "sandbox"
    assert "some thinking text" in raw


def test_predict_empty_response():
    draft = _make_draft([""])
    tc, raw = draft.predict_next_tool([{"role": "user", "content": "hi"}])
    assert tc is None
    assert raw == ""
