from types import SimpleNamespace
from unittest.mock import MagicMock

from spectool_agent.draft_model import DraftModel
from spectool_agent.prompts import SYSTEM_PROMPT


def _draft_response():
    message = SimpleNamespace(
        content="draft response",
        tool_calls=None,
        model_dump=lambda: {},
    )
    return SimpleNamespace(choices=[SimpleNamespace(
        finish_reason="stop",
        message=message,
    )])


def _draft_with_capturing_client(monkeypatch):
    client = MagicMock()
    client.chat.completions.create.return_value = _draft_response()
    import spectool_agent.draft_model as module
    monkeypatch.setattr(module, "OpenAI", lambda **_: client)
    return DraftModel(api_base="http://fake/v1", model_name="fake-draft"), client


def test_generate_full_preserves_caller_draft_system_prompt(monkeypatch):
    draft, client = _draft_with_capturing_client(monkeypatch)
    messages = [
        {"role": "system", "content": "DRAFT ROLE SENTINEL"},
        {"role": "user", "content": "Inspect the image."},
    ]

    draft.generate_full(messages)

    assert client.chat.completions.create.call_args.kwargs["messages"] == messages


def test_generate_full_uses_default_prompt_when_caller_has_no_system_prompt(monkeypatch):
    draft, client = _draft_with_capturing_client(monkeypatch)
    messages = [{"role": "user", "content": "Inspect the image."}]

    draft.generate_full(messages)

    assert client.chat.completions.create.call_args.kwargs["messages"] == [
        {"role": "system", "content": SYSTEM_PROMPT},
        *messages,
    ]
