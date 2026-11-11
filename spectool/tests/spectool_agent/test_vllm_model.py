import asyncio
import re
import threading
from types import SimpleNamespace

import pytest
from unittest.mock import AsyncMock, MagicMock

from spectool_agent.model import (_assistant_message_dict,
                                  _zero_one_distribution,
                                  RequestCancellation, RequestCancelled,
                                  VLLMModel)
from spectool_agent.types import AgentConfig


def test_vllm_model_generate_returns_message_content(monkeypatch):
    cfg = AgentConfig(api_base="http://fake/v1", model_name="fake-model")

    fake_client = MagicMock()
    fake_msg = MagicMock()
    fake_msg.content = "hello world"
    fake_msg.model_dump.return_value = {}
    fake_response = MagicMock()
    fake_response.choices = [MagicMock(message=fake_msg)]
    fake_client.chat.completions.create.return_value = fake_response

    def fake_openai_ctor(**kwargs):
        return fake_client

    import spectool_agent.model as mod
    monkeypatch.setattr(mod, "OpenAI", fake_openai_ctor)

    model = VLLMModel(cfg)
    out = model.generate([{"role": "user", "content": "hi"}])
    assert out == "hello world"

    # Verify it called with the right params
    fake_client.chat.completions.create.assert_called_once()
    kwargs = fake_client.chat.completions.create.call_args.kwargs
    assert kwargs["model"] == "fake-model"
    assert kwargs["temperature"] == 0.0
    assert kwargs["max_tokens"] == cfg.max_tokens
    assert "logprobs" not in kwargs
    assert "top_logprobs" not in kwargs


def test_vllm_model_generate_handles_none_content(monkeypatch):
    """vLLM occasionally returns choices[0].message.content == None
    (e.g. on length-stop with empty body). The agent loop's parsers
    don't accept None, so VLLMModel must normalize it to ''."""
    cfg = AgentConfig(api_base="http://fake/v1", model_name="fake-model")

    fake_client = MagicMock()
    fake_msg = MagicMock()
    fake_msg.content = None
    fake_msg.model_dump.return_value = {}
    fake_response = MagicMock()
    fake_response.choices = [MagicMock(message=fake_msg)]
    fake_client.chat.completions.create.return_value = fake_response

    import spectool_agent.model as mod
    monkeypatch.setattr(mod, "OpenAI", lambda **_: fake_client)

    out = VLLMModel(cfg).generate([{"role": "user", "content": "hi"}])
    assert out == ""


def test_assistant_message_preserves_reasoning_content():
    message = SimpleNamespace(
        content=None,
        reasoning_content="hidden reasoning",
        tool_calls=None,
        model_dump=lambda: {"reasoning_content": "hidden reasoning"},
    )

    assert _assistant_message_dict(message) == {
        "role": "assistant",
        "content": "hidden reasoning",
    }


def test_answer_if_sufficient_is_short_stateless_and_guided(monkeypatch):
    cfg = AgentConfig(api_base="http://fake/v1", model_name="fake-model")
    fake_client = MagicMock()
    fake_msg = MagicMock()
    fake_msg.content = "<answer>B</answer>"
    fake_msg.model_dump.return_value = {}
    fake_response = MagicMock()
    fake_response.choices = [MagicMock(message=fake_msg)]
    fake_client.chat.completions.create.return_value = fake_response

    import spectool_agent.model as mod
    monkeypatch.setattr(mod, "OpenAI", lambda **_: fake_client)

    model = VLLMModel(cfg)
    out = model.answer_if_sufficient(
        [{"role": "user", "content": "answer if sufficient"}],
        choices=["A", "B"],
    )

    assert out == "<answer>B</answer>"
    kwargs = fake_client.chat.completions.create.call_args.kwargs
    assert kwargs["max_tokens"] == 32
    assert kwargs["extra_body"]["chat_template_kwargs"]["enable_thinking"] is False
    assert kwargs["extra_body"]["structured_outputs"]["choice"] == [
        "<answer>A</answer>",
        "<answer>B</answer>",
        "INSUFFICIENT_EVIDENCE",
    ]


def test_assess_draft_step_returns_quality_and_answer_in_one_call(monkeypatch):
    cfg = AgentConfig(api_base="http://fake/v1", model_name="fake-model")
    fake_client = MagicMock()
    fake_msg = MagicMock()
    fake_msg.content = "GOOD|<evidence>the target is visible</evidence><answer>B</answer><verified>1</verified>"
    fake_msg.model_dump.return_value = {}
    fake_choice = MagicMock(message=fake_msg)
    fake_choice.logprobs = None
    fake_client.chat.completions.create.return_value = MagicMock(
        choices=[fake_choice])

    import spectool_agent.model as mod
    monkeypatch.setattr(mod, "OpenAI", lambda **_: fake_client)

    result = VLLMModel(cfg).assess_draft_step(
        [{"role": "user", "content": "assess"}], choices=["A", "B"])

    assert result["label"] == "GOOD"
    assert result["answer"] == "B"
    assert result["quality_valid"] is True
    fake_client.chat.completions.create.assert_called_once()
    kwargs = fake_client.chat.completions.create.call_args.kwargs
    pattern = kwargs["extra_body"]["structured_outputs"]["regex"]
    assert re.fullmatch(pattern, fake_msg.content)
    assert re.fullmatch(pattern, fake_msg.content.replace("verified>1", "verified>0"))
    assert re.fullmatch(pattern, "BAD|<verified>0</verified>")
    assert result["answerability_score"] is None


def test_fast_answer_gate_keeps_logprobs_for_threshold_score(monkeypatch):
    cfg = AgentConfig(api_base="http://fake/v1", model_name="fake-model")
    fake_client = MagicMock()
    fake_msg = MagicMock()
    fake_msg.content = "NA|<verified>0</verified>"
    fake_msg.model_dump.return_value = {}
    fake_choice = MagicMock(message=fake_msg)
    fake_choice.logprobs = None
    fake_client.chat.completions.create.return_value = MagicMock(
        choices=[fake_choice])

    import spectool_agent.model as mod
    monkeypatch.setattr(mod, "OpenAI", lambda **_: fake_client)

    VLLMModel(cfg).fast_answer_gate(
        [{"role": "user", "content": "answer if certain"}],
        choices=["A", "B"],
    )

    kwargs = fake_client.chat.completions.create.call_args.kwargs
    assert kwargs["max_tokens"] == 128
    assert kwargs["logprobs"] is True
    assert kwargs["top_logprobs"] == 20


def test_zero_one_distribution_uses_generated_decision_token():
    masked_candidates = [
        SimpleNamespace(token="0", logprob=-9999.0),
        SimpleNamespace(token="1", logprob=-9999.0),
    ]
    logprobs = [
        SimpleNamespace(
            token="NA", logprob=-0.001,
            top_logprobs=masked_candidates),
        SimpleNamespace(
            token="|", logprob=0.0,
            top_logprobs=masked_candidates),
        SimpleNamespace(
            token="1", logprob=-0.01,
            top_logprobs=[
                SimpleNamespace(token="1", logprob=-0.01),
                SimpleNamespace(token="0", logprob=-4.5),
            ]),
    ]

    assert _zero_one_distribution(logprobs) == {
        "0": -4.5,
        "1": -0.01,
    }


def test_assess_draft_step_does_not_turn_invalid_format_into_half_score(
        monkeypatch):
    cfg = AgentConfig(api_base="http://fake/v1", model_name="fake-model")
    fake_client = MagicMock()
    fake_msg = MagicMock()
    fake_msg.content = "The crop provides useful evidence."
    fake_msg.model_dump.return_value = {}
    fake_choice = MagicMock(message=fake_msg)
    fake_choice.logprobs = None
    fake_client.chat.completions.create.return_value = MagicMock(
        choices=[fake_choice])

    import spectool_agent.model as mod
    monkeypatch.setattr(mod, "OpenAI", lambda **_: fake_client)

    result = VLLMModel(cfg).assess_draft_step(
        [{"role": "user", "content": "assess"}], choices=["A", "B"])

    assert result["quality_valid"] is False
    assert result["score"] is None
    assert result["answerability_reason"] == "invalid_format"


def test_good_bad_judge_rejects_unstructured_response(monkeypatch):
    cfg = AgentConfig(api_base="http://fake/v1", model_name="fake-model")
    fake_client = MagicMock()
    fake_msg = MagicMock()
    fake_msg.content = "The crop provides useful evidence."
    fake_msg.model_dump.return_value = {}
    fake_choice = MagicMock(message=fake_msg)
    fake_choice.logprobs = None
    fake_client.chat.completions.create.return_value = MagicMock(
        choices=[fake_choice])

    import spectool_agent.model as mod
    monkeypatch.setattr(mod, "OpenAI", lambda **_: fake_client)

    with pytest.raises(ValueError, match="Structured GOOD/BAD"):
        VLLMModel(cfg).judge_good_bad(
            [{"role": "user", "content": "judge"}])

    kwargs = fake_client.chat.completions.create.call_args.kwargs
    assert kwargs["extra_body"]["structured_outputs"]["choice"] == [
        "GOOD", "BAD"]


def test_generate_forced_uses_vllm025_structured_choice(monkeypatch):
    cfg = AgentConfig(api_base="http://fake/v1", model_name="fake-model")
    fake_client = MagicMock()
    fake_msg = MagicMock()
    fake_msg.content = "<answer>B</answer>"
    fake_msg.model_dump.return_value = {}
    fake_client.chat.completions.create.return_value = MagicMock(
        choices=[MagicMock(message=fake_msg)])

    import spectool_agent.model as mod
    monkeypatch.setattr(mod, "OpenAI", lambda **_: fake_client)

    result = VLLMModel(cfg).generate_forced(
        [{"role": "user", "content": "answer"}], choices=["A", "B"])

    assert result == "<answer>B</answer>"
    kwargs = fake_client.chat.completions.create.call_args.kwargs
    assert kwargs["extra_body"]["structured_outputs"]["choice"] == [
        "<answer>A</answer>", "<answer>B</answer>"]
    assert "guided_choice" not in kwargs["extra_body"]


def test_generate_metadata_is_thread_local(monkeypatch):
    cfg = AgentConfig(api_base="http://fake/v1", model_name="fake-model")

    def create(**kwargs):
        marker = kwargs["messages"][-1]["marker"]
        function = SimpleNamespace(name=f"tool_{marker}", arguments="{}")
        tool_call = SimpleNamespace(id=f"call_{marker}", function=function)
        message = SimpleNamespace(
            content=marker,
            tool_calls=[tool_call],
            model_dump=lambda: {},
        )
        return SimpleNamespace(choices=[SimpleNamespace(message=message)])

    fake_client = MagicMock()
    fake_client.chat.completions.create.side_effect = create
    import spectool_agent.model as mod
    monkeypatch.setattr(mod, "OpenAI", lambda **_: fake_client)

    model = VLLMModel(cfg)
    generated = threading.Barrier(2)
    observed = {}

    def worker(marker):
        model.generate([{"marker": marker}], tools=[{"type": "function"}])
        generated.wait()
        observed[marker] = (
            model.last_tool_calls[0].tool_name,
            model.last_tool_call_ids[0],
            model.last_assistant_msg["content"],
        )

    workers = [threading.Thread(target=worker, args=(marker,))
               for marker in ("A", "B")]
    for worker_thread in workers:
        worker_thread.start()
    for worker_thread in workers:
        worker_thread.join()

    assert observed == {
        "A": ("tool_A", "call_A", "A"),
        "B": ("tool_B", "call_B", "B"),
    }


def test_generate_forced_inherits_global_thinking_for_open_answers(
        monkeypatch):
    cfg = AgentConfig(api_base="http://fake/v1", model_name="fake-model")
    fake_client = MagicMock()
    fake_msg = MagicMock()
    fake_msg.content = "<answer>open</answer>"
    fake_msg.model_dump.return_value = {}
    fake_client.chat.completions.create.return_value = MagicMock(
        choices=[MagicMock(message=fake_msg)])

    import spectool_agent.model as mod
    monkeypatch.setattr(mod, "OpenAI", lambda **_: fake_client)
    model = VLLMModel(cfg)

    monkeypatch.setenv("SPECTOOL_ENABLE_THINKING", "0")
    model.generate_forced([{"role": "user", "content": "answer"}])
    kwargs = fake_client.chat.completions.create.call_args.kwargs
    assert kwargs["extra_body"]["chat_template_kwargs"]["enable_thinking"] is False

    monkeypatch.setenv("SPECTOOL_ENABLE_THINKING", "1")
    model.generate_forced([{"role": "user", "content": "answer"}])
    kwargs = fake_client.chat.completions.create.call_args.kwargs
    assert kwargs["extra_body"]["chat_template_kwargs"]["enable_thinking"] is True

    model.generate_forced(
        [{"role": "user", "content": "answer"}], choices=["A", "B"])
    kwargs = fake_client.chat.completions.create.call_args.kwargs
    assert kwargs["extra_body"]["chat_template_kwargs"]["enable_thinking"] is False


def test_cancellable_assessment_preserves_normal_response(monkeypatch):
    cfg = AgentConfig(api_base="http://fake/v1", model_name="fake-model")
    fake_msg = MagicMock()
    fake_msg.content = "GOOD|<evidence>the target is visible</evidence><answer>B</answer><verified>1</verified>"
    fake_msg.model_dump.return_value = {}
    fake_choice = MagicMock(message=fake_msg)
    fake_choice.logprobs = None
    fake_response = MagicMock(choices=[fake_choice])
    async_client = MagicMock()
    async_client.chat.completions.create = AsyncMock(
        return_value=fake_response)
    async_client.close = AsyncMock()

    import spectool_agent.model as mod
    monkeypatch.setattr(mod, "OpenAI", lambda **_: MagicMock())
    monkeypatch.setattr(mod, "AsyncOpenAI", lambda **_: async_client)

    result = VLLMModel(cfg).assess_draft_step(
        [{"role": "user", "content": "assess"}], choices=["A", "B"],
        cancel_token=RequestCancellation())

    assert result["label"] == "GOOD"
    assert result["answer"] == "B"
    async_client.chat.completions.create.assert_awaited_once()
    async_client.close.assert_awaited_once()


class _BlockingAsyncCompletions:
    def __init__(self):
        self.entered = threading.Event()
        self.cancelled = threading.Event()
        self.kwargs = None

    async def create(self, **kwargs):
        self.kwargs = kwargs
        self.entered.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            self.cancelled.set()
            raise


class _BlockingAsyncClient:
    def __init__(self):
        self.completions = _BlockingAsyncCompletions()
        self.chat = SimpleNamespace(completions=self.completions)
        self.closed = threading.Event()

    async def close(self):
        self.closed.set()


def test_assess_draft_step_cancels_async_request_before_headers(monkeypatch):
    cfg = AgentConfig(api_base="http://fake/v1", model_name="fake-model")
    async_client = _BlockingAsyncClient()

    import spectool_agent.model as mod
    monkeypatch.setattr(mod, "OpenAI", lambda **_: MagicMock())
    monkeypatch.setattr(mod, "AsyncOpenAI", lambda **_: async_client)

    model = VLLMModel(cfg)
    cancel = RequestCancellation()
    errors = []

    def run_assessment():
        try:
            model.assess_draft_step(
                [{"role": "user", "content": "assess"}],
                choices=["A", "B"], cancel_token=cancel)
        except Exception as exc:
            errors.append(exc)

    worker = threading.Thread(target=run_assessment)
    worker.start()
    assert async_client.completions.entered.wait(0.5)
    cancel.cancel()
    worker.join(1.0)

    assert not worker.is_alive()
    assert async_client.completions.cancelled.is_set()
    assert async_client.closed.is_set()
    assert len(errors) == 1
    assert isinstance(errors[0], RequestCancelled)
    assert async_client.completions.kwargs["model"] == "fake-model"
