"""Regressions for VStar input aliases and Draft parser behavior."""
import json
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from PIL import Image

from spectool_agent.draft_model import DraftModel
from spectool_agent.model import VLLMModel
from spectool_agent.pool import DraftPool
from spectool_agent.prompts import MMSEARCH_PROMPT_SET
from spectool_agent.sandbox import (
    _init_code, _resolve_input_image_aliases, _shared_init_code)
from spectool_agent.speculative_agent import (
    CancellableTask, SpeculativeAgentLoop, _resolve_tool_turn)
from spectool_agent.types import AgentConfig, PoolEntry, ToolCall, Trajectory


@pytest.mark.parametrize("shared", [False, True])
@pytest.mark.parametrize("expression", [
    "Image.open('image_1')", "Image.open(fp='image_1')",
    "open_image('image_1', mode='r', formats=['PNG'])",
])
def test_reserved_input_alias_opens_actual_image(tmp_path, shared, expression):
    source = tmp_path / "source.png"
    Image.new("RGB", (17, 23), (100, 20, 30)).save(source)
    namespace = {}
    bootstrap = (_shared_init_code([str(source)], str(tmp_path), [str(source)])
                 if shared else _init_code([str(source)]))
    exec(bootstrap, namespace)
    original_open = Image.open
    exec(_resolve_input_image_aliases(
        "from PIL.Image import open as open_image\nimg = " + expression), namespace)
    assert namespace["img"].size == (17, 23)
    assert namespace["img"].getpixel((0, 0)) == (100, 20, 30)
    assert Image.open is original_open
    assert namespace["img"] is not namespace["image_1"]


def test_alias_repair_preserves_other_calls_and_later_turns(tmp_path):
    paths = []
    for index, color in enumerate(["red", "blue"]):
        path = tmp_path / f"{index}.png"
        Image.new("RGB", (3, 4), color).save(path)
        paths.append(str(path))
    namespace = {"Image": Image, "image_paths": paths}
    exec(_resolve_input_image_aliases(
        'from __future__ import annotations\n'
        'def later():\n    return Image.open("image_2")\n'
        'def custom(fp):\n    return fp\n'
        'unchanged = custom("image_1")\n'
        f'real = Image.open({paths[0]!r})'), namespace)
    assert namespace["unchanged"] == "image_1"
    assert namespace["real"].getpixel((0, 0)) == (255, 0, 0)
    del namespace["_spectool_open_input_alias"]  # transient state is not persisted
    exec(_resolve_input_image_aliases("result = later()"), namespace)
    assert namespace["result"].getpixel((0, 0)) == (0, 0, 255)
    with pytest.raises(FileNotFoundError):
        exec(_resolve_input_image_aliases('Image.open("image_3")'), namespace)


def _draft_run(content, *, truncated=False, tools=None, question=None,
               draft_max_steps=4):
    draft = SimpleNamespace(
        last_content=content, last_truncated=truncated,
        last_tool_calls=tools or [], last_tool_call_ids=[None] if tools else [],
        last_assistant_msg={"role": "assistant", "content": content})
    draft.generate_full = MagicMock(return_value="Reasoning considers B.\n" + content)
    registry = MagicMock()
    registry.dispatch.return_value = ("measured evidence", [])
    loop = SpeculativeAgentLoop(
        MagicMock(), draft, registry,
        AgentConfig(
            api_base="http://fake", model_name="fake",
            draft_max_steps=draft_max_steps,
        ))
    pool, traj = DraftPool(), Trajectory()
    try:
        loop._draft_loop(
            [{"role": "user", "content": question or "Color?\nA. red\nB. blue"}],
            Image.new("RGB", (4, 4)), pool, CancellableTask(), traj)
    finally:
        loop.shutdown()
    return draft, pool.get_all(), traj


def test_bare_final_letter_is_not_accepted_as_a_draft_candidate():
    draft, entries, traj = _draft_run(" A \n")
    assert draft.generate_full.call_count == 2
    assert all(entry.answer is None for entry in entries)
    assert traj.final_answer is None


def test_draft_last_step_has_no_forced_answer_reminder():
    draft, _, _ = _draft_run("still thinking", draft_max_steps=1)
    messages = draft.generate_full.call_args.args[0]
    assert "这是您的最后一步" not in str(messages)


def test_text_tool_fallback_uses_baseline_canonical_history_protocol():
    response = (
        "Inspect the image.\n"
        '<tool_call>{"name":"sandbox","arguments":{"code":"print(1)"}}'
        "</tool_call>"
    )
    model = SimpleNamespace(
        last_tool_calls=[],
        last_tool_call_ids=[],
        last_assistant_msg={"role": "assistant", "content": response},
    )

    calls, call_ids, assistant_message = _resolve_tool_turn(
        model, response, step_idx=3)

    assert len(calls) == 1
    assert call_ids == ["call_fallback_3_0"]
    assert assistant_message["tool_calls"][0]["id"] == "call_fallback_3_0"
    assert assistant_message["tool_calls"][0]["function"]["name"] == "sandbox"


@pytest.mark.parametrize("content,truncated,question", [
    ("A", True, None), ("C", False, None), ("A or B", False, None),
    ("I think A", False, None), ("<think>A</think>", False, None),
    ("", False, None), ("A", False, "Describe this image."),
])
def test_ambiguous_or_truncated_draft_stops_after_one_format_retry(
        content, truncated, question):
    draft, entries, traj = _draft_run(content, truncated=truncated, question=question)
    assert draft.generate_full.call_count == 2
    assert all(entry.answer is None for entry in entries)
    assert traj.final_answer is None


def test_tool_calls_take_priority_over_bare_letter():
    draft, entries, _ = _draft_run("A", tools=[ToolCall("sandbox", {"code": "print(1)"})])
    assert draft.generate_full.call_count == 4
    assert all(entry.tool_call is not None and entry.answer is None for entry in entries)


def test_draft_final_content_never_falls_back_to_reasoning(monkeypatch):
    client = MagicMock()
    message = SimpleNamespace(content=None, tool_calls=[],
                              model_dump=lambda: {"reasoning": "A"})
    client.chat.completions.create.return_value = SimpleNamespace(
        choices=[SimpleNamespace(message=message, finish_reason="stop")])
    monkeypatch.setattr("spectool_agent.draft_model.OpenAI", lambda **_: client)
    model = DraftModel("http://fake", "fake")
    assert model.generate_full([]) == "A"
    assert model.last_content == ""
    message.content = "B"
    model.generate_full([])
    assert model.last_content == "B"


def test_visual_review_keeps_evidence_without_draft_narrative():
    loop = SpeculativeAgentLoop(MagicMock(), MagicMock(), MagicMock(),
                               AgentConfig(api_base="http://fake", model_name="fake"))
    pool = DraftPool()
    pool.add(PoolEntry(step_idx=0, response_text="NARRATIVE_SENTINEL"))
    pool.add(PoolEntry(step_idx=1, tool_call=ToolCall("sandbox", {"code": "print(image_1.size)"}),
                       tool_observation="OBSERVATION_SENTINEL",
                       tool_result_images=[Image.new("RGB", (2, 2))]))
    pool.add(PoolEntry(step_idx=2, answer="D", reasoning="RATIONALE_SENTINEL"))
    try:
        messages = loop._build_main_context(Image.new("RGB", (2, 2)), "Which backpack?", pool, 3)
        text = json.dumps(messages)
        assert "NARRATIVE_SENTINEL" not in text
        assert "RATIONALE_SENTINEL" not in text
        assert "OBSERVATION_SENTINEL" in text
        assert text.count('"type": "image_url"') == 2
        assert "First locate" in text
        assert "Do not independently" not in text
        loop.prompt_set = MMSEARCH_PROMPT_SET
        report = json.dumps(loop._build_main_context(Image.new("RGB", (2, 2)), "Report", pool, 3))
        assert "NARRATIVE_SENTINEL" in report
        assert "Do not replace, extend, or independently derive" in report
    finally:
        loop.shutdown()


@pytest.mark.parametrize("score,raw,valid,accepted", [
    (0.9, "GOOD", True, True), (0.6, "GOOD", True, False),
    (0.99, "BAD", True, False), (0.99, "GOOD", False, False),
    (float("nan"), "GOOD", True, False), (float("inf"), "GOOD", True, False),
    (None, "GOOD", True, False), (0.99, "probably GOOD", True, False),
])
def test_final_review_rejects_invalid_verdicts(score, raw, valid, accepted):
    main = MagicMock()
    main.judge_good_bad.return_value = {"score": score, "raw": raw, "verdict_valid": valid}
    loop = SpeculativeAgentLoop(main, MagicMock(), MagicMock(),
        AgentConfig(api_base="http://fake", model_name="fake", quality_judge_threshold=0.7))
    pool = DraftPool()
    pool.add(PoolEntry(step_idx=0, answer="D"))
    try:
        verdict = loop._run_main_review(Image.new("RGB", (2, 2)), "Q", pool, 1, Trajectory())
        assert verdict["supported"] is accepted
    finally:
        loop.shutdown()


def test_label_only_good_does_not_count_as_valid_confidence(monkeypatch):
    client = MagicMock()
    client.chat.completions.create.return_value = SimpleNamespace(choices=[SimpleNamespace(
        message=SimpleNamespace(content="GOOD", model_dump=lambda: {}),
        logprobs=None, finish_reason="stop")])
    monkeypatch.setattr("spectool_agent.model.OpenAI", lambda **_: client)
    model = VLLMModel(AgentConfig(api_base="http://fake", model_name="fake"))
    assert model.judge_good_bad([])["verdict_valid"] is False


@pytest.mark.parametrize("finish_reason,valid", [("stop", True), ("length", False)])
def test_measured_judge_confidence_requires_completed_response(monkeypatch, finish_reason, valid):
    client = MagicMock()
    token = SimpleNamespace(token="GOOD", logprob=-0.1, top_logprobs=[
        SimpleNamespace(token="BAD", logprob=-3.0)])
    client.chat.completions.create.return_value = SimpleNamespace(choices=[SimpleNamespace(
        message=SimpleNamespace(content="GOOD", model_dump=lambda: {}),
        logprobs=SimpleNamespace(content=[token]), finish_reason=finish_reason)])
    monkeypatch.setattr("spectool_agent.model.OpenAI", lambda **_: client)
    verdict = VLLMModel(AgentConfig(api_base="http://fake", model_name="fake")).judge_good_bad([])
    assert verdict["verdict_valid"] is valid
    assert verdict["score"] > 0.9
