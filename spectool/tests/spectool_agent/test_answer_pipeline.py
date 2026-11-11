"""Answer-channel and final-schema regressions; no live model or sandbox calls."""
import copy
import json
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import MagicMock

import pytest
from PIL import Image

from spectool_agent.agent import AgentLoop
from spectool_agent.draft_model import DraftModel
from spectool_agent.model import (
    ModelResponse, VLLMModel, parse_answer, parse_tool_call, validate_final_answer,
)
from spectool_agent.pool import DraftPool
from spectool_agent.speculative_agent import CancellableTask, SpeculativeAgentLoop
from spectool_agent.tools import ToolRegistry
from spectool_agent.types import AgentConfig, PoolEntry, Trajectory

MCQ = "Choose one.\nA. Red\nB. Blue\nC. Green\nD. Yellow"
INTEGER = "Hint: Please answer the question requiring an integer answer and provide the final value.\nQuestion: How many?"
FLOAT = "Hint: Please answer the question requiring a floating-point number with two decimal places and provide the final value."
LIST = "Hint: Please answer the question requiring a Python list as an answer and provide the final list."


def response(content, reasoning="", finish="stop", native=False):
    tc = NS(id="tool_1", function=NS(name="sandbox", arguments='{"code":"print(1)"}'))
    msg = NS(content=content, tool_calls=[tc] if native else [],
             model_dump=lambda: {"reasoning_content": reasoning})
    return NS(choices=[NS(message=msg, finish_reason=finish)])


@pytest.mark.parametrize("role", ["main", "draft", "forced", "answerability"])
@pytest.mark.parametrize("reasoning", [
    "I should use <answer>...</answer>. ",
    "I need an <answer> tag. ",
    "Earlier guess: <answer>D</answer>. Now corrected. ",
    '<tool_call>{"name":"sandbox","arguments":{"code":"wrong"}}</tool_call>',
])
def test_response_parses_final_content_not_reasoning(monkeypatch, role, reasoning):
    client = MagicMock()
    client.chat.completions.create.return_value = response("<answer>A</answer>", reasoning)
    monkeypatch.setattr("spectool_agent.model.OpenAI", lambda **kw: client)
    monkeypatch.setattr("spectool_agent.draft_model.OpenAI", lambda **kw: client)
    cfg = AgentConfig(api_base="http://fake/v1", model_name="fake")
    if role == "draft":
        value = DraftModel(cfg.api_base, cfg.model_name).generate_full([])
    else:
        model = VLLMModel(cfg)
        if role == "main":
            value = model.generate([])
        elif role == "answerability":
            value = model.answer_if_sufficient([], choices=list("ABCD"))
        else:
            value = model.generate_forced([], choices=list("ABCD"))
    assert isinstance(value, str)
    assert value.reasoning == reasoning  # Logging still retains the full trace.
    assert value.content == "<answer>A</answer>"
    assert parse_answer(value, choices=list("ABCD"), question=MCQ) == "A"
    assert parse_tool_call(value) is None
    assert parse_answer(copy.deepcopy(value)) == "A"
    assert json.loads(json.dumps(value)) == str(value)


@pytest.mark.parametrize("content,finish", [(None,"length"),("","stop"),("<answer>A</answer>","length")])
def test_reasoning_only_or_truncated_generation_is_not_a_final_answer(content, finish):
    value = ModelResponse(content, "thinking <answer>A</answer>", finish)
    assert parse_answer(value) is None


@pytest.mark.parametrize("content,finish", [(None,"stop"),("","length"),("<answer>A</answer>","length")])
def test_answerability_does_not_fall_back_to_reasoning(monkeypatch, content, finish):
    client = MagicMock()
    client.chat.completions.create.return_value = response(content, "<answer>A</answer>", finish)
    monkeypatch.setattr("spectool_agent.model.OpenAI", lambda **kw: client)
    model = VLLMModel(AgentConfig(api_base="http://fake", model_name="fake"))
    assert parse_answer(model.answer_if_sufficient([], choices=list("ABCD"))) is None


def test_truncated_combined_assessment_cannot_publish_answer(monkeypatch):
    client = MagicMock()
    raw = "NA|<evidence>visible</evidence><answer>A</answer><verified>1</verified>"
    client.chat.completions.create.return_value = response(raw, finish="length")
    monkeypatch.setattr("spectool_agent.model.OpenAI", lambda **kw: client)
    model = VLLMModel(AgentConfig(api_base="http://fake", model_name="fake"))
    result = model.assess_draft_step([], choices=list("ABCD"), quality_enabled=False)
    assert result["answer"] is None
    assert result["answerable"] is False
    assert result["answerability_reason"] == "invalid_format"


@pytest.mark.parametrize("text", [
    "<answer>...</answer>", "<answer>…</answer>", "<answer></answer>",
    "<answer>A", "<answer>A<answer>B</answer>",
    "<answer>A</answer></answer>", "<answer>A</answer><answer>B</answer>",
    "<think>unfinished <answer>A</answer>",
])
def test_bad_answer_structure_is_rejected(text):
    assert parse_answer(text, choices=list("ABCD"), question=MCQ) is None


def test_explicit_think_block_is_ignored_but_math_symbols_survive():
    assert parse_answer("<think><answer>B</answer></think><answer>A</answer>") == "A"
    assert parse_answer("<answer>x < 1</answer>", question="What is the interval?") == "x < 1"
    assert parse_answer("<answer>A red car</answer>", question="Describe it") == "A red car"


@pytest.mark.parametrize("literal", ["<think>literal</think>", "<think>unfinished"])
def test_tool_code_tag_literals_are_not_rewritten(literal):
    code = f"print({literal!r})"
    payload = json.dumps({"name": "sandbox", "arguments": {"code": code}})
    assert parse_tool_call(f"<tool_call>{payload}</tool_call>").params["code"] == code


@pytest.mark.parametrize("answer,expected", [
    ("A","A"),("(b)","B"),("C.","C"),("F",None),("...",None),
    ("A or B",None),("42",None),("B. Blue","B"),("B. Red",None),
    ("A. I think B is right",None),
])
def test_mcq_schema(answer, expected):
    assert validate_final_answer(answer, choices=list("ABCD"), question=MCQ) == expected


@pytest.mark.parametrize("question,answer,expected", [
    (INTEGER,"-8","-8"),(INTEGER,"-7.5",None),(INTEGER,"8 items",None),
    (FLOAT,"1.06","1.06"),(FLOAT,"1e-3","1e-3"),
    (FLOAT,"1.06 V, 0.30 mA",None),(FLOAT,"nan",None),(FLOAT,"1e999",None),
    (LIST,"[2014, 2016]","[2014, 2016]"),(LIST,"2014,2016",None),
    ("What does integer mean?","whole number","whole number"),
])
def test_explicit_open_answer_hints_validate_without_guessing(question, answer, expected):
    assert parse_answer(f"<answer>{answer}</answer>", question=question) == expected


@pytest.fixture
def loop():
    main, draft = MagicMock(), MagicMock()
    for model in (main,draft):
        model.last_tool_calls=[]
        model.last_tool_call_ids=[]
        model.last_assistant_msg={"role":"assistant","content":""}
    draft.last_truncated=False
    cfg=AgentConfig(api_base="http://fake",model_name="fake",max_steps=3,
                    draft_max_steps=3,quality_judge_enabled=True)
    obj=SpeculativeAgentLoop(main,draft,ToolRegistry(),cfg)
    yield obj
    obj.shutdown()


@pytest.mark.parametrize("candidate", ["...","F","tag text <answer>A"])
def test_invalid_draft_candidate_cannot_pass_review_even_with_good_score(loop,candidate):
    pool,traj=DraftPool(),Trajectory()
    pool.add(PoolEntry(step_idx=0,answer=candidate))
    result=loop._run_main_review(Image.new("RGB",(2,2)),MCQ,pool,1,traj)
    assert result["supported"] is False
    loop.main_model.judge_good_bad.assert_not_called()
    # Guard publication independently, including alternate review adapters.
    result.update(supported=True,score=1.0)
    assert not loop._apply_review_result(result,pool,traj,None,MCQ,1,stop_reason="verified")
    assert traj.final_answer is None


def test_draft_repairs_invalid_output_within_existing_budget(loop):
    loop.draft_model.generate_full.side_effect=["<answer>F</answer>","<answer>B</answer>"]
    messages=[{"role":"system","content":"system"},
              {"role":"user","content":[{"type":"text","text":MCQ}]}]
    pool=DraftPool()
    loop._draft_loop(messages,Image.new("RGB",(2,2)),pool,CancellableTask(),Trajectory())
    assert pool.get_answer()=="B"
    assert pool.get_all()[0].answer is None
    assert loop.draft_model.generate_full.call_count==2


def test_fallback_rejects_invalid_option_then_accepts_next_valid_answer(loop):
    loop.main_model.generate.side_effect=["<answer>E</answer>","<answer>B</answer>"]
    traj=Trajectory()
    loop._run_fallback_steps([Image.new("RGB",(2,2))],MCQ,DraftPool(),traj,CancellableTask(),
        tools=loop.tools,inject_draft=False,qj_failed_entry=None,trigger="fallback_only")
    assert traj.final_answer=="B"
    assert traj.fallback_steps_used==2
    assert traj.fallback_step_budget==3
    loop.main_model.generate_forced.assert_not_called()


def test_fast_and_post_tool_answers_obey_numeric_schema(loop):
    traj=Trajectory()
    assert not loop._apply_fast_answer_result({"answerable":True,"answer":"1.0 V", "answerability_score":1.0},
        FLOAT,DraftPool(),traj)
    assert traj.steps[-1].answer=="initial_answerability_invalid_answer"
    loop.main_model.assess_draft_step.return_value={
        "score":1.0,"label":"GOOD","quality_valid":True,
        "answerable":True,"answerability_score":1.0,"answer":"1.0 V","raw":"raw"}
    _,outcome=loop._assess_tool_entry(Image.new("RGB",(2,2)),FLOAT,PoolEntry(step_idx=0))
    assert outcome["answer"] is None
    assert outcome["status"]=="invalid_answer"


def test_forced_adapter_uses_tagged_choice_protocol(monkeypatch):
    client=MagicMock()
    client.chat.completions.create.return_value=response("<answer>B</answer>","example <answer>A</answer>")
    monkeypatch.setattr("spectool_agent.model.OpenAI",lambda **kw:client)
    model=VLLMModel(AgentConfig(api_base="http://fake",model_name="fake"))
    result=model.generate_forced([],choices=list("ABCD"))
    request=client.chat.completions.create.call_args.kwargs
    assert request["extra_body"]["structured_outputs"]["choice"]==[f"<answer>{v}</answer>" for v in "ABCD"]
    assert request["extra_body"]["chat_template_kwargs"]["enable_thinking"] is False
    assert parse_answer(result,choices=list("ABCD"),question=MCQ)=="B"


def test_native_tool_calls_keep_priority_over_a_content_answer(monkeypatch):
    client=MagicMock()
    client.chat.completions.create.return_value=response("<answer>A</answer>",native=True)
    monkeypatch.setattr("spectool_agent.draft_model.OpenAI",lambda **kw:client)
    draft=DraftModel("http://fake","fake")
    value=draft.generate_full([])
    assert value.content=="<answer>A</answer>"
    assert draft.last_tool_calls[0].tool_name=="sandbox"
    assert draft.last_tool_call_ids==["tool_1"]


def test_response_content_does_not_leak_across_four_workers(monkeypatch):
    client=MagicMock()
    client.chat.completions.create.side_effect=lambda **kw:response(
        f"<answer>{kw['messages'][0]['content']}</answer>","example <answer>Z</answer>")
    monkeypatch.setattr("spectool_agent.model.OpenAI",lambda **kw:client)
    model=VLLMModel(AgentConfig(api_base="http://fake",model_name="fake"))
    def task(letter):
        value=model.generate([{"role":"user","content":letter}])
        return parse_answer(value,choices=list("ABCD"),question=MCQ)
    with ThreadPoolExecutor(max_workers=4) as executor:
        assert list(executor.map(task,"ABCD"))==list("ABCD")


def test_standard_agent_uses_same_schema(loop):
    loop.main_model.generate.side_effect=["<answer>F</answer>","<answer>A</answer>"]
    agent=AgentLoop(loop.main_model,ToolRegistry(),loop.config)
    result=agent.run(Image.new("RGB",(2,2)),MCQ)
    assert result.final_answer=="A"
    assert loop.main_model.generate.call_count==2


@pytest.mark.parametrize("model_name", ["Qwen3.5-27B", "Qwen3.5-9B"])
def test_baseline_and_draft_only_wrapper_share_fallback_validation(monkeypatch, tmp_path, model_name):
    from vlmeval.api.spectool_speculative import SpeculativeSpectoolAgentAPI

    for key in list(os.environ):
        if key.startswith("SPECTOOL_"):
            monkeypatch.delenv(key)
    client = MagicMock()
    client.chat.completions.create.side_effect = [
        response("<answer>F</answer>", "example <answer>A</answer>"),
        response("<answer>B</answer>", "reconsider <answer>C</answer>"),
    ]
    monkeypatch.setattr("spectool_agent.model.OpenAI", lambda **kw: client)
    draft_client = MagicMock(side_effect=AssertionError("fallback-only must not create a draft client"))
    monkeypatch.setattr("spectool_agent.draft_model.OpenAI", draft_client)
    monkeypatch.setattr("spectool_agent.tools.register_default_tools", lambda *a, **kw: [])
    path = tmp_path / "input.png"
    Image.new("RGB", (8, 8)).save(path)
    api = SpeculativeSpectoolAgentAPI(
        model=model_name, api_base="http://fake/v1", fallback_only=True,
        agent_repo_root=str(Path(__file__).resolve().parents[2]),
        max_steps=3, retry=1, verbose=False,
    )
    code, answer, record = api.generate_inner([
        {"type": "image", "value": str(path)}, {"type": "text", "value": MCQ},
    ], dataset="MMStar", sample_id="offline-test")
    assert (code, answer) == (0, "B")
    assert record["fallback_mode"] == "main_clean_restart"
    assert record["fallback_trigger"] == "fallback_only"
    assert record["fallback_steps_used"] == 2
    assert record["fallback_step_budget"] == 3
    assert record["draft_replica_count"] == 0
    assert client.chat.completions.create.call_count == 2
    draft_client.assert_not_called()
