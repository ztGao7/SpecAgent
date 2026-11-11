"""Regression coverage for the isolated MMSearch benchmark profile."""
from pathlib import Path
from unittest.mock import MagicMock, patch

from PIL import Image


def _schema_names(schemas):
    return [item["function"]["name"] for item in schemas]


def test_profile_resolution_and_sample_id_are_mmsearch_only():
    from spectool_agent.benchmarks import (
        DEFAULT_PROFILE,
        MMSEARCH_PROFILE,
        resolve_benchmark_profile,
        resolve_sample_id,
    )

    inputs = [{"type": "image", "value": "/tmp/mmsearch-017.jpg"}]
    assert resolve_benchmark_profile(dataset="MMSearchVDR_0_5") == MMSEARCH_PROFILE
    assert resolve_benchmark_profile(dataset="mm-search-mini") == MMSEARCH_PROFILE
    assert resolve_benchmark_profile(dataset="MathVision_MINI") == DEFAULT_PROFILE
    assert resolve_benchmark_profile("default", "MMSearchVDR_0_5") == DEFAULT_PROFILE
    assert resolve_sample_id("", inputs, dataset="MMSearchVDR_0_5") == "mmsearch-017"
    assert resolve_sample_id("", inputs, dataset="VStarBench") == ""
    assert resolve_sample_id("row-id", inputs, dataset="MMSearchVDR_0_5") == "row-id"


def test_legacy_environment_flag_cannot_leak_into_other_benches(monkeypatch):
    from spectool_agent.prompts import get_prompt_set
    from spectool_agent.tools import (DEFAULT_TOOL_SCHEMAS,
                                      MMSEARCH_TOOL_SCHEMAS,
                                      get_tool_schemas)

    monkeypatch.setenv("SPECTOOL_MMSEARCH_TOOLS", "1")
    default_prompts = get_prompt_set(dataset="VStarBench")
    mmsearch_prompts = get_prompt_set(dataset="MMSearchVDR_0_5")

    assert default_prompts.profile == "default"
    assert "sandbox(code=...)" in default_prompts.system_prompt
    assert "crop_and_search" not in default_prompts.system_prompt
    assert default_prompts.allow_early_answer is True
    assert default_prompts.normalize_mcq_answers is True
    assert get_tool_schemas(dataset="VStarBench") is DEFAULT_TOOL_SCHEMAS

    assert mmsearch_prompts.profile == "mmsearch"
    assert "self-contained final report" in mmsearch_prompts.system_prompt
    assert "crop_and_search" in mmsearch_prompts.system_prompt
    assert "thorough, multi-source investigations" in mmsearch_prompts.system_prompt
    assert "begin with one precise web query" not in mmsearch_prompts.system_prompt
    assert mmsearch_prompts.allow_early_answer is False
    assert mmsearch_prompts.normalize_mcq_answers is False
    schemas = get_tool_schemas(dataset="MMSearchVDR_0_5")
    assert schemas is MMSEARCH_TOOL_SCHEMAS

    search_params = schemas[0]["function"]["parameters"]["properties"]["query"]
    visit_params = schemas[1]["function"]["parameters"]["properties"]["url"]
    bbox_params = schemas[3]["function"]["parameters"]["properties"]["bbox"]
    assert search_params["minItems"] == 1
    assert "maxItems" not in search_params
    assert "maxItems" not in visit_params
    assert "maxItems" not in bbox_params
    assert "top 10 results" in schemas[0]["function"]["description"]


def test_tool_registration_is_selected_per_sample(monkeypatch):
    from spectool_agent.tools import ToolRegistry, register_default_tools

    default_registry = ToolRegistry()
    default_schemas = register_default_tools(
        default_registry, dataset="MMBench_DEV_EN")
    assert default_registry.names() == ["sandbox"]
    assert _schema_names(default_schemas) == ["sandbox"]

    def fake_register_mmsearch(registry):
        for name in ("search", "visit", "PythonInterpreter", "crop_and_search"):
            registry.register(name, lambda image, **kwargs: ("ok", []))

    monkeypatch.setattr(
        "spectool_agent.tools._register_mmsearch_tools", fake_register_mmsearch)
    mmsearch_registry = ToolRegistry()
    mmsearch_schemas = register_default_tools(
        mmsearch_registry, dataset="MMSearchVDR_0_5")
    assert mmsearch_registry.names() == [
        "PythonInterpreter", "crop_and_search", "search", "visit"]
    assert _schema_names(mmsearch_schemas) == [
        "search", "visit", "PythonInterpreter", "crop_and_search"]


def test_agent_loop_uses_instance_prompt_and_schema():
    from spectool_agent.agent import AgentLoop
    from spectool_agent.prompts import get_prompt_set
    from spectool_agent.tools import MMSEARCH_TOOL_SCHEMAS
    from spectool_agent.types import AgentConfig

    model = MagicMock()
    model.generate.return_value = "<answer>A supported MMSearch report.</answer>"
    model.last_tool_calls = []
    model.last_tool_call_ids = []
    model.last_assistant_msg = {"role": "assistant", "content": ""}
    tools = MagicMock()
    cfg = AgentConfig(api_base="http://fake", model_name="fake", max_steps=2)
    prompts = get_prompt_set("mmsearch")

    traj = AgentLoop(
        model, tools, cfg,
        prompt_set=prompts, tool_schemas=MMSEARCH_TOOL_SCHEMAS,
    ).run(Image.new("RGB", (8, 8)), "Identify this and explain the fact.")

    messages = model.generate.call_args.args[0]
    assert messages[0]["content"] == prompts.system_prompt
    assert model.generate.call_args.kwargs["tools"] is MMSEARCH_TOOL_SCHEMAS
    assert traj.final_answer == "A supported MMSearch report."


def test_mmsearch_final_step_disables_tools_and_returns_report():
    from spectool_agent.agent import AgentLoop
    from spectool_agent.prompts import get_prompt_set
    from spectool_agent.tools import MMSEARCH_TOOL_SCHEMAS, ToolRegistry
    from spectool_agent.types import AgentConfig

    model = MagicMock()
    model.generate.side_effect = [
        '<tool_call>{"name":"search","arguments":{"query":["x"]}}</tool_call>',
        "<answer>A bounded final report with supporting evidence.</answer>",
    ]
    model.last_tool_calls = []
    model.last_tool_call_ids = []
    model.last_assistant_msg = {"role": "assistant", "content": ""}
    tools = ToolRegistry()
    tools.register("search", lambda image, **kwargs: ("search evidence", []))
    cfg = AgentConfig(
        api_base="http://fake", model_name="fake", max_steps=2,
        max_tokens=1024, max_context_tokens=12000,
        max_total_output_tokens=8000,
    )

    traj = AgentLoop(
        model, tools, cfg,
        prompt_set=get_prompt_set("mmsearch"),
        tool_schemas=MMSEARCH_TOOL_SCHEMAS,
    ).run(Image.new("RGB", (8, 8)), "Identify this and report the evidence.")

    assert model.generate.call_args_list[0].kwargs["tools"] is MMSEARCH_TOOL_SCHEMAS
    assert model.generate.call_args_list[1].kwargs["tools"] is None
    assert traj.tool_call_count == 1
    assert traj.budget_trigger == "max_steps"
    assert traj.final_answer == "A bounded final report with supporting evidence."
    assert traj.stop_reason == "answer"


def test_mmsearch_context_budget_compacts_evidence_before_final_call():
    from spectool_agent.agent import AgentLoop
    from spectool_agent.prompts import get_prompt_set
    from spectool_agent.tools import MMSEARCH_TOOL_SCHEMAS, ToolRegistry
    from spectool_agent.types import AgentConfig

    model = MagicMock()
    model.generate.side_effect = [
        '<tool_call>{"name":"search","arguments":{"query":["x"]}}</tool_call>',
        "<answer>Report produced from compacted evidence.</answer>",
    ]
    model.last_tool_calls = []
    model.last_tool_call_ids = []
    model.last_assistant_msg = {"role": "assistant", "content": ""}
    tools = ToolRegistry()
    huge_evidence = "relevant search evidence " * 5000
    tools.register("search", lambda image, **kwargs: (huge_evidence, []))
    cfg = AgentConfig(
        api_base="http://fake", model_name="fake", max_steps=5,
        max_tokens=1000, max_context_tokens=8000,
        max_total_output_tokens=20000,
    )

    traj = AgentLoop(
        model, tools, cfg,
        prompt_set=get_prompt_set("mmsearch"),
        tool_schemas=MMSEARCH_TOOL_SCHEMAS,
    ).run(Image.new("RGB", (8, 8)), "Identify this and report the evidence.")

    final_messages = model.generate.call_args_list[1].args[0]
    flattened = str(final_messages)
    assert model.generate.call_args_list[1].kwargs["tools"] is None
    assert "Collected evidence from earlier MMSearch tool calls" in flattened
    assert len(flattened) < len(huge_evidence) // 2
    assert traj.budget_trigger == "context_tokens"
    assert traj.final_answer == "Report produced from compacted evidence."


def test_mmsearch_does_not_execute_tool_or_force_answer_after_budget():
    from spectool_agent.agent import AgentLoop
    from spectool_agent.prompts import get_prompt_set
    from spectool_agent.tools import MMSEARCH_TOOL_SCHEMAS, ToolRegistry
    from spectool_agent.types import AgentConfig

    model = MagicMock()
    model.generate.side_effect = [
        '<tool_call>{"name":"search","arguments":{"query":["first"]}}</tool_call>',
        '<tool_call>{"name":"search","arguments":{"query":["too late"]}}</tool_call>',
    ]
    model.generate_forced = MagicMock(return_value=(
        "<answer>Final report recovered without another search.</answer>"))
    model.last_tool_calls = []
    model.last_tool_call_ids = []
    model.last_assistant_msg = {"role": "assistant", "content": ""}
    tools = ToolRegistry()
    search = MagicMock(return_value=("enough evidence", []))
    tools.register("search", search)
    cfg = AgentConfig(
        api_base="http://fake", model_name="fake", max_steps=2,
        max_tokens=1024, max_context_tokens=12000,
        max_total_output_tokens=8000,
    )

    traj = AgentLoop(
        model, tools, cfg,
        prompt_set=get_prompt_set("mmsearch"),
        tool_schemas=MMSEARCH_TOOL_SCHEMAS,
    ).run(Image.new("RGB", (8, 8)), "Identify this and report the evidence.")

    assert search.call_count == 1
    assert model.generate.call_args_list[1].kwargs["tools"] is None
    model.generate_forced.assert_not_called()
    assert traj.tool_call_count == 1
    assert traj.stop_reason == "max_steps"
    assert traj.final_answer is None


def test_mmsearch_budget_environment_does_not_change_default_wrapper(monkeypatch):
    from vlmeval.api.spectool_agent import SpectoolAgentAPI

    monkeypatch.setenv("MMSEARCH_MAX_STEPS", "7")
    monkeypatch.setenv("MMSEARCH_MAX_TOKENS", "2500")
    monkeypatch.setenv("MMSEARCH_MAX_CONTEXT_TOKENS", "90000")
    monkeypatch.setenv("MMSEARCH_MAX_TOTAL_OUTPUT_TOKENS", "15000")
    monkeypatch.delenv("SPECTOOL_MAX_STEPS", raising=False)
    monkeypatch.delenv("SPECTOOL_MAX_TOKENS", raising=False)
    agent_root = str(Path(__file__).resolve().parents[2])

    with patch("spectool_agent.model.OpenAI"):
        mmsearch_api = SpectoolAgentAPI(
            model="fake", api_base="http://fake/v1",
            agent_repo_root=agent_root, benchmark_profile="mmsearch",
            retry=1, verbose=False)
        default_api = SpectoolAgentAPI(
            model="fake", api_base="http://fake/v1",
            agent_repo_root=agent_root, max_steps=3, max_tokens=777,
            retry=1, verbose=False)

    assert mmsearch_api.cfg.max_steps == 7
    assert mmsearch_api.cfg.max_tokens == 2500
    assert mmsearch_api.cfg.max_context_tokens == 90000
    assert mmsearch_api.cfg.max_total_output_tokens == 15000
    assert default_api.cfg.max_steps == 3
    assert default_api.cfg.max_tokens == 777
    assert default_api.cfg.max_context_tokens == 0
    assert default_api.cfg.max_total_output_tokens == 0


def test_speculative_mmsearch_requires_final_report_review():
    from spectool_agent.pool import DraftPool
    from spectool_agent.prompts import get_prompt_set
    from spectool_agent.speculative_agent import SpeculativeAgentLoop
    from spectool_agent.tools import MMSEARCH_TOOL_SCHEMAS, ToolRegistry
    from spectool_agent.types import AgentConfig, PoolEntry, ToolCall

    main = MagicMock()
    main.assess_draft_step.return_value = {
        "quality_valid": True,
        "score": 0.99,
        "label": "GOOD",
        "log_good": 0.0,
        "log_bad": -10.0,
        "answerable": True,
        "answerability_score": 0.99,
        "answerability_reason": "answer",
        "answer": "premature short answer",
        "raw": "GOOD|1|<answer>premature short answer</answer>",
    }
    cfg = AgentConfig(
        api_base="http://fake", model_name="fake",
        quality_judge_enabled=True, quality_judge_threshold=0.7)
    prompts = get_prompt_set("mmsearch")
    loop = SpeculativeAgentLoop(
        main, MagicMock(), ToolRegistry(), cfg,
        prompt_set=prompts, tool_schemas=MMSEARCH_TOOL_SCHEMAS)
    entry = PoolEntry(
        step_idx=0,
        tool_call=ToolCall("search", {"query": ["example"]}),
        tool_observation="useful evidence",
    )

    quality, answerability = loop._assess_tool_entry(
        Image.new("RGB", (8, 8)), "question", entry)
    review_messages = loop._build_main_context(
        Image.new("RGB", (8, 8)), "question", DraftPool(), 0)

    assert quality["passed"] is True
    assert answerability["answer"] is None
    assert answerability["status"] == "profile_requires_final_review"
    assert review_messages[0]["content"] == prompts.review_system_prompt
    assert "self-contained report" in review_messages[-1]["content"][0]["text"]
    loop.shutdown()


def test_mmsearch_wrapper_wires_profile_and_recovers_sample_id(tmp_path):
    from spectool_agent.tools import MMSEARCH_TOOL_SCHEMAS
    from spectool_agent.types import Trajectory
    from vlmeval.api.spectool_agent import SpectoolAgentAPI

    image_path = tmp_path / "sample-42.jpg"
    Image.new("RGB", (8, 8)).save(image_path)
    agent_root = str(Path(__file__).resolve().parents[2])
    with patch("spectool_agent.model.OpenAI"):
        api = SpectoolAgentAPI(
            model="fake", api_base="http://fake/v1",
            agent_repo_root=agent_root, benchmark_profile="mmsearch",
            retry=1, verbose=False)

    fake_agent = MagicMock()
    fake_agent.run.return_value = Trajectory(
        sample_id="sample-42", final_answer="report", stop_reason="answer")
    inputs = [
        {"type": "image", "value": str(image_path)},
        {"type": "text", "value": "question"},
    ]
    with patch("spectool_agent.tools.register_default_tools",
               return_value=MMSEARCH_TOOL_SCHEMAS), patch(
                   "spectool_agent.agent.AgentLoop", return_value=fake_agent
               ) as loop_cls:
        code, answer, metadata = api.generate_inner(
            inputs, dataset="MMSearchVDR_0_5")

    assert code == 0
    assert answer == "report"
    assert metadata["benchmark_profile"] == "mmsearch"
    assert fake_agent.run.call_args.kwargs["sample_id"] == "sample-42"
    assert loop_cls.call_args.kwargs["prompt_set"].profile == "mmsearch"
    assert loop_cls.call_args.kwargs["tool_schemas"] is MMSEARCH_TOOL_SCHEMAS


def test_default_wrapper_keeps_metadata_and_sample_id_unchanged(tmp_path):
    from spectool_agent.tools import DEFAULT_TOOL_SCHEMAS
    from spectool_agent.types import Trajectory
    from vlmeval.api.spectool_agent import SpectoolAgentAPI

    image_path = tmp_path / "ordinary-image.jpg"
    Image.new("RGB", (8, 8)).save(image_path)
    agent_root = str(Path(__file__).resolve().parents[2])
    with patch("spectool_agent.model.OpenAI"):
        api = SpectoolAgentAPI(
            model="fake", api_base="http://fake/v1",
            agent_repo_root=agent_root, retry=1, verbose=False)

    fake_agent = MagicMock()
    fake_agent.run.return_value = Trajectory(
        final_answer="B", stop_reason="answer")
    inputs = [
        {"type": "image", "value": str(image_path)},
        {"type": "text", "value": "question"},
    ]
    with patch("spectool_agent.tools.register_default_tools",
               return_value=DEFAULT_TOOL_SCHEMAS), patch(
                   "spectool_agent.agent.AgentLoop", return_value=fake_agent
               ) as loop_cls:
        code, answer, metadata = api.generate_inner(
            inputs, dataset="MMBench_DEV_EN")

    assert code == 0
    assert answer == "B"
    assert "benchmark_profile" not in metadata
    assert fake_agent.run.call_args.kwargs["sample_id"] == ""
    assert loop_cls.call_args.kwargs["prompt_set"].profile == "default"
    assert loop_cls.call_args.kwargs["tool_schemas"] is DEFAULT_TOOL_SCHEMAS
