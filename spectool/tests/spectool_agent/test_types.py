from spectool_agent.types import ToolCall, StepResult, Trajectory, AgentConfig


def test_tool_call_construction():
    tc = ToolCall(tool_name="crop", params={"box": [0, 0, 0.5, 0.5]})
    assert tc.tool_name == "crop"
    assert tc.params == {"box": [0, 0, 0.5, 0.5]}


def test_step_result_defaults():
    sr = StepResult(step_idx=0)
    assert sr.step_idx == 0
    assert sr.reasoning == ""
    assert sr.tool_call is None
    assert sr.observation is None
    assert sr.answer is None
    assert sr.inference_time == 0.0
    assert sr.tool_time == 0.0


def test_trajectory_defaults():
    t = Trajectory()
    assert t.sample_id == ""
    assert t.steps == []
    assert t.final_answer is None
    assert t.tool_call_count == 0
    assert t.stop_reason == ""
    assert t.total_inference_time == 0.0
    assert t.total_tool_time == 0.0
    assert t.total_time == 0.0
    assert t.timing_version == 2
    assert t.timing_events == []
    assert t.model_active_wall_time == 0.0
    assert t.model_overlap_time == 0.0


def test_parallel_timing_uses_interval_union():
    t = Trajectory(timing_origin=100.0, total_time=10.0)
    t.record_timing_event("main", "quality", 100.0, 104.0)
    t.record_timing_event("draft", "draft_step", 101.0, 103.0)
    t.record_timing_event("tool", "draft_tool", 104.0, 106.0)
    t.record_timing_event("client", "context", 106.0, 107.0)

    t.finalize_timing()

    assert t.main_request_time == 4.0
    assert t.draft_request_time == 2.0
    assert t.total_draft_time == 2.0
    assert t.model_request_time == 6.0
    assert t.model_active_wall_time == 4.0
    assert t.model_overlap_time == 2.0
    assert t.tool_request_time == 2.0
    assert t.tool_active_wall_time == 2.0
    assert t.client_active_wall_time == 1.0
    assert t.pipeline_active_wall_time == 7.0
    assert t.orchestration_wait_time == 3.0
    assert t.model_active_wall_ratio == 0.4


def test_disjoint_model_requests_have_no_overlap():
    t = Trajectory(timing_origin=20.0, total_time=6.0)
    t.record_timing_event("draft", "draft_step", 20.0, 22.0)
    t.record_timing_event("main", "review", 23.0, 25.0)

    t.finalize_timing()

    assert t.model_request_time == 4.0
    assert t.model_active_wall_time == 4.0
    assert t.model_overlap_time == 0.0
    assert t.orchestration_wait_time == 2.0


def test_agent_config_defaults():
    cfg = AgentConfig(api_base="http://x", model_name="m")
    assert cfg.api_key == "EMPTY"
    assert cfg.temperature == 0.0
    assert cfg.max_tokens == 4096
    assert cfg.timeout == 120
    assert cfg.max_steps == 8
    assert cfg.max_image_history == 6
