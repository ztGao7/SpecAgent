import threading
import time

from PIL import Image

from spectool_agent.speculative_agent import SpeculativeAgentLoop
from spectool_agent.model import RequestCancelled
from spectool_agent.types import AgentConfig, ToolCall


class _Tools:
    def set_input_image(self, image):
        self.image = image

    def dispatch(self, tool_call, image):
        return "useful visual evidence", []


class _Draft:
    def __init__(self, qj_started=None):
        self.calls = 0
        self.qj_started = qj_started
        self.last_tool_calls = []
        self.last_tool_call_ids = []
        self.last_assistant_msg = {"role": "assistant", "content": ""}
        self.last_truncated = False

    def generate_full(self, messages, tools=None, max_tokens=None):
        self.calls += 1
        if self.calls == 1:
            time.sleep(0.03)
            self.last_tool_calls = [ToolCall("mock_tool", {})]
            self.last_tool_call_ids = ["draft-call-1"]
            self.last_assistant_msg = {
                "role": "assistant",
                "content": "",
                "tool_calls": [{
                    "id": "draft-call-1",
                    "type": "function",
                    "function": {"name": "mock_tool", "arguments": "{}"},
                }],
            }
            return ""

        # This in-flight request overlaps Main's combined QJ/answerability call.
        self.last_tool_calls = []
        self.last_tool_call_ids = []
        if self.qj_started is not None:
            assert self.qj_started.wait(0.5)
        time.sleep(0.15)
        return "<answer>B</answer>"


class _Main:
    last_tool_calls = []
    last_tool_call_ids = []
    last_assistant_msg = {"role": "assistant", "content": "<answer>B</answer>"}

    def __init__(self):
        self.qj_started = threading.Event()

    def assess_draft_step(self, messages, choices=None, cancel_token=None,
                          tools=None):
        self.qj_started.set()
        time.sleep(0.08)
        return {
            "score": 0.9,
            "label": "GOOD",
            "quality_valid": True,
            "answerable": True,
            "answerability_score": 0.99,
            "answer": "B",
            "raw": "GOOD<answer>B</answer>",
            "log_good": -0.1,
            "log_bad": -2.0,
        }

    def generate(self, messages, tools=None):
        return "<answer>B</answer>"


def test_current_pipeline_records_parallel_main_and_draft_intervals():
    config = AgentConfig(
        api_base="http://fake",
        model_name="main",
        quality_judge_enabled=True,
        quality_judge_threshold=0.7,
        draft_max_steps=2,
    )
    main = _Main()
    loop = SpeculativeAgentLoop(
        main, _Draft(main.qj_started), _Tools(), config)

    try:
        traj = loop.run(Image.new("RGB", (16, 16)), "Choose A or B")
    finally:
        loop.shutdown()

    assert traj.final_answer == "B"
    assert traj.stop_reason == "main_answer_from_draft_evidence"
    assert len([e for e in traj.timing_events if e["component"] == "draft"]) == 2
    assert len([e for e in traj.timing_events if e["component"] == "main"]) == 1
    assert traj.main_request_time > 0.05
    assert traj.draft_request_time > 0.15
    assert traj.model_request_time > traj.model_active_wall_time
    assert traj.model_overlap_time > 0.05
    assert 0.0 < traj.model_active_wall_ratio <= 1.0


class _SlowQJMain(_Main):
    def __init__(self):
        self.qj_started = threading.Event()
        self.qj_cancelled = threading.Event()

    def assess_draft_step(self, messages, choices=None, cancel_token=None,
                          tools=None):
        self.qj_started.set()
        deadline = time.monotonic() + 2.0
        while time.monotonic() < deadline:
            if cancel_token is not None and cancel_token.cancelled:
                self.qj_cancelled.set()
                raise RequestCancelled("test cancellation")
            time.sleep(0.005)
        raise AssertionError("QJ was not cancelled after Draft answered")


class _FastAnswerDraft(_Draft):
    def generate_full(self, messages, tools=None, max_tokens=None):
        self.calls += 1
        if self.calls == 1:
            time.sleep(0.01)
            self.last_tool_calls = [ToolCall("mock_tool", {})]
            self.last_tool_call_ids = ["draft-call-1"]
            self.last_assistant_msg = {
                "role": "assistant",
                "content": "",
                "tool_calls": [{
                    "id": "draft-call-1",
                    "type": "function",
                    "function": {"name": "mock_tool", "arguments": "{}"},
                }],
            }
            return ""
        self.last_tool_calls = []
        self.last_tool_call_ids = []
        if self.qj_started is not None:
            assert self.qj_started.wait(0.5)
        time.sleep(0.05)
        return "<answer>B</answer>"


def test_draft_answer_aborts_inflight_qj_and_runs_review():
    config = AgentConfig(
        api_base="http://fake",
        model_name="main",
        quality_judge_enabled=True,
        quality_judge_threshold=0.7,
        draft_max_steps=2,
    )
    main = _SlowQJMain()
    loop = SpeculativeAgentLoop(
        main, _FastAnswerDraft(main.qj_started), _Tools(), config)

    try:
        traj = loop.run(Image.new("RGB", (16, 16)), "Choose A or B")
    finally:
        loop.shutdown()

    assert main.qj_started.is_set()
    assert main.qj_cancelled.wait(0.5)
    assert traj.final_answer == "B"
    assert any(
        step.verdict_mode == "quality_judge_cancelled"
        for step in traj.steps
    )
