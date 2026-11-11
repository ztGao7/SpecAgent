from unittest.mock import MagicMock
from PIL import Image

from spectool_agent import prompts
from spectool_agent.agent import AgentLoop
from spectool_agent.prompts import (DRAFT_SYSTEM_PROMPT,
                                    SPECTOOL_REVIEW_SYSTEM_PROMPT,
                                    SYSTEM_PROMPT)
from spectool_agent.tools import ToolRegistry, register_default_tools
from spectool_agent.types import AgentConfig, ToolCall


def make_loop(responses: list[str]) -> AgentLoop:
    cfg = AgentConfig(api_base="http://fake", model_name="m", max_steps=5)
    model = MagicMock()
    model.generate.side_effect = responses
    tools = ToolRegistry()
    register_default_tools(tools)
    return AgentLoop(model, tools, cfg)


def _sandbox_code(code):
    return (
        '<tool_call>{"name": "sandbox", "arguments": '
        f'{{"code": {__import__("json").dumps(code)}}}}}'
        '</tool_call>'
    )


def _sandbox_ok_response(stdout=""):
    resp = MagicMock()
    resp.raise_for_status.return_value = None
    resp.json.return_value = {
        "status": "success",
        "output": {"stdout": stdout, "stderr": "", "result": "", "images": []},
    }
    return resp


def _mock_sandbox_posts(monkeypatch):
    monkeypatch.setattr(
        "spectool_agent.sandbox.requests.post",
        lambda *args, **kwargs: _sandbox_ok_response(),
    )


def test_answer_on_first_turn():
    loop = make_loop(["<answer>B</answer>"])
    img = Image.new("RGB", (40, 40), (0, 0, 0))
    traj = loop.run(img, "What letter?")
    assert traj.final_answer == "B"
    assert traj.stop_reason == "answer"
    assert traj.tool_call_count == 0
    assert len(traj.steps) == 1
    assert traj.steps[0].inference_time >= 0.0


def test_full_agent_uses_standard_prompt_without_shared_policy():
    loop = make_loop(["<answer>B</answer>"])
    loop.run(Image.new("RGB", (40, 40)), "What letter?")

    messages = loop.model.generate.call_args.args[0]
    assert messages[0] == {
        "role": "system",
        "content": SYSTEM_PROMPT,
    }
    assert not hasattr(prompts, "SHARED_VERIFICATION_POLICY")
    assert SYSTEM_PROMPT in DRAFT_SYSTEM_PROMPT
    assert "[Spectool draft role]" in DRAFT_SYSTEM_PROMPT
    assert "evidence verifier" in SPECTOOL_REVIEW_SYSTEM_PROMPT


def test_tool_call_then_answer(monkeypatch):
    _mock_sandbox_posts(monkeypatch)
    responses = [
        _sandbox_code('crop = image_1.crop((0, 0, 20, 20))\nplt.imshow(crop)\nplt.show()'),
        '<answer>red</answer>',
    ]
    loop = make_loop(responses)
    img = Image.new("RGB", (40, 40), (255, 0, 0))
    traj = loop.run(img, "What color?")
    assert traj.final_answer == "red"
    assert traj.tool_call_count == 1
    assert len(traj.steps) == 2
    assert traj.steps[0].tool_call is not None
    assert traj.steps[0].tool_call.tool_name == "sandbox"
    assert traj.steps[0].tool_time > 0.0


def test_max_steps_exhausted(monkeypatch):
    _mock_sandbox_posts(monkeypatch)
    loop = make_loop([_sandbox_code('print("hello")')] * 10)
    img = Image.new("RGB", (40, 40), (255, 0, 0))
    traj = loop.run(img, "q")
    assert traj.final_answer is None
    assert traj.stop_reason == "max_steps"
    assert traj.tool_call_count == 5


def test_max_steps_does_not_force_an_extra_answer(monkeypatch):
    _mock_sandbox_posts(monkeypatch)
    loop = make_loop([_sandbox_code('print("x")')] * 10)
    loop.model.generate_forced = MagicMock(return_value="B")
    question = "What color?\nOptions:\nA. red\nB. green\nC. blue\n"
    traj = loop.run(Image.new("RGB", (40, 40), (0, 0, 0)), question)
    assert traj.final_answer is None
    assert traj.stop_reason == "max_steps"
    loop.model.generate_forced.assert_not_called()


def test_last_step_does_not_append_a_forced_answer_instruction():
    cfg = AgentConfig(api_base="http://fake", model_name="m", max_steps=2)
    model = MagicMock()
    model.generate.side_effect = ["still thinking", "<answer>A</answer>"]
    tools = ToolRegistry()
    register_default_tools(tools)
    traj = AgentLoop(model, tools, cfg).run(
        Image.new("RGB", (10, 10)), "Choose one")

    assert traj.final_answer == "A"
    final_messages = model.generate.call_args_list[-1].args[0]
    assert len(final_messages) == 4
    assert "这是您的最后一步" not in str(final_messages)
    assert "must use <answer>" not in str(final_messages)


def test_no_tool_no_answer_continues():
    responses = ["just rambling, no tags", "<answer>X</answer>"]
    loop = make_loop(responses)
    img = Image.new("RGB", (40, 40))
    traj = loop.run(img, "q")
    assert traj.final_answer == "X"
    assert traj.tool_call_count == 0
    assert len(traj.steps) == 2


def test_model_error_stops_loop():
    cfg = AgentConfig(api_base="http://fake", model_name="m", max_steps=3)
    model = MagicMock()
    model.generate.side_effect = RuntimeError("boom")
    tools = ToolRegistry()
    register_default_tools(tools)
    loop = AgentLoop(model, tools, cfg)
    traj = loop.run(Image.new("RGB", (10, 10)), "q")
    assert traj.final_answer is None
    assert "model_error" in traj.stop_reason
    assert "boom" in traj.stop_reason


def test_timing_fields_populated():
    loop = make_loop(["<answer>Z</answer>"])
    traj = loop.run(Image.new("RGB", (10, 10)), "q")
    assert traj.total_time >= 0.0
    assert traj.total_inference_time >= 0.0
    assert traj.total_tool_time == 0.0


def test_bare_answer_requires_answer_tag():
    loop = make_loop(["A. right", "<answer>A</answer>"])
    traj = loop.run(Image.new("RGB", (10, 10)), "What direction?")
    assert traj.final_answer == "A"
    assert traj.stop_reason == "answer"
    assert len(traj.steps) == 2
    assert traj.steps[0].answer is None
    assert traj.steps[-1].answer == "A"


def test_multiple_input_images_reach_model_and_sandbox():
    loop = make_loop(["<answer>B</answer>"])
    images = [
        Image.new("RGB", (10, 10), (255, 0, 0)),
        Image.new("RGB", (12, 8), (0, 0, 255)),
    ]
    try:
        traj = loop.run(images, "Compare both images")
        assert traj.final_answer == "B"
        messages = loop.model.generate.call_args.args[0]
        image_parts = [
            part
            for part in messages[1]["content"]
            if part.get("type") == "image_url"
        ]
        assert len(image_parts) == 2
        assert len(loop.tools._input_image_paths) == 2
    finally:
        loop.tools.cleanup()


def test_fallback_does_not_steal_from_tool_call():
    responses = [
        '<tool_call>{"name": "broken"',
        '<answer>D</answer>',
    ]
    loop = make_loop(responses)
    traj = loop.run(Image.new("RGB", (10, 10)), "q")
    assert traj.final_answer == "D"
    assert traj.steps[0].answer is None


def test_empty_native_sandbox_call_never_reaches_dispatch_and_recovers():
    """Malformed native calls must not consume all remaining agent steps."""
    class NativeToolModel:
        def __init__(self):
            self.turn = 0
            self.calls = []
            self.last_tool_calls = []
            self.last_tool_call_ids = []
            self.last_assistant_msg = {"role": "assistant", "content": ""}

        def generate(self, messages, *, tools=None):
            self.calls.append(tools)
            self.turn += 1
            if self.turn <= 2:
                self.last_tool_calls = [ToolCall("sandbox", {})]
                self.last_tool_call_ids = [f"bad-{self.turn}"]
                self.last_assistant_msg = {
                    "role": "assistant", "content": "",
                    "tool_calls": [{
                        "id": f"bad-{self.turn}", "type": "function",
                        "function": {"name": "sandbox", "arguments": "{}"},
                    }],
                }
                return ""
            self.last_tool_calls = []
            self.last_tool_call_ids = []
            self.last_assistant_msg = {
                "role": "assistant", "content": "<answer>B</answer>"}
            return "<answer>B</answer>"

    cfg = AgentConfig(api_base="http://fake", model_name="m", max_steps=4)
    model = NativeToolModel()
    tools = ToolRegistry()
    tools.register("sandbox", lambda image, code: ("unused", []))
    dispatch = MagicMock(wraps=tools.dispatch)
    tools.dispatch = dispatch
    trajectory = AgentLoop(model, tools, cfg).run(
        Image.new("RGB", (10, 10)), "Choose A or B")

    assert trajectory.final_answer == "B"
    assert trajectory.stop_reason == "answer"
    assert trajectory.tool_call_count == 0
    assert dispatch.call_count == 0
    assert model.calls[-1] is None
    assert "requires a non-empty string argument named 'code'" in trajectory.steps[0].observation


def test_tools_always_operate_on_original_image(monkeypatch):
    _mock_sandbox_posts(monkeypatch)
    seen_sizes = []

    def spy_sandbox(image, code):
        seen_sizes.append(image.size)
        from spectool_agent.tools import ToolRegistry
        r = ToolRegistry()
        r.set_input_image(image)
        return r._sandbox(image, code)

    cfg = AgentConfig(api_base="http://fake", model_name="m", max_steps=4)
    tools = ToolRegistry()
    tools.register("sandbox", spy_sandbox)
    model = MagicMock()
    model.generate.side_effect = [
        _sandbox_code('crop = image_1.crop((0,0,50,40))\nplt.imshow(crop)\nplt.show()'),
        _sandbox_code('crop = image_1.crop((50,40,100,80))\nplt.imshow(crop)\nplt.show()'),
        '<answer>done</answer>',
    ]
    loop = AgentLoop(model, tools, cfg)
    img = Image.new("RGB", (100, 80), (0, 0, 0))
    traj = loop.run(img, "q")
    assert traj.final_answer == "done"
    assert seen_sizes == [(100, 80), (100, 80)]
