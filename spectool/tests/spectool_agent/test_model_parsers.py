import pytest
from PIL import Image

from spectool_agent.model import (_repair_json, parse_answer,
                                  parse_fallback_answer, parse_tool_call,
                                  pil_to_data_url)
from spectool_agent.types import ToolCall


def test_parse_answer_basic():
    assert parse_answer("Some text <answer>B</answer> trailing") == "B"


def test_parse_answer_missing():
    assert parse_answer("no tag here") is None


def test_parse_answer_multiline():
    text = "<think>...</think>\n<answer>The cat is\nbrown</answer>"
    assert parse_answer(text) == "The cat is\nbrown"


def test_parse_tool_call_basic():
    text = '<tool_call>{"name": "crop", "arguments": {"box": [0, 0, 0.5, 0.5]}}</tool_call>'
    tc = parse_tool_call(text)
    assert isinstance(tc, ToolCall)
    assert tc.tool_name == "crop"
    assert tc.params == {"box": [0, 0, 0.5, 0.5]}


def test_parse_tool_call_missing():
    assert parse_tool_call("no tool here") is None


def test_parse_tool_call_bad_json():
    text = '<tool_call>{not valid json}</tool_call>'
    assert parse_tool_call(text) is None


def test_parse_tool_call_missing_name():
    text = '<tool_call>{"arguments": {}}</tool_call>'
    assert parse_tool_call(text) is None


def test_pil_to_data_url():
    img = Image.new("RGB", (4, 4), color=(255, 0, 0))
    url = pil_to_data_url(img)
    assert url.startswith("data:image/jpeg;base64,")
    assert len(url) > len("data:image/jpeg;base64,")


@pytest.mark.parametrize("text", [
    "",
    "A",
    "A.",
    "A. right",
    " B ",
    "(C)",
    "D) yes",
    "The bird is a dove.\n(B)",
    "The answer: (C)",
    "just rambling, no letters",
    "AB",                                      # not a single letter
    "the answer is A",                         # leading prose, no parens
    '<tool_call>{"name": "x"}</tool_call>',    # has a tool call
    'a',                                       # lowercase
    # Multi-line without answer pattern at end
    "Options are: (A) dog, (B) cat. The dog is brown.",
    # Has <tool_call> but also has (X) at end — tool_call takes priority
    '<tool_call>{"name": "x"}</tool_call>\n(A)',
])
def test_parse_fallback_answer_rejects(text):
    assert parse_fallback_answer(text) is None


class TestRepairJson:
    def test_repair_missing_brace(self):
        malformed = '{"name": "sandbox", "arguments": {"code": "print(1)"}'
        repaired = _repair_json(malformed)
        import json
        data = json.loads(repaired)
        assert data["name"] == "sandbox"
        assert data["arguments"]["code"] == "print(1)"

    def test_repair_missing_brace_with_newlines(self):
        malformed = ('{"name": "sandbox", "arguments": {"code": "import cv2\\n'
                     'import numpy as np\\nplt.show()"}')
        repaired = _repair_json(malformed)
        import json
        data = json.loads(repaired, strict=False)
        assert data["name"] == "sandbox"
        assert "plt.show()" in data["arguments"]["code"]

    def test_repair_missing_bracket(self):
        malformed = '{"name": "crop", "arguments": {"box": [0, 0, 0.5, 0.5'
        repaired = _repair_json(malformed)
        import json
        data = json.loads(repaired)
        assert data["arguments"]["box"] == [0, 0, 0.5, 0.5]

    def test_repair_noop_on_valid_json(self):
        valid = '{"name": "sandbox", "arguments": {"code": "x=1"}}'
        assert _repair_json(valid) == valid

    def test_parse_tool_call_with_repair(self):
        text = ('\n\n<tool_call>{"name": "sandbox", "arguments": '
                '{"code": "import cv2\\nprint(1)"}</tool_call>')
        tc = parse_tool_call(text)
        assert isinstance(tc, ToolCall)
        assert tc.tool_name == "sandbox"
        assert "print(1)" in tc.params["code"]

    def test_parse_tool_call_with_code_and_newlines(self):
        text = (
            '\n\n<tool_call>{"name": "sandbox", "arguments": '
            '{"code": "import matplotlib.pyplot as plt\\n'
            'import cv2\\n'
            'import numpy as np\\n\\n'
            'image = np.array(image_1)\\n'
            'h, w, _ = image.shape\\n'
            'y1, y2 = int(h * 0.58), int(h * 0.70)\\n'
            'x1, x2 = int(w * 0.29), int(w * 0.32)\\n'
            'poster_roi = image[y1:y2, x1:x2]\\n'
            'plt.figure(figsize=(15, 10))\\n'
            'plt.imshow(poster_roi)\\n'
            'plt.axis(\'off\')\\n'
            'plt.tight_layout()\\n'
            'plt.show()"}</tool_call>'
        )
        tc = parse_tool_call(text)
        assert isinstance(tc, ToolCall)
        assert tc.tool_name == "sandbox"
        assert "poster_roi" in tc.params["code"]

    def test_parse_tool_call_still_rejects_bad_json(self):
        text = '<tool_call>{not valid json}</tool_call>'
        assert parse_tool_call(text) is None
