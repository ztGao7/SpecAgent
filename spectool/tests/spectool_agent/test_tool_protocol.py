import json

from spectool_agent.model import (
    build_assistant_tool_call_message,
    parse_tool_call,
)


def test_parse_qwen3_xml_tool_call():
    response = """Need a closer view.
<tool_call>
<function=sandbox>
<parameter=code>
crop = image_1.crop((10, 20, 30, 40))
print(crop.size)
</parameter>
</function>
</tool_call>"""

    call = parse_tool_call(response)

    assert call is not None
    assert call.tool_name == "sandbox"
    assert call.params == {
        "code": "crop = image_1.crop((10, 20, 30, 40))\nprint(crop.size)"
    }


def test_parse_legacy_json_tool_call():
    response = (
        '<tool_call>{"name":"web_search","arguments":'
        '{"query":"Qwen tool format"}}</tool_call>'
    )

    call = parse_tool_call(response)

    assert call is not None
    assert call.tool_name == "web_search"
    assert call.params == {"query": "Qwen tool format"}


def test_text_fallback_is_reinserted_as_structured_tool_call():
    response = (
        'Inspect the image.\n'
        '<tool_call>{"name":"sandbox","arguments":{"code":"print(1)"}}'
        '</tool_call>'
    )
    call = parse_tool_call(response)
    assert call is not None

    message = build_assistant_tool_call_message(
        response, [call], ["call_fallback_0_0"])

    assert message["role"] == "assistant"
    assert message["content"] == "Inspect the image."
    assert message["tool_calls"][0]["id"] == "call_fallback_0_0"
    function = message["tool_calls"][0]["function"]
    assert function["name"] == "sandbox"
    assert json.loads(function["arguments"]) == {"code": "print(1)"}
