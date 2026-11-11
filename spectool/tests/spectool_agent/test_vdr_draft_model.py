from types import SimpleNamespace

from spectool_agent.vdr.draft_model_vdr import DraftModelVDR


class _FakeCompletions:
    def __init__(self):
        self.kwargs = None

    def create(self, **kwargs):
        self.kwargs = kwargs
        message = SimpleNamespace(content='{"tool_name":"sandbox","params":{}}')
        return SimpleNamespace(choices=[SimpleNamespace(message=message)])


def test_vdr_tool_prediction_uses_vllm_structured_json():
    model = DraftModelVDR.__new__(DraftModelVDR)
    completions = _FakeCompletions()
    model._client = SimpleNamespace(
        chat=SimpleNamespace(completions=completions),
    )
    model._model_name = "draft"
    model._timeout = 10

    result = model._generate([{"role": "user", "content": "test"}])

    assert result == '{"tool_name":"sandbox","params":{}}'
    extra_body = completions.kwargs["extra_body"]
    assert "guided_json" not in extra_body
    schema = extra_body["structured_outputs"]["json"]
    assert schema["required"] == ["tool_name", "params"]
    assert schema["properties"]["tool_name"]["enum"] == [
        "sandbox",
        "web_search",
    ]
