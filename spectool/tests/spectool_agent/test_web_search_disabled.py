"""Default visual benchmarks must not expose or execute web_search."""

from unittest.mock import MagicMock

import pytest
from PIL import Image

from spectool_agent.prompts import get_prompt_set
from spectool_agent.tools import ToolRegistry, get_tool_schemas, register_default_tools
from spectool_agent.types import ToolCall


@pytest.mark.parametrize("dataset", [
    None, "VStarBench", "MMStar", "HRBench4K", "POPE", "MathVista_MINI",
])
def test_default_tools_and_prompts_disable_web_search(dataset):
    registry = ToolRegistry()
    schemas = register_default_tools(registry, dataset=dataset)

    assert registry.names() == ["sandbox"]
    assert [item["function"]["name"] for item in schemas] == ["sandbox"]
    assert schemas is get_tool_schemas(dataset=dataset)
    assert not hasattr(registry, "_web_search")

    prompts = get_prompt_set(dataset=dataset)
    for prompt in (prompts.system_prompt, prompts.draft_system_prompt):
        assert "web_search(query=" not in prompt
        assert "Web search is not available." in prompt


def test_fabricated_web_search_cannot_execute(monkeypatch):
    import spectool_agent.web_search as web_search

    search = MagicMock(side_effect=AssertionError("Network must not be reached"))
    monkeypatch.setattr(web_search, "search", search)
    registry = ToolRegistry()
    register_default_tools(registry, dataset="MMStar")
    call = ToolCall("web_search", {"query": "anything"})

    error = registry.validate_call(call)
    assert error is not None and "Unknown tool 'web_search'" in error
    _, images = registry.dispatch(call, Image.new("RGB", (8, 8)))
    assert images == []
    search.assert_not_called()


def test_clean_fallback_does_not_restore_web_search(monkeypatch):
    monkeypatch.setattr("spectool_agent.sandbox.clear_session", lambda context: None)
    registry = ToolRegistry()
    register_default_tools(registry, dataset="MMStar")
    child = registry.fork_clean_sandbox([Image.new("RGB", (8, 8))])
    try:
        assert child.names() == ["sandbox"]
        assert child.validate_call(ToolCall("web_search", {"query": "x"}))
    finally:
        child.cleanup()
