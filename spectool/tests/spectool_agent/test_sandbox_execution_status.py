import pytest
from PIL import Image

from spectool_agent import sandbox
from spectool_agent.speculative_agent import SpeculativeAgentLoop
from spectool_agent.tools import ToolRegistry, register_default_tools
from spectool_agent.types import PoolEntry, ToolCall


JUPYTER_MISSING_FILE = (
    "---------------------------------------------------------------------------\n"
    "FileNotFoundError                         Traceback (most recent call last)\n"
    "Cell In[1], line 63\n"
    "---> 63 img = Image.open('image_1')\n"
    "FileNotFoundError: [Errno 2] No such file or directory: 'image_1'"
)
JUPYTER_ATTRIBUTE_ERROR = (
    "AttributeError                            Traceback (most recent call last)\n"
    "Cell In[2], line 63\n"
    "AttributeError: 'PngImageFile' object has no attribute 'read'"
)


def _payload(stderr="", stdout="", **extra):
    return {
        "status": "success", "output": {
            "stdout": stdout, "stderr": stderr, "result": "", "images": [],
        }, **extra,
    }


@pytest.mark.parametrize("stderr", [
    JUPYTER_MISSING_FILE,
    JUPYTER_ATTRIBUTE_ERROR,
    "Traceback (most recent call last):\n  File '<cell>', line 1\nValueError: bad input",
    "  Cell In[1], line 3\n    invalid {{\nSyntaxError: invalid syntax",
    "\x1b[31mNameError\x1b[0m: name 'img' is not defined",
    "numpy.linalg.LinAlgError: Singular matrix",
    "KeyboardInterrupt",
    "SystemExit: 1",
    "CustomFailure Traceback (most recent call last)\nCustomFailure: missing data",
    "Timed out waiting for Jupyter kernel output.",
    "Timed out waiting for Jupyter kernel execution reply.",
])
def test_transport_success_does_not_hide_execution_error(tmp_path, stderr):
    context = {"session_id": "existing-kernel", "initialized": True}
    result = sandbox._response_to_result(
        _payload(stderr=stderr, stdout="partial output"), str(tmp_path), context)
    assert result.success is False
    assert result.error
    assert "partial output" in result.output
    assert result.execution_context == context
    entry = PoolEntry(step_idx=0, tool_observation="stderr:\n" + stderr)
    assert SpeculativeAgentLoop._tool_entry_has_error(entry)


@pytest.mark.parametrize("stderr", [
    "",
    "UserWarning: image metadata was ignored",
    "/tmp/plot.py:1: RuntimeWarning: invalid value encountered in divide",
    "Progress: 2/3\nCompleted.",
    "The documentation mentions FileNotFoundError and Traceback examples.",
])
def test_warnings_and_stderr_messages_are_not_execution_errors(tmp_path, stderr):
    result = sandbox._response_to_result(
        _payload(stderr=stderr, stdout="actual evidence"), str(tmp_path), {})
    assert result.success is True
    assert result.error is None
    entry = PoolEntry(step_idx=0, tool_observation=result.output)
    assert not SpeculativeAgentLoop._tool_entry_has_error(entry)


@pytest.mark.parametrize("stderr", ["", "UserWarning: note"])
def test_printed_exception_text_is_not_a_failed_execution(tmp_path, stderr):
    result = sandbox._response_to_result(
        _payload(stdout="FileNotFoundError: quoted text", stderr=stderr),
        str(tmp_path), {})
    assert result.success is True
    assert not SpeculativeAgentLoop._tool_entry_has_error(
        PoolEntry(step_idx=0, tool_observation=result.output))


@pytest.mark.parametrize("response", [
    _payload(status="error", stderr="worker unavailable"),
    _payload(error="explicit server failure"),
])
def test_explicit_server_failure_still_fails(tmp_path, response):
    result = sandbox._response_to_result(response, str(tmp_path), {})
    assert result.success is False
    assert result.error


def test_registry_marks_kernel_error_and_keeps_recovery_session(monkeypatch):
    monkeypatch.setenv("SPECTOOL_SANDBOX_SHARED_FS", "1")
    monkeypatch.setattr(sandbox, "clear_session", lambda context: None)
    responses = iter([
        _payload(stderr=JUPYTER_MISSING_FILE),
        _payload(stdout="image evidence recovered"),
    ])
    session_ids = []

    def post(session_id, code, timeout):
        session_ids.append(session_id)
        return next(responses), None

    monkeypatch.setattr(sandbox, "_post_code", post)
    registry = ToolRegistry()
    image = Image.new("RGB", (2, 2))
    registry.set_input_image(image)
    register_default_tools(registry)
    try:
        error, images = registry.dispatch(
            ToolCall("sandbox", {"code": "bad_image_access()"}), image)
        assert "FileNotFoundError" in error
        assert images == []
        assert SpeculativeAgentLoop._tool_entry_has_error(
            PoolEntry(step_idx=0, tool_observation=error))
        recovered, _ = registry.dispatch(
            ToolCall("sandbox", {"code": "print(image_1.size)"}), image)
        assert recovered == "image evidence recovered"
        assert len(session_ids) == 2 and session_ids[0] == session_ids[1]
    finally:
        registry.cleanup()
