import base64
import os
from io import BytesIO
from unittest.mock import MagicMock

import pytest
from PIL import Image

from spectool_agent.tools import ToolRegistry, register_default_tools
from spectool_agent.types import ToolCall


def _image_b64(size=(50, 40), color=(255, 0, 0)):
    img = Image.new("RGB", size, color=color)
    buf = BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def _response(payload):
    resp = MagicMock()
    resp.raise_for_status.return_value = None
    resp.json.return_value = payload
    return resp


@pytest.fixture
def sample_image():
    return Image.new("RGB", (100, 80), color=(128, 64, 32))


@pytest.fixture
def registry(sample_image):
    r = ToolRegistry()
    r.set_input_image(sample_image)
    register_default_tools(r)
    return r


def test_sandbox_create_resized_image(registry, sample_image, monkeypatch):
    posts = [
        _response({"status": "success", "output": {"stdout": "", "stderr": "", "result": "", "images": []}}),
        _response({"status": "success", "output": {"stdout": "", "stderr": "", "result": "", "images": [_image_b64((50, 40))]}}),
    ]
    monkeypatch.setattr("spectool_agent.sandbox.requests.post", lambda *args, **kwargs: posts.pop(0))

    code = "plt.imshow(image_1.resize((50, 40)))\nplt.show()"
    obs, imgs = registry.dispatch(ToolCall("sandbox", {"code": code}), sample_image)
    assert "error" not in obs.lower()
    assert len(imgs) == 1
    assert imgs[0].size == (50, 40)


def test_sandbox_returns_stdout(registry, sample_image, monkeypatch):
    posts = [
        _response({"status": "success", "output": {"stdout": "", "stderr": "", "result": "", "images": []}}),
        _response({"status": "success", "output": {"stdout": "hello from sandbox", "stderr": "", "result": "", "images": []}}),
    ]
    monkeypatch.setattr("spectool_agent.sandbox.requests.post", lambda *args, **kwargs: posts.pop(0))

    obs, imgs = registry.dispatch(ToolCall("sandbox", {"code": 'print("hello from sandbox")'}), sample_image)
    assert "hello from sandbox" in obs
    assert imgs == []


def test_sandbox_error_on_syntax(registry, sample_image, monkeypatch):
    posts = [
        _response({"status": "success", "output": {"stdout": "", "stderr": "", "result": "", "images": []}}),
        _response({"status": "error", "output": {"stdout": "", "stderr": "SyntaxError: invalid syntax", "result": "", "images": []}}),
    ]
    monkeypatch.setattr("spectool_agent.sandbox.requests.post", lambda *args, **kwargs: posts.pop(0))

    obs, imgs = registry.dispatch(ToolCall("sandbox", {"code": "this is invalid python {{{"}), sample_image)
    assert "error" in obs.lower() or "sandbox error" in obs.lower()
    assert imgs == []


def test_sandbox_http_failure(registry, sample_image, monkeypatch):
    def _boom(*args, **kwargs):
        raise RuntimeError("network down")

    monkeypatch.setattr("spectool_agent.sandbox.requests.post", _boom)
    obs, imgs = registry.dispatch(ToolCall("sandbox", {"code": "print(1)"}), sample_image)
    assert "request to jupyter sandbox failed" in obs.lower()
    assert imgs == []


def test_sandbox_variable_persistence(registry, sample_image, monkeypatch):
    calls = []

    def _post(*args, **kwargs):
        calls.append(kwargs["json"])
        if len(calls) == 1:
            return _response({"status": "success", "output": {"stdout": "", "stderr": "", "result": "", "images": []}})
        if len(calls) == 2:
            return _response({"status": "success", "output": {"stdout": "", "stderr": "", "result": "", "images": []}})
        return _response({"status": "success", "output": {"stdout": "42", "stderr": "", "result": "", "images": []}})

    monkeypatch.setattr("spectool_agent.sandbox.requests.post", _post)

    registry.dispatch(ToolCall("sandbox", {"code": "x = 42"}), sample_image)
    obs2, _ = registry.dispatch(ToolCall("sandbox", {"code": "print(x)"}), sample_image)
    assert "42" in obs2
    assert calls[1]["session_id"] == calls[2]["session_id"]


def test_sandbox_shared_fs_uses_one_request_and_returns_path(
    registry, sample_image, monkeypatch
):
    calls = []
    output_path = os.path.join(registry._output_dir, "shared-result.png")
    Image.new("RGB", (32, 24), color=(0, 255, 0)).save(output_path)

    def _post(*args, **kwargs):
        calls.append(kwargs["json"])
        return _response({
            "status": "success",
            "output": {
                "stdout": (
                    "evidence\n"
                    f"__SPECTOOL_IMAGE_PATH__={output_path}\n"
                ),
                "stdout_raw": (
                    "evidence\n"
                    f"__SPECTOOL_IMAGE_PATH__={output_path}\n"
                ),
                "stderr": "",
                "result": "",
                "images": [],
            },
        })

    monkeypatch.setenv("SPECTOOL_SANDBOX_SHARED_FS", "1")
    monkeypatch.setattr("spectool_agent.sandbox.requests.post", _post)

    obs, imgs = registry.dispatch(
        ToolCall("sandbox", {"code": 'print("evidence")'}),
        sample_image,
    )

    assert len(calls) == 1
    assert registry._input_image_path in calls[0]["code"]
    assert "_img_base64" not in calls[0]["code"]
    assert "__SPECTOOL_IMAGE_PATH__" not in obs
    assert "evidence" in obs
    assert len(imgs) == 1
    assert imgs[0].size == (32, 24)


def test_cleanup_removes_sample_directory_and_clears_session(
    registry, monkeypatch
):
    output_dir = registry._output_dir
    context = {"session_id": "cleanup-test"}
    registry._previous_context = context
    generated_path = os.path.join(output_dir, "generated.png")
    Image.new("RGB", (8, 8)).save(generated_path)
    cleared = []
    monkeypatch.setattr(
        "spectool_agent.sandbox.clear_session",
        lambda value: cleared.append(value),
    )

    registry.cleanup()

    assert cleared == [context]
    assert not os.path.exists(output_dir)
    assert registry._output_dir == ""
    assert registry._input_image_path == ""
    assert registry._previous_context is None


def test_unknown_tool(registry, sample_image):
    obs, imgs = registry.dispatch(ToolCall("nonexistent", {}), sample_image)
    assert "unknown tool" in obs.lower()
    assert imgs == []


def test_bad_params_does_not_crash(registry, sample_image):
    obs, imgs = registry.dispatch(ToolCall("sandbox", {"wrong_key": "value"}), sample_image)
    assert "error" in obs.lower() or "sandbox error" in obs.lower()
    assert imgs == []
