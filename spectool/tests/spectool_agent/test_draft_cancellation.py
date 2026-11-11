import asyncio
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from unittest.mock import MagicMock

from spectool_agent.draft_model import DraftModel
from spectool_agent.model import RequestCancellation, RequestCancelled
from spectool_agent.speculative_agent import CancellableTask, SpeculativeAgentLoop


def _tool_response(marker: str):
    function = SimpleNamespace(name=f"tool_{marker}", arguments="{}")
    tool_call = SimpleNamespace(id=f"call_{marker}", function=function)
    message = SimpleNamespace(
        content=marker,
        tool_calls=[tool_call],
        model_dump=lambda: {},
    )
    return SimpleNamespace(choices=[SimpleNamespace(
        finish_reason="length" if marker == "A" else "stop",
        message=message,
    )])


def test_draft_generation_metadata_is_thread_local(monkeypatch):
    def create(**kwargs):
        marker = kwargs["messages"][-1]["marker"]
        return _tool_response(marker)

    fake_client = MagicMock()
    fake_client.chat.completions.create.side_effect = create
    import spectool_agent.draft_model as mod
    monkeypatch.setattr(mod, "OpenAI", lambda **_: fake_client)

    model = DraftModel(api_base="http://fake/v1", model_name="fake-draft")
    generated = threading.Barrier(2)
    observed = {}

    def worker(marker):
        output = model.generate_full(
            [{"marker": marker}], tools=[{"type": "function"}])
        generated.wait()
        observed[marker] = (
            output,
            model.last_truncated,
            model.last_tool_calls[0].tool_name,
            model.last_tool_call_ids[0],
            model.last_assistant_msg["content"],
        )

    workers = [threading.Thread(target=worker, args=(marker,))
               for marker in ("A", "B")]
    for worker_thread in workers:
        worker_thread.start()
    for worker_thread in workers:
        worker_thread.join(timeout=1.0)

    assert all(not worker_thread.is_alive() for worker_thread in workers)
    assert observed == {
        "A": ("A", True, "tool_A", "call_A", "A"),
        "B": ("B", False, "tool_B", "call_B", "B"),
    }


class _BlockingAsyncCompletions:
    def __init__(self):
        self.entered = threading.Event()
        self.cancelled = threading.Event()
        self.kwargs = None

    async def create(self, **kwargs):
        self.kwargs = kwargs
        self.entered.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            self.cancelled.set()
            raise


class _BlockingAsyncClient:
    def __init__(self):
        self.completions = _BlockingAsyncCompletions()
        self.chat = SimpleNamespace(completions=self.completions)
        self.closed = threading.Event()

    async def close(self):
        self.closed.set()


def test_draft_request_can_be_cancelled_before_headers(monkeypatch):
    async_client = _BlockingAsyncClient()
    import spectool_agent.draft_model as mod
    monkeypatch.setattr(mod, "OpenAI", lambda **_: MagicMock())
    monkeypatch.setattr(mod, "AsyncOpenAI", lambda **_: async_client)

    model = DraftModel(api_base="http://fake/v1", model_name="fake-draft")
    request_cancel = RequestCancellation()
    errors = []

    def generate():
        try:
            model.generate_full(
                [{"role": "user", "content": "explore"}],
                cancel_token=request_cancel)
        except Exception as exc:
            errors.append(exc)

    worker = threading.Thread(target=generate)
    worker.start()
    assert async_client.completions.entered.wait(0.5)
    request_cancel.cancel()
    worker.join(timeout=1.0)

    assert not worker.is_alive()
    assert async_client.completions.cancelled.is_set()
    assert async_client.closed.is_set()
    assert len(errors) == 1
    assert isinstance(errors[0], RequestCancelled)
    assert async_client.completions.kwargs["model"] == "fake-draft"


def test_stop_draft_propagates_request_cancel_and_waits_for_exit(monkeypatch):
    monkeypatch.setenv("SPECTOOL_DRAFT_CANCEL_WAIT_SECONDS", "0.5")
    cooperative_cancel = CancellableTask()
    request_cancel = RequestCancellation()
    started = threading.Event()

    def run_request():
        assert cooperative_cancel.register_request(request_cancel)
        started.set()
        try:
            while not request_cancel.cancelled:
                time.sleep(0.005)
        finally:
            cooperative_cancel.unregister_request(request_cancel)

    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(run_request)
        assert started.wait(0.5)
        SpeculativeAgentLoop._stop_draft(future, cooperative_cancel)
        assert request_cancel.cancelled
        assert future.done()
