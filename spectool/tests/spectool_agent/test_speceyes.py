import math
import threading
from types import SimpleNamespace

import pytest

from PIL import Image

from spectool_agent.model import _scaled_image_size
from spectool_agent.speceyes import (
    DIRECT_SYSTEM_PROMPT,
    ROUTER_SYSTEM_PROMPT,
    SpecEyesBatchOrchestrator,
    SpecEyesOrchestrator,
    SpecEyesRequest,
    answer_separability_from_logprobs,
)
from spectool_agent.types import AgentConfig, Trajectory


def _token(*values):
    return SimpleNamespace(
        logprob=values[0],
        top_logprobs=[SimpleNamespace(logprob=value) for value in values],
    )


def test_separability_uses_logprob_invariance():
    # Adding a token-wise normalisation constant does not change the z-score.
    base = [_token(0.0, -2.0, -3.0, -4.0), _token(-0.1, -1.5, -2.0, -3.0)]
    shifted = [_token(-5.0, -7.0, -8.0, -9.0), _token(-9.1, -10.5, -11.0, -12.0)]
    base_score, _ = answer_separability_from_logprobs(base, top_k=4)
    shifted_score, _ = answer_separability_from_logprobs(shifted, top_k=4)
    assert base_score == pytest.approx(shifted_score)


def test_separability_refuses_incomplete_logprob_distribution():
    score, details = answer_separability_from_logprobs(
        [SimpleNamespace(logprob=-0.1, top_logprobs=[])], top_k=64)
    assert score is None
    assert details["reason"] == "incomplete_nonterminal_top_logprobs"


def test_separability_rejects_unknown_aggregation_mode():
    with pytest.raises(ValueError, match="Unsupported"):
        answer_separability_from_logprobs([_token(0.0, -1.0)], mode="unknown")


def test_separability_ignores_terminal_token_without_candidates():
    score, details = answer_separability_from_logprobs([
        _token(0.0, -2.0),
        SimpleNamespace(token="<|im_end|>", logprob=-0.1, top_logprobs=[]),
    ], top_k=2)
    assert score is not None
    assert details["terminal_tokens"] == 1


def test_separability_refuses_nonterminal_token_without_full_topk():
    score, details = answer_separability_from_logprobs([
        _token(0.0, -2.0),
        SimpleNamespace(token="A", logprob=-0.1, top_logprobs=[]),
    ], top_k=2)
    assert score is None
    assert details["reason"] == "incomplete_nonterminal_top_logprobs"
    assert details["incomplete_nonterminal_tokens"] == 1


def test_log_mode_matches_official_mean_greedy_probability():
    tokens = [
        SimpleNamespace(
            token="A", logprob=-0.2,
            top_logprobs=[SimpleNamespace(token="A", logprob=-0.2)],
        ),
        SimpleNamespace(
            token="B", logprob=-0.8,
            top_logprobs=[SimpleNamespace(token="B", logprob=-0.8)],
        ),
        # An EOS-like marker is excluded, like official actual_len handling.
        SimpleNamespace(token="<|im_end|>", logprob=-0.1, top_logprobs=[]),
    ]
    score, details = answer_separability_from_logprobs(tokens, mode="log")
    assert score == pytest.approx(math.exp(-0.5))
    assert details["terminal_tokens"] == 1


def test_speceyes_resize_matches_official_smart_resize_defaults():
    # 100 rounds to 4*28, while a tiny image is upscaled to min_pixels.
    assert _scaled_image_size(
        100, 100, max_size=0, max_pixels=12_845_056,
        min_pixels=3_136, factor=28, resize_policy="speceyes",
    ) == (112, 112)
    assert _scaled_image_size(
        2, 2, max_size=0, max_pixels=12_845_056,
        min_pixels=3_136, factor=28, resize_policy="speceyes",
    ) == (56, 56)


def _full_distribution():
    return SimpleNamespace(
        token="winner",
        logprob=0.0,
        top_logprobs=[
            SimpleNamespace(token="winner", logprob=0.0),
            *[
                SimpleNamespace(token=f"candidate_{index}", logprob=-20.0)
                for index in range(63)
            ],
        ],
    )


class _StubModel:
    def __init__(self, response):
        self.response = response
        self.calls = []

    def generate_text(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        return self.response


class _NoFallbackSpecEyes(SpecEyesOrchestrator):
    def _main_agent(self, *args, **kwargs):
        raise AssertionError("high-confidence direct answer must not run fallback")


def test_orchestrator_accepts_high_confidence_direct_draft():
    cfg = AgentConfig(api_base="http://unused/v1", model_name="main")
    main = _StubModel({"text": "no", "content_logprobs": []})
    draft = _StubModel({
        # Official phase II requests an untagged final answer.
        "text": "A",
        "content_logprobs": [_full_distribution()],
    })
    result = _NoFallbackSpecEyes(
        main, draft, cfg, confidence_threshold=0.98, confidence_top_k=64,
    ).run(Image.new("RGB", (2, 2)), "Which option is correct?\nA. yes\nB. no")
    assert result.final_answer == "A"
    assert result.decision == "draft_accepted"
    assert main.calls[0][0][0][0]["content"] == ROUTER_SYSTEM_PROMPT
    assert main.calls[0][1]["max_tokens"] == 10
    assert draft.calls[0][0][0][0]["content"] == DIRECT_SYSTEM_PROMPT
    assert draft.calls[0][0][0][1]["content"][-1]["text"].endswith("\nAnswer:")
    assert draft.calls[0][1]["max_tokens"] == 20


class _BatchMainModel:
    def __init__(self):
        self.router_calls = 0
        self.lock = threading.Lock()

    def generate_text(self, messages, **kwargs):
        assert messages[0]["content"] == ROUTER_SYSTEM_PROMPT
        question = messages[1]["content"][-1]["text"]
        with self.lock:
            self.router_calls += 1
        return {"text": "yes" if "needs tool" in question else "no"}


class _BatchDraftModel:
    def __init__(self, main, expected_router_calls):
        self.main = main
        self.expected_router_calls = expected_router_calls
        self.calls = 0
        self.lock = threading.Lock()

    def generate_text(self, messages, **kwargs):
        assert messages[0]["content"] == DIRECT_SYSTEM_PROMPT
        # The router executor must fully drain before Phase II starts.
        assert self.main.router_calls == self.expected_router_calls
        question = messages[1]["content"][-1]["text"]
        with self.lock:
            self.calls += 1
        answer = "A" if "first" in question else "C"
        return {"text": answer, "content_logprobs": [_full_distribution()]}


@pytest.mark.parametrize("concurrency", [1, 4])
def test_batch_orchestrator_preserves_official_phase_barriers_and_order(concurrency):
    cfg = AgentConfig(api_base="http://unused/v1", model_name="main")
    main = _BatchMainModel()
    draft = _BatchDraftModel(main, expected_router_calls=3)
    batch = SpecEyesBatchOrchestrator(
        main, draft, cfg, confidence_threshold=0.98, confidence_top_k=64,
        fallback_concurrency=concurrency)
    fallback_calls = []

    def fallback(images, question, **kwargs):
        # Phase III starts only after every direct candidate completed Phase II.
        assert draft.calls == 2
        fallback_calls.append(kwargs["sample_id"])
        return SimpleNamespace(final_answer="Tool answer")

    batch._single._main_agent = fallback
    results = batch.run([
        SpecEyesRequest(Image.new("RGB", (2, 2)), "first direct", sample_id="one"),
        SpecEyesRequest(Image.new("RGB", (2, 2)), "needs tool", sample_id="two"),
        SpecEyesRequest(Image.new("RGB", (2, 2)), "third direct", sample_id="three"),
    ])

    assert [result.final_answer for result in results] == ["A", "Tool answer", "C"]
    assert [result.decision for result in results] == [
        "draft_accepted", "main_sandbox", "draft_accepted"]
    assert main.router_calls == 3
    assert draft.calls == 2
    assert fallback_calls == ["two"]
    assert [result.batch_index for result in results] == [0, 1, 2]
    assert all(result.batch_size == 3 for result in results)
    assert all(result.fallback_concurrency == concurrency for result in results)


def test_parallel_fallback_has_four_workers_and_isolated_tools(monkeypatch):
    import spectool_agent.speceyes as module

    cfg = AgentConfig(api_base="http://unused/v1", model_name="main")
    main = _BatchMainModel()
    draft = _BatchDraftModel(main, expected_router_calls=6)
    batch = SpecEyesBatchOrchestrator(main, draft, cfg, fallback_concurrency=4)
    barrier = threading.Barrier(4)
    lock = threading.Lock()
    tools_created = []
    active = 0
    peak = 0

    class IsolatedTools:
        def __init__(self):
            self.cleaned = False
            self.sample = None
            with lock:
                tools_created.append(self)

        def register(self, *args):
            pass

        def _sandbox(self):
            pass

        def cleanup(self):
            self.cleaned = True

    class IsolatedAgent:
        def __init__(self, model, draft_model, tools, config, **kwargs):
            self.tools = tools

        def run(self, images, question, sample_id, fallback_only=False, **kwargs):
            nonlocal active, peak
            assert fallback_only
            assert main.router_calls == 6
            assert draft.calls == 0
            index = int(sample_id)
            assert images[0].getpixel((0, 0))[0] == index
            assert self.tools.sample is None
            self.tools.sample = sample_id
            with lock:
                active += 1
                peak = max(peak, active)
            try:
                if index < 4:
                    barrier.wait(timeout=5)
                if index == 4:
                    raise RuntimeError("one sample failed")
                assert self.tools.sample == sample_id
                return Trajectory(sample_id=sample_id, final_answer=f"answer-{sample_id}")
            finally:
                with lock:
                    active -= 1

        def shutdown(self):
            pass

    monkeypatch.setattr(module, "ToolRegistry", IsolatedTools)
    monkeypatch.setattr(
        module, "register_default_tools", lambda registry, **kwargs: ["sandbox", "web_search"])
    monkeypatch.setattr(module, "SpeculativeAgentLoop", IsolatedAgent)
    results = batch.run([
        SpecEyesRequest(Image.new("RGB", (2, 2), (index, 0, 0)),
                        "needs tool", sample_id=str(index))
        for index in range(6)
    ])
    assert peak == 4
    assert [result.final_answer for result in results] == [
        "answer-0", "answer-1", "answer-2", "answer-3", None, "answer-5"]
    assert [result.batch_index for result in results] == list(range(6))
    assert len(tools_created) == len({id(tool) for tool in tools_created}) == 6
    assert all(tool.cleaned for tool in tools_created)
    assert "one sample failed" in results[4].error
    assert all(not result.error for index, result in enumerate(results) if index != 4)
    assert results[0].metadata()["speceyes_fallback_concurrency"] == 4
    assert results[0].metadata()["speceyes_batch_execution"].endswith("parallel_fallback")


def test_fallback_concurrency_defaults_and_validation(monkeypatch):
    cfg = AgentConfig(api_base="http://unused/v1", model_name="main")
    monkeypatch.delenv("SPECEYE_FALLBACK_NPROC", raising=False)
    assert SpecEyesBatchOrchestrator(None, None, cfg).fallback_concurrency == 1
    monkeypatch.setenv("SPECEYE_FALLBACK_NPROC", "4")
    assert SpecEyesBatchOrchestrator(None, None, cfg).fallback_concurrency == 4
    with pytest.raises(ValueError, match="at least 1"):
        SpecEyesBatchOrchestrator(None, None, cfg, fallback_concurrency=0)
