"""SpecEyes orchestration on top of Spectool's vLLM and sandbox stack.

The original SpecEyes decision tree is retained, while its local Transformers
models and in-process ``exec`` sandbox are deliberately not reused:

1. the 27B model classifies whether visual-tool evidence is required;
2. for direct-answer tasks, 9B answers once and returns top-K logprobs;
3. the original token-separability confidence gate accepts only high-confidence
   9B answers; and
4. every other sample enters Spectool's clean-context Main fallback, using the
   same tool registration, context construction, budgets, and finalization
   policy as the ordinary Spectool fallback-only baseline.

This module owns no sandbox state.  ``ToolRegistry`` remains the sole owner of
per-sample sandbox sessions and temporary input/output images.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
import math
import os
import time
from typing import Any, Optional

from PIL import Image

from .model import VLLMModel, parse_answer, pil_to_data_url
from .prompts import PromptSet, USER_PROMPT_TEMPLATE, get_prompt_set
from .speculative_agent import SpeculativeAgentLoop
from .tools import ToolRegistry, register_default_tools
from .types import AgentConfig, Trajectory


# These two Phase-I/Phase-II prompts are intentionally byte-for-byte the
# prompts used by the official SpecEyes repository.  Keeping the yes/no
# wording (rather than translating it to an internal route label) matters:
# official routing sends only a response beginning with "no" to the small
# model, and treats every other response conservatively as a tool route.
ROUTER_SYSTEM_PROMPT = """You decide based on answer reliability.

Answer "yes" if using a tool would significantly improve correctness or confidence of the answer.

Answer "no" if tools would add little or no value.

Output only "yes" or "no".
"""
ROUTER_USER_SUFFIX = "\nWould tools significantly improve answer reliability? yes or no."

DIRECT_SYSTEM_PROMPT = (
    "You are a helpful assistant. Output only the final answer itself. Do not "
    "include explanations, reasoning, labels, or any extra text."
)
DIRECT_USER_SUFFIX = "\nAnswer:"

@dataclass
class SpecEyesResult:
    """One SpecEyes sample outcome plus stage-level audit metadata."""

    final_answer: Optional[str]
    route: str
    decision: str
    router_raw: str = ""
    draft_raw: str = ""
    confidence: Optional[float] = None
    confidence_details: dict[str, Any] = field(default_factory=dict)
    main_trajectory: Optional[Trajectory] = None
    error: str = ""
    router_time: float = 0.0
    draft_time: float = 0.0
    fallback_time: float = 0.0
    total_time: float = 0.0
    batch_size: int = 1
    batch_index: Optional[int] = None
    batch_execution: str = "single_request"
    fallback_concurrency: int = 1

    def metadata(self) -> dict[str, Any]:
        """Return JSON-ready data while retaining Spectool trajectory metrics."""
        trajectory = self.main_trajectory
        return {
            "speceyes_route": self.route,
            "speceyes_decision": self.decision,
            "speceyes_router_raw": self.router_raw,
            "speceyes_draft_raw": self.draft_raw,
            "speceyes_confidence": self.confidence,
            "speceyes_confidence_details": self.confidence_details,
            "speceyes_router_time": self.router_time,
            "speceyes_draft_time": self.draft_time,
            "speceyes_fallback_time": self.fallback_time,
            "speceyes_total_time": self.total_time,
            "speceyes_batch_size": self.batch_size,
            "speceyes_batch_index": self.batch_index,
            "speceyes_batch_execution": self.batch_execution,
            "speceyes_fallback_concurrency": self.fallback_concurrency,
            "speceyes_error": self.error,
            "tool_call_count": trajectory.tool_call_count if trajectory else 0,
            "stop_reason": trajectory.stop_reason if trajectory else self.decision,
            "total_inference_time": (
                trajectory.total_inference_time if trajectory else self.router_time + self.draft_time
            ),
            "total_tool_time": trajectory.total_tool_time if trajectory else 0.0,
            "trajectory": trajectory,
        }


@dataclass
class SpecEyesRequest:
    """One independently answerable item in a SpecEyes batch.

    A batch is an execution unit, never a multi-turn conversation: each item
    retains its own images, question, sample id, and sandbox session.
    """

    image: Image.Image | list[Image.Image]
    question: str
    sample_id: str = ""
    extra_prompt: str = ""
    prompt_set: Optional[PromptSet] = None


@dataclass
class _BatchState:
    """Internal, fully prepared state for one item in a batch."""

    request: SpecEyesRequest
    batch_index: int
    images: list[Image.Image]
    prompt_set: PromptSet
    request_question: str
    started: float
    route: str = "tool_required"
    router_raw: str = ""
    router_time: float = 0.0
    draft_raw: str = ""
    draft_time: float = 0.0
    confidence: Optional[float] = None
    confidence_details: dict[str, Any] = field(default_factory=dict)


def _env_int(name: str, default: int, minimum: int = 1) -> int:
    try:
        return max(minimum, int(os.environ.get(name, str(default))))
    except (TypeError, ValueError):
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, str(default)))
    except (TypeError, ValueError):
        return default


def _token_top_logprobs(token: Any, top_k: int) -> list[float]:
    """Extract one token's top-K logprobs from OpenAI-compatible objects.

    vLLM returns logprobs rather than raw logits.  Softmax normalization differs
    from logits by a per-token constant, which cancels out in the original
    SpecEyes z-score; using top logprobs is therefore mathematically equivalent
    for this metric.
    """
    values: list[float] = []
    seen_tokens: set[str] = set()
    current = getattr(token, "logprob", None)
    if current is not None:
        values.append(float(current))
        current_text = getattr(token, "token", None)
        if current_text is not None:
            seen_tokens.add(str(current_text))
    for candidate in getattr(token, "top_logprobs", None) or []:
        candidate_text = getattr(candidate, "token", None)
        if candidate_text is not None and str(candidate_text) in seen_tokens:
            continue
        value = getattr(candidate, "logprob", None)
        if value is not None:
            values.append(float(value))
            if candidate_text is not None:
                seen_tokens.add(str(candidate_text))
    values.sort(reverse=True)
    return values[:top_k]


def _is_official_terminal_token(token: Any) -> bool:
    """Return whether a vLLM logprob entry is Qwen's EOS or PAD token.

    Official SpecEyes trims ``output.logits`` by the tokenizer IDs for EOS and
    PAD *before* it scores the answer.  OpenAI-compatible logprob responses do
    not expose token IDs, but vLLM preserves Qwen's special-token text.  Do
    not treat an empty token or a short top-K list as terminal: either can also
    occur for an ordinary generated token, in which case accepting a score
    would no longer reproduce the original full-logit calculation.
    """
    return getattr(token, "token", None) in {"<|im_end|>", "<|endoftext|>"}


def answer_separability_from_logprobs(
    content_logprobs: list[Any], *, top_k: int = 64,
    mode: str = "min", alpha: float = 0.6,
) -> tuple[Optional[float], dict[str, Any]]:
    """Reproduce SpecEyes' separability score from vLLM top-logprobs.

    The original code aggregates greedy-token z-scores using ``min`` by
    default, then applies ``sigmoid(alpha * score)``.  The official path first
    removes only EOS/PAD entries, then has full vocabulary logits for every
    remaining token.  Accordingly, a non-terminal entry whose vLLM top-K list
    is incomplete is not treated as confidence: callers must fall back.
    """
    if top_k < 1:
        raise ValueError("top_k must be at least 1")
    if mode not in {"min", "mean", "bottom20", "log"}:
        raise ValueError(f"Unsupported SPECEYE_CONFIDENCE_MODE: {mode}")
    if mode != "log" and top_k < 2:
        raise ValueError("top_k must be at least 2 for separability scoring")
    scores: list[float] = []
    greedy_logprobs: list[float] = []
    terminal_entries = 0
    incomplete_nonterminal = 0
    for token in content_logprobs or []:
        if _is_official_terminal_token(token):
            terminal_entries += 1
            continue
        values = _token_top_logprobs(token, top_k)
        # Official ``mode=log`` is exp(mean(log(max_prob))) over content
        # tokens.  At temperature zero the generated token is the max-prob
        # token, and its vLLM logprob is exactly log(max_prob).
        if mode == "log":
            current = getattr(token, "logprob", None)
            # At temperature zero the emitted token is top-1, so its own
            # logprob is sufficient for the official max-probability metric.
            if current is None:
                incomplete_nonterminal += 1
                continue
            greedy_logprobs.append(float(current))
            continue
        # The original implementation uses torch.topk(logits, K) for every
        # generated token.  Anything less than K would silently change the
        # score, so route it to the conservative 27B fallback instead.
        if len(values) < top_k:
            incomplete_nonterminal += 1
            continue
        winner = values[0]
        competitors = values[1:]
        mean = sum(competitors) / len(competitors)
        variance = sum((value - mean) ** 2 for value in competitors) / len(competitors)
        scores.append((winner - mean) / (math.sqrt(variance) + 1e-6))

    details: dict[str, Any] = {
        "top_k": top_k,
        "mode": mode,
        "alpha": alpha,
        "scored_tokens": len(greedy_logprobs) if mode == "log" else len(scores),
        "terminal_tokens": terminal_entries,
        "incomplete_nonterminal_tokens": incomplete_nonterminal,
    }
    # The Transformers implementation always has full logits for every
    # non-terminal entry.  A short vLLM list is therefore an observation gap,
    # not evidence for a high-confidence answer.
    if incomplete_nonterminal:
        details["reason"] = "incomplete_nonterminal_top_logprobs"
        return None, details
    if mode == "log":
        if not greedy_logprobs:
            details["reason"] = "missing_top_logprobs"
            return None, details
        score = math.exp(sum(greedy_logprobs) / len(greedy_logprobs))
        details["aggregate_logprob"] = sum(greedy_logprobs) / len(greedy_logprobs)
        return score, details

    if not scores:
        details["reason"] = "missing_top_logprobs"
        return None, details
    if mode == "min":
        aggregate = min(scores)
    elif mode == "mean":
        aggregate = sum(scores) / len(scores)
    elif mode == "bottom20":
        count = max(1, int(0.2 * len(scores)))
        aggregate = sum(sorted(scores)[:count]) / count
    # Stable sigmoid for unusually sharp distributions.
    clipped = max(-60.0, min(60.0, alpha * aggregate))
    score = 1.0 / (1.0 + math.exp(-clipped))
    details["aggregate_zscore"] = aggregate
    return score, details


class SpecEyesOrchestrator:
    """Run SpecEyes while delegating all agentic work to Spectool primitives."""

    def __init__(
        self,
        main_model: VLLMModel,
        draft_model: VLLMModel,
        config: AgentConfig,
        *,
        confidence_threshold: Optional[float] = None,
        confidence_top_k: Optional[int] = None,
        confidence_mode: Optional[str] = None,
        confidence_alpha: Optional[float] = None,
        router_max_tokens: Optional[int] = None,
        direct_max_tokens: Optional[int] = None,
    ):
        self.main_model = main_model
        self.draft_model = draft_model
        self.config = config
        self.confidence_threshold = (
            confidence_threshold if confidence_threshold is not None else
            _env_float("SPECEYE_CONFIDENCE_THRESHOLD", 0.98)
        )
        self.confidence_top_k = (
            confidence_top_k if confidence_top_k is not None else
            _env_int("SPECEYE_TOP_LOGPROBS", 64, minimum=2)
        )
        self.confidence_mode = (
            confidence_mode if confidence_mode is not None else
            os.environ.get("SPECEYE_CONFIDENCE_MODE", "min").strip().lower()
        )
        self.confidence_alpha = (
            confidence_alpha if confidence_alpha is not None else
            _env_float("SPECEYE_CONFIDENCE_ALPHA", 0.6)
        )
        self.router_max_tokens = (
            router_max_tokens if router_max_tokens is not None else
            _env_int("SPECEYE_ROUTER_MAX_TOKENS", 10)
        )
        self.direct_max_tokens = (
            direct_max_tokens if direct_max_tokens is not None else
            _env_int("SPECEYE_DIRECT_MAX_TOKENS", 20)
        )

    @staticmethod
    def _messages(images: list[Image.Image], system: str, question: str) -> list[dict]:
        content = [
            {"type": "image_url", "image_url": {"url": pil_to_data_url(image)}}
            for image in images
        ]
        content.append({"type": "text", "text": question})
        return [
            {"role": "system", "content": system},
            {"role": "user", "content": content},
        ]

    @staticmethod
    def _route_from_text(text: str) -> str:
        if (text or "").lstrip().lower().startswith("no"):
            return "direct"
        # This is the official behavior: yes and malformed/empty output both
        # take the conservative tool route.
        return "tool_required"

    @staticmethod
    def _direct_answer(text: str, *, normalize_mcq: bool) -> Optional[str]:
        """Extract the official direct branch's plain-text answer.

        Official SpecEyes uses ``SYSTEM_PROMPT_ANSWER`` and returns raw text,
        extracting an ``<answer>`` span only when a model emitted one despite
        that instruction.  Reuse Spectool's MCQ normalization by placing a raw
        answer inside a temporary tag only for parsing.
        """
        tagged = parse_answer(text, normalize_mcq=normalize_mcq)
        if tagged is not None:
            return tagged
        raw = (text or "").strip()
        if not raw:
            return None
        return parse_answer(
            f"<answer>{raw}</answer>", normalize_mcq=normalize_mcq)

    def _main_agent(
        self, images: list[Image.Image], question: str, *, sample_id: str,
        prompt_set: PromptSet, extra_prompt: str,
    ) -> Trajectory:
        """Run Spectool's fallback-only Main path from a clean context.

        Phase I/II remain SpecEyes policy.  Once an item reaches Phase III,
        however, it must be comparable with Spectool's own fallback baseline:
        a fresh tool registry, the normal prompt set and registered tools, and
        ``SpeculativeAgentLoop.run(..., fallback_only=True)``.  In that mode
        Spectool deliberately skips new Draft work and performs its clean Main
        takeover with the normal budgets and last-step finalization policy.
        """
        tools = ToolRegistry()
        tool_schemas = register_default_tools(
            tools, benchmark_profile=prompt_set.profile)
        agent = SpeculativeAgentLoop(
            self.main_model,
            # fallback_only never schedules Draft work; retaining the existing
            # instance satisfies the shared constructor without another model
            # request or a second 9B lifecycle.
            self.draft_model,
            tools,
            self.config,
            prompt_set=prompt_set,
            tool_schemas=tool_schemas,
        )
        try:
            return agent.run(
                images, question, sample_id=sample_id,
                extra_prompt=extra_prompt, fallback_only=True)
        finally:
            agent.shutdown()
            tools.cleanup()

    def run(
        self, image: Image.Image | list[Image.Image], question: str, *,
        sample_id: str = "", extra_prompt: str = "",
        prompt_set: Optional[PromptSet] = None,
    ) -> SpecEyesResult:
        """Run the three-stage SpecEyes policy for one sample."""
        started = time.perf_counter()
        images = [image] if isinstance(image, Image.Image) else list(image)
        if not images or not all(isinstance(item, Image.Image) for item in images):
            raise ValueError("SpecEyes requires one or more PIL input images")
        prompt_set = prompt_set or get_prompt_set()
        request_question = USER_PROMPT_TEMPLATE.format(question=question)
        if extra_prompt:
            request_question = extra_prompt + "\n\n" + request_question

        router_raw = ""
        router_time = 0.0
        try:
            router_started = time.perf_counter()
            router = self.main_model.generate_text(
                self._messages(
                    images, ROUTER_SYSTEM_PROMPT,
                    request_question + ROUTER_USER_SUFFIX),
                max_tokens=self.router_max_tokens,
            )
            router_time = time.perf_counter() - router_started
            router_raw = router["text"]
            route = self._route_from_text(router_raw)
        except Exception as exc:
            route = "tool_required"
            router_raw = f"[router_error] {type(exc).__name__}: {exc}"

        if route == "direct":
            draft_raw = ""
            draft_time = 0.0
            confidence: Optional[float] = None
            details: dict[str, Any] = {}
            try:
                draft_started = time.perf_counter()
                draft = self.draft_model.generate_text(
                    self._messages(
                        images, DIRECT_SYSTEM_PROMPT,
                        request_question + DIRECT_USER_SUFFIX),
                    max_tokens=self.direct_max_tokens,
                    logprobs=True,
                    top_logprobs=self.confidence_top_k,
                )
                draft_time = time.perf_counter() - draft_started
                draft_raw = draft["text"]
                confidence, details = answer_separability_from_logprobs(
                    draft.get("content_logprobs", []),
                    top_k=self.confidence_top_k,
                    mode=self.confidence_mode,
                    alpha=self.confidence_alpha,
                )
                answer = self._direct_answer(
                    draft_raw, normalize_mcq=prompt_set.normalize_mcq_answers)
                details["threshold"] = self.confidence_threshold
                if answer is not None and confidence is not None and confidence > self.confidence_threshold:
                    return SpecEyesResult(
                        final_answer=answer,
                        route=route,
                        decision="draft_accepted",
                        router_raw=router_raw,
                        draft_raw=draft_raw,
                        confidence=confidence,
                        confidence_details=details,
                        router_time=router_time,
                        draft_time=draft_time,
                        total_time=time.perf_counter() - started,
                    )
                if answer is None:
                    details["fallback_reason"] = "draft_missing_answer"
                elif confidence is None:
                    details["fallback_reason"] = "draft_missing_logprobs"
                else:
                    details["fallback_reason"] = "draft_low_confidence"
            except Exception as exc:
                draft_raw = f"[draft_error] {type(exc).__name__}: {exc}"
                details = {"fallback_reason": "draft_error"}

            fallback_started = time.perf_counter()
            trajectory = self._main_agent(
                images, question,
                sample_id=sample_id, prompt_set=prompt_set,
                extra_prompt=extra_prompt,
            )
            fallback_time = time.perf_counter() - fallback_started
            return SpecEyesResult(
                final_answer=trajectory.final_answer,
                route=route,
                decision="main_fallback_after_draft",
                router_raw=router_raw,
                draft_raw=draft_raw,
                confidence=confidence,
                confidence_details=details,
                main_trajectory=trajectory,
                router_time=router_time,
                draft_time=draft_time,
                fallback_time=fallback_time,
                total_time=time.perf_counter() - started,
            )

        fallback_started = time.perf_counter()
        trajectory = self._main_agent(
            images, question,
            sample_id=sample_id, prompt_set=prompt_set,
            extra_prompt=extra_prompt,
        )
        fallback_time = time.perf_counter() - fallback_started
        return SpecEyesResult(
            final_answer=trajectory.final_answer,
            route=route,
            decision="main_sandbox",
            router_raw=router_raw,
            main_trajectory=trajectory,
            router_time=router_time,
            fallback_time=fallback_time,
            total_time=time.perf_counter() - started,
        )


class SpecEyesBatchOrchestrator:
    """Execute the official SpecEyes three-stage funnel for a batch.

    The official implementation makes one padded Transformers request for all
    Phase-I items, then one for the ``no`` subset in Phase II, and finally
    handles its residual tool-agent items serially.  Spectool deliberately
    keeps its existing OpenAI-compatible vLLM servers, whose chat endpoint has
    no heterogeneous request-array API.  We therefore submit each stateless
    phase concurrently behind a strict phase barrier; vLLM can continuously
    batch those requests, while preserving the same routing and fallback
    semantics. Fallbacks are serial by default. An explicit worker limit can
    parallelize independent items; each still owns its ToolRegistry/session.
    """

    execution_mode = "phase_barrier_http_concurrency"

    def __init__(
        self,
        main_model: VLLMModel,
        draft_model: VLLMModel,
        config: AgentConfig,
        *,
        confidence_threshold: Optional[float] = None,
        confidence_top_k: Optional[int] = None,
        confidence_mode: Optional[str] = None,
        confidence_alpha: Optional[float] = None,
        router_max_tokens: Optional[int] = None,
        direct_max_tokens: Optional[int] = None,
        fallback_concurrency: Optional[int] = None,
    ):
        self.fallback_concurrency = (
            _env_int("SPECEYE_FALLBACK_NPROC", 1)
            if fallback_concurrency is None else int(fallback_concurrency)
        )
        if self.fallback_concurrency < 1:
            raise ValueError("fallback_concurrency must be at least 1")
        self._single = SpecEyesOrchestrator(
            main_model, draft_model, config,
            confidence_threshold=confidence_threshold,
            confidence_top_k=confidence_top_k,
            confidence_mode=confidence_mode,
            confidence_alpha=confidence_alpha,
            router_max_tokens=router_max_tokens,
            direct_max_tokens=direct_max_tokens,
        )

    def _prepare(self, request: SpecEyesRequest, batch_index: int) -> _BatchState:
        images = ([request.image] if isinstance(request.image, Image.Image)
                  else list(request.image))
        if not images or not all(isinstance(item, Image.Image) for item in images):
            raise ValueError("SpecEyes requires one or more PIL input images")
        prompt_set = request.prompt_set or get_prompt_set()
        request_question = USER_PROMPT_TEMPLATE.format(question=request.question)
        if request.extra_prompt:
            request_question = request.extra_prompt + "\n\n" + request_question
        return _BatchState(
            request=request,
            batch_index=batch_index,
            images=images,
            prompt_set=prompt_set,
            request_question=request_question,
            started=time.perf_counter(),
        )

    def _router_stage(self, state: _BatchState) -> None:
        started = time.perf_counter()
        try:
            router = self._single.main_model.generate_text(
                self._single._messages(
                    state.images, ROUTER_SYSTEM_PROMPT,
                    state.request_question + ROUTER_USER_SUFFIX),
                max_tokens=self._single.router_max_tokens,
            )
            state.router_raw = router["text"]
            state.route = self._single._route_from_text(state.router_raw)
        except Exception as exc:
            state.route = "tool_required"
            state.router_raw = f"[router_error] {type(exc).__name__}: {exc}"
        finally:
            state.router_time = time.perf_counter() - started

    def _draft_stage(self, state: _BatchState) -> None:
        started = time.perf_counter()
        try:
            draft = self._single.draft_model.generate_text(
                self._single._messages(
                    state.images, DIRECT_SYSTEM_PROMPT,
                    state.request_question + DIRECT_USER_SUFFIX),
                max_tokens=self._single.direct_max_tokens,
                logprobs=True,
                top_logprobs=self._single.confidence_top_k,
            )
            state.draft_raw = draft["text"]
            state.confidence, state.confidence_details = (
                answer_separability_from_logprobs(
                    draft.get("content_logprobs", []),
                    top_k=self._single.confidence_top_k,
                    mode=self._single.confidence_mode,
                    alpha=self._single.confidence_alpha,
                ))
        except Exception as exc:
            state.draft_raw = f"[draft_error] {type(exc).__name__}: {exc}"
            state.confidence_details = {"fallback_reason": "draft_error"}
        finally:
            state.draft_time = time.perf_counter() - started

    @staticmethod
    def _parallel(states: list[_BatchState], stage) -> None:
        """Run a stateless stage for every item, then wait at its barrier."""
        if not states:
            return
        with ThreadPoolExecutor(max_workers=len(states)) as executor:
            # Calling ``result`` propagates programming errors while the stage
            # itself converts service errors into conservative fallbacks.
            for future in [executor.submit(stage, state) for state in states]:
                future.result()

    def _fallback_result(self, state: _BatchState, *, after_draft: bool) -> SpecEyesResult:
        fallback_started = time.perf_counter()
        try:
            trajectory = self._single._main_agent(
                state.images, state.request.question,
                sample_id=state.request.sample_id, prompt_set=state.prompt_set,
                extra_prompt=state.request.extra_prompt,
            )
            answer = trajectory.final_answer
            error = ""
        except Exception as exc:
            trajectory = None
            answer = None
            error = f"main_agent_error: {type(exc).__name__}: {exc}"
        fallback_time = time.perf_counter() - fallback_started
        return SpecEyesResult(
            final_answer=answer,
            route=state.route,
            decision=("main_fallback_after_draft" if after_draft else "main_sandbox"),
            router_raw=state.router_raw,
            draft_raw=state.draft_raw,
            confidence=state.confidence,
            confidence_details=state.confidence_details,
            main_trajectory=trajectory,
            error=error,
            router_time=state.router_time,
            draft_time=state.draft_time,
            fallback_time=fallback_time,
            total_time=time.perf_counter() - state.started,
        )

    def _with_batch_metadata(
        self, result: SpecEyesResult, *, batch_size: int, batch_index: int,
    ) -> SpecEyesResult:
        result.batch_size = batch_size
        result.batch_index = batch_index
        result.fallback_concurrency = self.fallback_concurrency
        result.batch_execution = (
            self.execution_mode if self.fallback_concurrency == 1
            else "phase_barrier_http_concurrency_parallel_fallback"
        )
        return result

    def run(self, requests: list[SpecEyesRequest]) -> list[SpecEyesResult]:
        """Return results in request order after the Phase-I → II → III funnel."""
        if not requests:
            return []
        states = [self._prepare(request, index) for index, request in enumerate(requests)]
        batch_size = len(states)

        # Official Phase I: all questions are screened before Phase II begins.
        self._parallel(states, self._router_stage)

        direct_states = [state for state in states if state.route == "direct"]
        # Official Phase II: only the no-tool subset reaches the small model.
        self._parallel(direct_states, self._draft_stage)

        results: list[Optional[SpecEyesResult]] = [None] * batch_size
        fallback_states: list[tuple[_BatchState, bool]] = []
        for state in states:
            if state.route != "direct":
                fallback_states.append((state, False))
                continue

            details = state.confidence_details
            details["threshold"] = self._single.confidence_threshold
            answer = self._single._direct_answer(
                state.draft_raw,
                normalize_mcq=state.prompt_set.normalize_mcq_answers,
            )
            if (answer is not None and state.confidence is not None
                    and state.confidence > self._single.confidence_threshold):
                results[state.batch_index] = SpecEyesResult(
                    final_answer=answer,
                    route=state.route,
                    decision="draft_accepted",
                    router_raw=state.router_raw,
                    draft_raw=state.draft_raw,
                    confidence=state.confidence,
                    confidence_details=details,
                    router_time=state.router_time,
                    draft_time=state.draft_time,
                    total_time=time.perf_counter() - state.started,
                )
                continue
            if "fallback_reason" not in details:
                if answer is None:
                    details["fallback_reason"] = "draft_missing_answer"
                elif state.confidence is None:
                    details["fallback_reason"] = "draft_missing_logprobs"
                else:
                    details["fallback_reason"] = "draft_low_confidence"
            fallback_states.append((state, True))

        # Keep the official serial default. Parallel workers share only the
        # thread-safe model client; _main_agent creates per-item tools/session.
        if self.fallback_concurrency == 1:
            for state, after_draft in fallback_states:
                results[state.batch_index] = self._fallback_result(
                    state, after_draft=after_draft)
        elif fallback_states:
            with ThreadPoolExecutor(max_workers=self.fallback_concurrency) as executor:
                futures = {
                    executor.submit(self._fallback_result, state, after_draft=after_draft):
                    state.batch_index
                    for state, after_draft in fallback_states
                }
                for future in as_completed(futures):
                    results[futures[future]] = future.result()

        assert all(result is not None for result in results)
        return [self._with_batch_metadata(
            result, batch_size=batch_size, batch_index=index)
            for index, result in enumerate(results)]
