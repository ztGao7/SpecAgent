"""Stateless OpenAI transport; preserves native tool calls and actual usage."""
import asyncio
import copy
import json
import math
import re
import threading
import time
import uuid

from .core import BudgetExceeded, InfrastructureError, ProtocolError, Turn
from .prompts import scoring_messages, verification_messages


def positive_probability(choice):
    """DREAM-R TBVM score: P(positive) / (P(positive) + P(negative)) at the first token."""
    content = getattr(getattr(choice, "logprobs", None), "content", None) or []
    if not content:
        return None
    first = content[0]
    seen = {}
    for cand in [first, *(getattr(first, "top_logprobs", None) or [])]:
        text = (getattr(cand, "token", "") or "").strip().lower()
        lp = getattr(cand, "logprob", None)
        if not text or lp is None:
            continue
        lp = float(lp)
        if math.isfinite(lp) and lp <= 0:
            seen[text] = max(lp, seen.get(text, -math.inf))
    mass = {"positive": 0.0, "negative": 0.0}
    for text, lp in seen.items():
        for label in mass:
            if label.startswith(text):  # a label may start with a sub-word token
                mass[label] += math.exp(lp)
    total = mass["positive"] + mass["negative"]
    return mass["positive"] / total if total > 0 else None


class PendingTurn:
    """A Main generation running in the background that can be abandoned."""

    def __init__(self, endpoint, kwargs, event, started, ledger, phase, question):
        self.endpoint, self.kwargs, self.event, self.started = endpoint, kwargs, event, started
        self.ledger, self.phase, self.question = ledger, phase, question
        self._lock = threading.Lock()
        self._cancelled = False
        self._loop = self._task = self._outcome = None
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    async def _acreate(self):
        from openai import AsyncOpenAI
        client = AsyncOpenAI(base_url=self.endpoint.base_url, api_key=self.endpoint.api_key,
            max_retries=0)
        try:
            with self._lock:
                self._loop, self._task = asyncio.get_running_loop(), asyncio.current_task()
                if self._cancelled:
                    raise asyncio.CancelledError()
            # Cancelling this task closes the HTTP request, so vLLM aborts the generation.
            return await client.chat.completions.create(**self.kwargs)
        finally:
            await client.close()

    def _mark_cancelled(self):
        self.event.update(status="cancelled", elapsed_s=time.perf_counter() - self.started)
        self.ledger.record_cancelled(self.event)
        self._outcome = ("cancelled", None)

    def _run(self):
        try:
            if self.endpoint.injected_client:  # tests: a sync client cannot be aborted
                response = self.endpoint.client.chat.completions.create(**self.kwargs)
            else:
                response = asyncio.run(self._acreate())
        except asyncio.CancelledError:
            return self._mark_cancelled()
        except Exception as exc:
            if self._cancelled:
                return self._mark_cancelled()
            try:
                self.endpoint._failed(self.event, self.started, exc, self.ledger, self.phase)
            except Exception as wrapped:
                self._outcome = ("error", wrapped)
            return
        try:
            choice = self.endpoint._returned(self.event, self.started, response, self.ledger)
            self._outcome = ("turn", parse_turn(choice, self.question))
        except Exception as exc:
            self._outcome = ("error", exc)

    def cancel(self):
        """Abandon the request; returns what happened to it for the trajectory."""
        with self._lock:
            self._cancelled = True
            if self._loop is not None and self._task is not None and self._outcome is None:
                try:
                    self._loop.call_soon_threadsafe(self._task.cancel)
                except RuntimeError:
                    pass  # loop already closed: the request has finished
        self._thread.join(timeout=15)
        kind = self._outcome[0] if self._outcome else "still_running"
        return {"turn": "completed_discarded", "error": "errored_discarded"}.get(kind, kind)

    def result(self):
        self._thread.join()
        kind, value = self._outcome
        if kind == "turn":
            return value
        if kind == "error":
            raise value
        raise ProtocolError("rewrite_cancelled")


def parse_score(content):
    text = content.strip() if isinstance(content, str) else ""
    return int(text) if re.fullmatch(r"[0-9]", text) else None


def parse_turn(choice, question):
    from spectool_agent.agent import _detect_mcq_choices
    from spectool_agent.model import (ModelResponse, ToolCall, parse_answer,
        parse_tool_call, build_assistant_tool_call_message)

    msg = choice.message
    raw = msg.model_dump() if hasattr(msg, "model_dump") else vars(msg)
    content = raw.get("content") or ""
    reasoning = raw.get("reasoning_content") or raw.get("reasoning") or ""
    canonical = {"role": "assistant", "content": content}
    if reasoning:
        canonical["reasoning_content"] = reasoning
    calls, ids = [], []
    native = raw.get("tool_calls") or []
    try:
        for tc in native:
            if not isinstance(tc, dict):
                tc = tc.model_dump()
            fn = tc["function"]
            params = json.loads(fn.get("arguments") or "{}")
            if not isinstance(params, dict):
                raise ValueError("tool arguments must be an object")
            calls.append(ToolCall(tool_name=fn["name"], params=params))
            ids.append(tc.get("id") or f"call_{uuid.uuid4().hex}")
        if native:
            canonical["tool_calls"] = [
                {"id": cid, "type": "function", "function": {
                    "name": call.tool_name,
                    "arguments": json.dumps(call.params, ensure_ascii=False)}}
                for call, cid in zip(calls, ids)]
        else:
            call = parse_tool_call(ModelResponse(content, reasoning))
            if call is not None:
                calls, ids = [call], [f"call_{uuid.uuid4().hex}"]
                canonical = build_assistant_tool_call_message(content, calls, ids)
                if reasoning:
                    canonical["reasoning_content"] = reasoning
    except (ValueError, TypeError, KeyError) as exc:
        return Turn(canonical, valid=False, error=f"malformed_tool_call: {exc}")
    finish = getattr(choice, "finish_reason", None)
    answer = None if calls else parse_answer(
        ModelResponse(content, reasoning, finish_reason=finish),
        choices=_detect_mcq_choices(question), question=question)
    valid = finish != "length" and bool(content.strip() or reasoning.strip() or calls)
    error = "truncated_turn" if finish == "length" else "empty_turn"
    if not calls and "<tool_call>" in content:
        valid, error = False, "malformed_tool_call"
    if not calls and "<answer>" in content and answer is None:
        valid, error = False, "malformed_final_answer"
    return Turn(canonical, calls, ids, answer, valid, "" if valid else error)


class Endpoint:
    def __init__(self, *, role, model, base_url, timeout, thinking,
                 max_tokens=4096, api_key="EMPTY", client=None):
        self.role, self.model = role, model
        self.timeout, self.thinking, self.max_tokens = timeout, thinking, max_tokens
        self.base_url, self.api_key, self.injected_client = base_url, api_key, client is not None
        if client is None:
            from openai import OpenAI
            client = OpenAI(base_url=base_url, api_key=api_key, max_retries=0)
        self.client = client

    def _prepare(self, messages, schemas, ledger, phase, *, scoring=False, verify=False):
        judge = scoring or verify
        maximum = ledger.allowance(1 if judge else min(self.max_tokens, ledger.config.max_tokens))
        remaining_wall = ledger.config.wall_seconds - (time.perf_counter() - ledger.started)
        timeout = min(self.timeout, max(0.01, remaining_wall)) if ledger.config.wall_seconds else self.timeout
        kwargs = dict(model=self.model, messages=copy.deepcopy(messages), temperature=0.0,
            max_tokens=maximum, timeout=timeout,
            extra_body={"chat_template_kwargs": {"enable_thinking": False if judge else self.thinking}})
        if not judge:
            kwargs["extra_body"]["repetition_penalty"] = 1.1
        if schemas:
            kwargs.update(tools=schemas, tool_choice="none" if judge else "auto")
        if verify:
            # DREAM-R reads positive/negative from the top-20 logprobs of one token.
            # As with the digit score, the continuation is primed and constrained so
            # that Qwen3.5 cannot spend its single token on prose or whitespace.
            kwargs.update(logprobs=True, top_logprobs=20)
            kwargs['extra_body'].update(add_generation_prompt=False,
                continue_final_message=True,
                structured_outputs={'choice': ['positive', 'negative']})
        if scoring:
            kwargs.update(logprobs=True, top_logprobs=10)
            # Qwen3.5 may otherwise emit prose or whitespace as its sole token.
            # Prime a continuation (as upstream does) and constrain the score's
            # format, rather than silently extracting a digit from top-logprobs.
            kwargs['extra_body'].update(add_generation_prompt=False,
                continue_final_message=True, structured_outputs={'regex': '[0-9]'})
        started = time.perf_counter()
        event = {"role": self.role, "phase": phase, "model": self.model,
            "started_s": started - ledger.started,
            "request": {k: copy.deepcopy(v) for k, v in kwargs.items() if k != "timeout"}}
        return kwargs, event, started

    def _failed(self, event, started, exc, ledger, phase):
        # Includes HTTP errors and timeouts: no hidden retry/extra allowance.
        event.update(status="raised", error=f"{type(exc).__name__}: {exc}",
            elapsed_s=time.perf_counter() - started)
        ledger.requests.append(event)
        ledger.usage_complete = False
        if ledger.config.wall_seconds and time.perf_counter() - ledger.started >= ledger.config.wall_seconds:
            raise BudgetExceeded("wall_budget") from exc
        raise InfrastructureError(f"{self.role}/{phase}: {type(exc).__name__}: {exc}") from exc

    def _returned(self, event, started, response, ledger):
        event.update(status="returned", elapsed_s=time.perf_counter() - started,
            usage=(response.usage.model_dump() if response.usage is not None else None),
            response=response.model_dump())
        ledger.record_request(event)
        if not response.choices:
            raise ProtocolError("empty_choices")
        return response.choices[0]

    def _request(self, messages, schemas, ledger, phase, *, scoring=False, verify=False):
        kwargs, event, started = self._prepare(messages, schemas, ledger, phase,
            scoring=scoring, verify=verify)
        try:
            response = self.client.chat.completions.create(**kwargs)
        except Exception as exc:
            self._failed(event, started, exc, ledger, phase)
        return self._returned(event, started, response, ledger)

    @staticmethod
    def _question(history):
        return "\n".join(part.get("text", "") for message in history[:2]
            for part in (message.get("content", []) if isinstance(message.get("content"), list) else []))

    def generate(self, history, schemas, ledger, phase):
        choice = self._request(history, schemas, ledger, phase)
        return parse_turn(choice, self._question(history))

    def start_generate(self, history, schemas, ledger, phase):
        """Start a generation in the background (DREAM-R parallel Main rewrite)."""
        kwargs, event, started = self._prepare(history, schemas, ledger, phase)
        return PendingTurn(self, kwargs, event, started, ledger, phase, self._question(history))

    def verify(self, history, candidate, schemas, ledger):
        choice = self._request(verification_messages(history, candidate.message), schemas,
            ledger, "verify", verify=True)
        if getattr(choice, "finish_reason", None) not in {"stop", "length"}:
            return None
        return positive_probability(choice)

    def score(self, history, candidate, schemas, ledger):
        choice = self._request(scoring_messages(history, candidate.message), schemas,
            ledger, "score", scoring=True)
        if getattr(choice, "finish_reason", None) not in {"stop", "length"}:
            return None
        # max_tokens=1 normally ends with length; a single digit is still valid.
        return parse_score(choice.message.content)
