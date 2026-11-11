"""Stateless OpenAI transport; preserves native tool calls and actual usage."""
import copy
import json
import re
import time
import uuid

from .core import BudgetExceeded, InfrastructureError, ProtocolError, Turn
from .prompts import scoring_messages


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
        if client is None:
            from openai import OpenAI
            client = OpenAI(base_url=base_url, api_key=api_key, max_retries=0)
        self.client = client

    def _request(self, messages, schemas, ledger, phase, *, scoring=False):
        maximum = ledger.allowance(1 if scoring else min(self.max_tokens, ledger.config.max_tokens))
        remaining_wall = ledger.config.wall_seconds - (time.perf_counter() - ledger.started)
        timeout = min(self.timeout, max(0.01, remaining_wall)) if ledger.config.wall_seconds else self.timeout
        kwargs = dict(model=self.model, messages=copy.deepcopy(messages), temperature=0.0,
            max_tokens=maximum, timeout=timeout,
            extra_body={"chat_template_kwargs": {"enable_thinking": False if scoring else self.thinking}})
        if not scoring:
            kwargs["extra_body"]["repetition_penalty"] = 1.1
        if schemas:
            kwargs.update(tools=schemas, tool_choice="none" if scoring else "auto")
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
        try:
            response = self.client.chat.completions.create(**kwargs)
        except Exception as exc:
            # Includes HTTP errors and timeouts: no hidden retry/extra allowance.
            event.update(status="raised", error=f"{type(exc).__name__}: {exc}",
                elapsed_s=time.perf_counter() - started)
            ledger.requests.append(event)
            ledger.usage_complete = False
            if ledger.config.wall_seconds and time.perf_counter() - ledger.started >= ledger.config.wall_seconds:
                raise BudgetExceeded("wall_budget") from exc
            raise InfrastructureError(f"{self.role}/{phase}: {type(exc).__name__}: {exc}") from exc
        event.update(status="returned", elapsed_s=time.perf_counter() - started,
            usage=(response.usage.model_dump() if response.usage is not None else None),
            response=response.model_dump())
        ledger.record_request(event)
        if not response.choices:
            raise ProtocolError("empty_choices")
        return response.choices[0]

    def generate(self, history, schemas, ledger, phase):
        choice = self._request(history, schemas, ledger, phase)
        question = "\n".join(part.get("text", "") for message in history[:2]
            for part in (message.get("content", []) if isinstance(message.get("content"), list) else []))
        return parse_turn(choice, question)

    def score(self, history, candidate, schemas, ledger):
        choice = self._request(scoring_messages(history, candidate.message), schemas,
            ledger, "score", scoring=True)
        if getattr(choice, "finish_reason", None) not in {"stop", "length"}:
            return None
        # max_tokens=1 normally ends with length; a single digit is still valid.
        return parse_score(choice.message.content)
