"""Additive, evidence-at-a-time variant of the production speculative loop.

The production ``run`` lifecycle, assessment protocol, candidate verifier,
original-image fast gate, and fallback implementation remain inherited.  Only
Draft/reviewer scheduling changes: every tool entry receives an explicit
decision before another Draft generation is permitted.  Execution errors retain
the production three-error allowance and are marked as policy-only decisions.

Import this module with the normal Spectool package on ``PYTHONPATH``.  No
production module or global class binding is changed by importing it.
"""

from __future__ import annotations

import copy
import sys
import threading
import time
from dataclasses import dataclass, field

from spectool_agent.draft_model import DraftModel
from spectool_agent.model import (
    RequestCancellation,
    RequestCancelled,
    build_tool_result_messages,
    parse_tool_call,
    strip_reasoning_tags,
)
from spectool_agent.prompt import DRAFT_ANSWER_FORMAT_REPAIR
from spectool_agent.speculative_agent import (
    CancellableTask,
    SpeculativeAgentLoop,
    _MAX_CONSECUTIVE_QUALITY_BAD,
    _MAX_CONSECUTIVE_TOOL_ERRORS,
    _MAX_TRUNCATED_DRAFT_STEPS,
    _POOL_OBSERVATION_CHARS,
    _diagnose_tool_call,
    _truncate_image_history,
)
from spectool_agent.types import PoolEntry, StepResult
from .sync_lifecycle import run as _synchronous_run


@dataclass
class _EvidenceHandshake:
    """Per-sample state; never attached to a shared model or loop instance."""

    condition: threading.Condition = field(default_factory=threading.Condition)
    entries: dict = field(default_factory=dict)
    decisions: dict = field(default_factory=dict)
    sequence: int = 0
    last_evidence_id: int | None = None
    closed: bool = False
    draft_finished: bool = False


class SynchronousSpeculativeAgentLoop(SpeculativeAgentLoop):
    """Run one Draft evidence action, Main assessment, then an explicit gate."""

    # The frozen parent loop reads this opt-in before starting its first Draft
    # request; the fast-answer gate therefore completes before exploration.
    synchronous_initial_fast_answer = True
    run = _synchronous_run

    def __init__(self, main_model, draft_model, tools, config,
                 prompt_set=None, tool_schemas=None):
        if not config.quality_judge_enabled:
            raise ValueError(
                "Synchronous evidence testing requires quality_judge_enabled=True")
        super().__init__(
            main_model, draft_model, tools, config,
            prompt_set=prompt_set, tool_schemas=tool_schemas)

    @staticmethod
    def _handshake(pool):
        # Both workers can be first to initialize.  The pool's existing lock
        # serializes creation; the condition has its own independent lock.
        with pool._lock:
            state = getattr(pool, "_synchronous_evidence_handshake", None)
            if state is None:
                state = _EvidenceHandshake()
                pool._synchronous_evidence_handshake = state
            return state

    def _sync_event(self, traj, phase, *, entry=None, evidence_id=None,
                    status="ok", started_at=None, **details):
        """Persist audit events using Trajectory's native timing serializer."""
        now = time.perf_counter()
        with self._timing_lock:
            traj.record_timing_event(
                "sync", phase, now if started_at is None else started_at, now,
                step_idx=-1 if entry is None else entry.step_idx,
                status=status, thread_id=threading.get_ident())
            event = traj.timing_events[-1]
            if evidence_id is not None:
                event["evidence_id"] = evidence_id
            if entry is not None:
                event["draft_step_idx"] = entry.step_idx
            event.update(details)

    def _publish(self, pool, cancel, traj, entry):
        state = self._handshake(pool)
        with state.condition:
            if state.closed or cancel.cancelled:
                return None
            evidence_id = None
            if entry.tool_call is not None:
                state.sequence += 1
                evidence_id = state.sequence
                state.entries[id(entry)] = (evidence_id, entry)
                state.last_evidence_id = evidence_id
            pool.add(entry)
            if evidence_id is not None:
                self._sync_event(
                    traj, "evidence_published", entry=entry,
                    evidence_id=evidence_id)
            state.condition.notify_all()
            return evidence_id

    def _decide_locked(self, state, traj, entry, decision, reason, **details):
        """Caller holds the condition; publication is exactly once per entry."""
        key = id(entry)
        if key in state.decisions:
            return state.decisions[key]
        evidence_id, _ = state.entries[key]
        state.decisions[key] = decision
        if decision != "continue":
            state.closed = True
        self._sync_event(
            traj, "decision", entry=entry, evidence_id=evidence_id,
            decision=decision, reason=reason, **details)
        state.condition.notify_all()
        return decision

    def _decide(self, pool, traj, entry, decision, reason, **details):
        state = self._handshake(pool)
        with state.condition:
            return self._decide_locked(
                state, traj, entry, decision, reason, **details)

    def _close_handshake(self, pool, traj, decision="stop", reason="reviewer_exit"):
        state = self._handshake(pool)
        with state.condition:
            state.closed = True
            for _, entry in state.entries.values():
                if id(entry) not in state.decisions:
                    self._decide_locked(state, traj, entry, decision, reason)
            state.condition.notify_all()

    def _await_decision(self, pool, cancel, traj, entry):
        state = self._handshake(pool)
        started = time.perf_counter()
        with state.condition:
            while id(entry) not in state.decisions:
                if cancel.cancelled or state.closed:
                    # The inherited fast gate can cancel without entering the
                    # reviewer.  Always give its pending evidence a terminal
                    # decision as well; cancellation never implies continue.
                    self._decide_locked(
                        state, traj, entry,
                        "answer" if traj.final_answer is not None else "stop",
                        "cancelled_or_closed")
                    break
                state.condition.wait(timeout=0.05)
            decision = state.decisions[id(entry)]
            evidence_id, _ = state.entries[id(entry)]
            self._sync_event(
                traj, "draft_wait", entry=entry, evidence_id=evidence_id,
                started_at=started, decision=decision)
            return (decision == "continue" and not cancel.cancelled
                    and not state.closed)

    @staticmethod
    def _first_action_message(assistant_message, call_id):
        """Keep a valid native assistant/tool pair when extra calls are dropped."""
        message = copy.deepcopy(assistant_message)
        native_calls = message.get("tool_calls") or []
        if native_calls:
            matching = [call for call in native_calls if call.get("id") == call_id]
            message["tool_calls"] = (matching or native_calls)[:1]
        return message

    def _draft_loop(self, initial_messages, image, pool, cancel, traj):
        state = self._handshake(pool)
        draft_msgs = copy.deepcopy(initial_messages)
        user_content = initial_messages[1].get("content", "") if len(initial_messages) > 1 else ""
        user_text = (user_content if isinstance(user_content, str) else "\n".join(
            part.get("text", "") for part in user_content if part.get("type") == "text"))
        stuck_steps = 0
        try:
            for step_idx in range(self.config.draft_max_steps):
                with state.condition:
                    if cancel.cancelled or state.closed:
                        break
                    self._sync_event(
                        traj, "draft_generation_start", draft_step_idx=step_idx,
                        previous_evidence_id=state.last_evidence_id)

                t0 = time.perf_counter()
                request_cancel = RequestCancellation()
                if not cancel.register_request(request_cancel):
                    break
                try:
                    kwargs = {"tools": self.tool_schemas,
                              "max_tokens": self.config.max_tokens}
                    if isinstance(self.draft_model, DraftModel):
                        kwargs["cancel_token"] = request_cancel
                    response_text = self._timed_call(
                        traj, "draft", "draft_step", self.draft_model.generate_full,
                        draft_msgs, step_idx=step_idx, **kwargs)
                except RequestCancelled:
                    break
                except Exception as exc:
                    print(f"  [WARN] Draft 模型失败: {exc}", file=sys.stderr)
                    break
                finally:
                    cancel.unregister_request(request_cancel)
                with state.condition:
                    if cancel.cancelled or state.closed:
                        break

                draft_time = time.perf_counter() - t0
                truncated = self.draft_model.last_truncated
                tool_calls = list(self.draft_model.last_tool_calls)
                call_ids = list(self.draft_model.last_tool_call_ids)
                if not tool_calls:
                    parsed = parse_tool_call(response_text)
                    if parsed is not None:
                        tool_calls, call_ids = [parsed], [None]
                answer = self._parse_final_answer(response_text, user_text)
                if not tool_calls and answer is not None:
                    self._publish(pool, cancel, traj, PoolEntry(
                        step_idx=step_idx, response_text=response_text,
                        reasoning=strip_reasoning_tags(response_text),
                        answer=answer, draft_time=draft_time))
                    break

                assistant_message = copy.deepcopy(self.draft_model.last_assistant_msg)
                if tool_calls:
                    stuck_steps = 0
                    # Even executing a sibling after continue would reuse an
                    # action generated before assessment.  Discard siblings and
                    # regenerate the next action from the next accepted state.
                    tc = tool_calls[0]
                    cid = call_ids[0] if call_ids else None
                    assistant_message = self._first_action_message(assistant_message, cid)
                    if len(tool_calls) > 1:
                        self._sync_event(
                            traj, "discarded_parallel_actions",
                            draft_step_idx=step_idx,
                            discarded_count=len(tool_calls) - 1)
                    draft_msgs.append(assistant_message)
                    if cancel.cancelled:
                        break
                    observation, result_images = self._timed_call(
                        traj, "tool", "draft_tool", self.tools.dispatch,
                        tc, image, step_idx=step_idx)
                    entry = PoolEntry(
                        step_idx=step_idx, response_text=response_text,
                        reasoning=strip_reasoning_tags(response_text),
                        tool_call=tc, tool_observation=observation,
                        tool_result_images=result_images, draft_time=draft_time,
                        assistant_message=assistant_message, tool_call_id=cid)
                    if self._publish(pool, cancel, traj, entry) is None:
                        break
                    draft_msgs.extend(build_tool_result_messages(
                        cid, observation, result_images))
                    if not self._await_decision(pool, cancel, traj, entry):
                        break
                else:
                    draft_msgs.append(assistant_message)
                    nudge = _diagnose_tool_call(response_text)
                    if not truncated:
                        nudge += ("\n" + DRAFT_ANSWER_FORMAT_REPAIR + "\n"
                                  + self._answer_format_instruction(user_text))
                    draft_msgs.append({"role": "user", "content": [
                        {"type": "text", "text": nudge}]})
                    self._publish(pool, cancel, traj, PoolEntry(
                        step_idx=step_idx, response_text=response_text,
                        reasoning=strip_reasoning_tags(response_text),
                        draft_time=draft_time))
                    if truncated:
                        stuck_steps += 1
                        if stuck_steps >= _MAX_TRUNCATED_DRAFT_STEPS:
                            break
                    else:
                        stuck_steps = 0
                _truncate_image_history(draft_msgs, self.config.max_image_history)
        finally:
            with state.condition:
                state.draft_finished = True
                state.condition.notify_all()

    def _review_loop(self, image, user_text, pool, traj, cancel,
                     draft_future, fallback_coordinator, fast_future=None,
                     fast_request_cancel=None, fast_answer_consumer=None):
        state = self._handshake(pool)
        active_request = None
        active_future = None
        exit_reason = "reviewer_exit"
        try:
            context_started = time.perf_counter()
            shared_ctx = self._shared_prompt_context(image, user_text)
            self._record_timing_span(
                traj, "client", "shared_review_context", context_started,
                time.perf_counter())
            pool_check_index = 0
            consecutive_tool_errors = 0
            consecutive_quality_bad = 0

            def handoff_fast_answer():
                if fast_answer_consumer is not None:
                    self._timed_call(
                        traj, "client", "fast_answer_handoff",
                        fast_answer_consumer, handoff=True)
                elif fast_future is not None and not fast_future.done():
                    if fast_request_cancel is not None:
                        fast_request_cancel.cancel()
                    fast_future.cancel()
                return traj.final_answer is not None

            def review_candidate():
                if traj.final_answer is not None or handoff_fast_answer():
                    return
                current_step_idx = pool.entry_count
                review = self._run_main_review(
                    image, user_text, pool, current_step_idx, traj,
                    shared_ctx=shared_ctx, pool_entries=pool.get_all())
                if self._apply_review_result(
                        review, pool, traj, image, user_text, current_step_idx,
                        stop_reason="main_answer_draft_evidence_verified"):
                    cancel.cancel()

            while not cancel.cancelled:
                new_entries, pool_check_index = pool.snapshot_since(pool_check_index)
                for entry in new_entries:
                    if cancel.cancelled or traj.final_answer is not None:
                        return
                    current_step_idx = pool.entry_count
                    if entry.answer is not None:
                        review_candidate()
                        return
                    if entry.tool_call is None:
                        consecutive_tool_errors = 0
                        continue

                    with state.condition:
                        evidence_id, _ = state.entries[id(entry)]
                    if self._tool_entry_has_error(entry):
                        consecutive_tool_errors += 1
                        error_text = (entry.tool_observation or "<empty observation>").strip()
                        terminal = consecutive_tool_errors >= _MAX_CONSECUTIVE_TOOL_ERRORS
                        mode = "tool_error_fallback" if terminal else "tool_error_tolerated"
                        traj.steps.append(StepResult(
                            step_idx=current_step_idx,
                            reasoning=(f"tool_error_streak={consecutive_tool_errors}: "
                                       f"{error_text[:_POOL_OBSERVATION_CHARS]}"),
                            answer=f"{mode}:{consecutive_tool_errors}",
                            verdict_mode=mode, judge_score=-1.0))
                        self._sync_event(
                            traj, "assessment_complete", entry=entry,
                            evidence_id=evidence_id,
                            status="skipped_tool_error_policy",
                            quality_passed=None,
                            answerability_status="skipped_tool_error_policy")
                        if not terminal:
                            self._decide(pool, traj, entry, "continue", "tool_error_tolerated")
                            continue
                        if handoff_fast_answer():
                            self._decide(pool, traj, entry, "answer", "fast_main_answer")
                            return
                        traj.stop_reason = "draft_tool_error_streak"
                        self._decide(pool, traj, entry, "takeover", "tool_error_streak")
                        self._stop_draft(draft_future, cancel)
                        self._run_fallback_once(
                            fallback_coordinator, image, user_text, pool, traj,
                            CancellableTask(), inject_draft=False,
                            qj_failed_entry=entry, trigger="tool_error_streak")
                        return
                    consecutive_tool_errors = 0

                    if handoff_fast_answer():
                        self._decide(pool, traj, entry, "answer", "fast_main_answer")
                        return
                    active_request = RequestCancellation()
                    evidence_count = pool.position_after(entry)
                    self._sync_event(
                        traj, "assessment_start", entry=entry, evidence_id=evidence_id)
                    active_future = self._executor.submit(
                        self._assess_tool_entry, image, user_text, entry,
                        shared_ctx=shared_ctx, step_idx=current_step_idx,
                        traj=traj, request_cancel=active_request)
                    while not active_future.done():
                        if cancel.cancelled:
                            active_request.cancel()
                            active_future.cancel()
                            exit_reason = "assessment_cancelled"
                            return
                        with state.condition:
                            state.condition.wait(timeout=0.02)
                    judge_outcome, answer_outcome = active_future.result()
                    active_future = None
                    active_request = None
                    quality_passed = judge_outcome.get("passed")
                    self._sync_event(
                        traj, "assessment_complete", entry=entry,
                        evidence_id=evidence_id, quality_passed=quality_passed,
                        quality_status=judge_outcome.get("status", "ok"),
                        answerability_status=answer_outcome.get("status"))
                    self._record_quality_judge(traj, judge_outcome)
                    if quality_passed is not None:
                        pool.mark_quality(entry, quality_passed)

                    main_answer = answer_outcome.get("answer")
                    if main_answer is not None and quality_passed is True:
                        status = self._commit_early_answer(
                            pool, traj, main_answer, evidence_count,
                            "main_answer_from_draft_evidence")
                        if status == "accepted":
                            self._record_answerability_judge(traj, answer_outcome)
                            self._decide(pool, traj, entry, "answer", "main_answer_from_draft_evidence")
                            self._stop_draft(draft_future, cancel)
                            return
                        answer_outcome = dict(answer_outcome, answer=None, status=status)
                    self._record_answerability_judge(traj, answer_outcome)
                    if cancel.cancelled or traj.final_answer is not None:
                        exit_reason = "cancelled_after_assessment"
                        return
                    if quality_passed is True:
                        consecutive_quality_bad = 0
                        self._decide(pool, traj, entry, "continue", "quality_good_insufficient")
                    elif quality_passed is None:
                        # UNKNOWN preserves any existing BAD streak.
                        self._decide(pool, traj, entry, "continue", "quality_unknown")
                    else:
                        consecutive_quality_bad += 1
                        if consecutive_quality_bad < _MAX_CONSECUTIVE_QUALITY_BAD:
                            self._decide(pool, traj, entry, "continue", "first_quality_bad")
                            continue
                        traj.stop_reason = "draft_quality_bad_streak"
                        self._decide(pool, traj, entry, "takeover", "quality_bad_streak")
                        self._stop_draft(draft_future, cancel)
                        self._run_fallback_once(
                            fallback_coordinator, image, user_text, pool, traj,
                            CancellableTask(), inject_draft=False,
                            qj_failed_entry=entry, trigger="quality_bad_streak")
                        return

                if traj.final_answer is not None:
                    return
                with state.condition:
                    # Check pool state while holding the condition to avoid a
                    # missed wake-up between snapshot and sleeping.
                    if pool.entry_count > pool_check_index:
                        continue
                    if state.draft_finished or draft_future.done():
                        break
                    state.condition.wait(timeout=0.05)

            # Inherited run cancels after Draft exits.  A just-published final
            # candidate still needs exactly the same verifier as production.
            if traj.final_answer is None and pool.has_answer():
                review_candidate()
        except RequestCancelled:
            exit_reason = "assessment_cancelled"
            cancel.cancel()
        except BaseException:
            exit_reason = "reviewer_exception"
            # A context/verifier failure can happen while Draft is still in
            # its first request.  Abort that request as well as closing any
            # evidence gate in finally, so outer run can reach fallback.
            cancel.cancel()
            raise
        finally:
            if active_request is not None:
                active_request.cancel()
            if active_future is not None:
                active_future.cancel()
            self._close_handshake(
                pool, traj,
                "answer" if traj.final_answer is not None else "stop",
                exit_reason)
