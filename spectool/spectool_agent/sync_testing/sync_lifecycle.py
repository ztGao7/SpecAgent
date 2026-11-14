"""Generated from the current lifecycle; only initial fast scheduling differs."""
import threading
import time
import traceback
from concurrent.futures import Future, TimeoutError
from PIL import Image
from spectool_agent.model import RequestCancellation
from spectool_agent.pool import DraftPool
from spectool_agent.prompts import USER_PROMPT_TEMPLATE
from spectool_agent.types import Trajectory
from spectool_agent.speculative_agent import (
    CancellableTask, FallbackCoordinator, _as_input_images, _original_image_content,
    _fast_answer_enabled, _fast_answer_handoff_grace_seconds, _boundary,
)

def run(self, image: Image.Image | list[Image.Image], question: str,
        sample_id: str = "",
        extra_prompt: str = "",
        fallback_only: bool = False) -> Trajectory:
    """运行推测性执行推理循环。

    Draft 模型在后台执行工具探索；当 Draft 产出结果（答案或工具调用）时，
    Main 模型审阅所有 pool 证据并直接给出最终答案。
    Draft 工具调用由 QJ 审核；Draft 最终答案由 Main review。

    参数:
        image: 一张输入图片，或按题目顺序排列的多张图片
        question: 问题文本
        sample_id: 样本 ID
        extra_prompt: 附加提示词
    返回:
        推理轨迹 Trajectory
    """
    t_run_start = time.perf_counter()
    input_images = _as_input_images(image)
    primary_image = input_images[0]

    traj = Trajectory(sample_id=sample_id, timing_origin=t_run_start)
    if hasattr(self.tools, "set_input_images"):
        self._timed_call(
            traj, "client", "sample_setup",
            self.tools.set_input_images, input_images)
    else:
        self._timed_call(
            traj, "client", "sample_setup",
            self.tools.set_input_image, primary_image)

    user_text = USER_PROMPT_TEMPLATE.format(question=question)
    if extra_prompt:
        user_text = extra_prompt + "\n\n" + user_text

    pool = DraftPool()
    fallback_coordinator = FallbackCoordinator()
    if fallback_only:
        self._run_fallback_once(
            fallback_coordinator, input_images, user_text, pool, traj,
            CancellableTask(), inject_draft=False,
            trigger="fallback_only")
        traj.total_time = time.perf_counter() - t_run_start
        traj.finalize_timing()
        return traj

    context_started = time.perf_counter()
    draft_user_content = _original_image_content(input_images)
    draft_user_content.append({"type": "text", "text": user_text})
    draft_initial = [
        {"role": "system", "content": self.prompt_set.draft_system_prompt},
        {"role": "user", "content": draft_user_content},
    ]
    self._record_timing_span(
        traj, "client", "draft_initial_context", context_started,
        time.perf_counter())
    cancel = CancellableTask()


    synchronous_fast_answer_done = False
    if (_fast_answer_enabled() and self.prompt_set.allow_early_answer):
        fast_started = time.perf_counter()
        fast_request_cancel = RequestCancellation()
        fast_accepted = False
        fast_status = "ok"
        try:
            fast_result = self._run_fast_answer_gate(
                input_images, user_text, traj, fast_request_cancel)
            fast_result = dict(fast_result, evidence_count=0)
            fast_accepted = self._apply_fast_answer_result(
                fast_result, user_text, pool, traj)
        except Exception as exc:
            fast_status = "error"
            _boundary(f"fast_answer_exception type={type(exc).__name__}: {exc}")
        self._sync_event(traj, "fast_answer_gate_complete",
            started_at=fast_started, status=fast_status, accepted=fast_accepted)
        synchronous_fast_answer_done = True
        if fast_accepted:
            traj.total_time = time.perf_counter() - t_run_start
            traj.finalize_timing()
            return traj

    draft_future: Future = self._executor.submit(
        self._draft_loop, draft_initial, primary_image, pool, cancel, traj,
    )

    # A highly conservative Main check races Draft only on the original
    # task. Uncertain samples continue through the normal QJ path below.
    fast_request_cancel = None
    fast_future = None
    if (not synchronous_fast_answer_done and _fast_answer_enabled()
            and self.prompt_set.allow_early_answer):
        fast_request_cancel = RequestCancellation()
        fast_future = self._executor.submit(
            self._run_fast_answer_gate, input_images, user_text, traj,
            fast_request_cancel)

    fast_lock = threading.Lock()
    fast_consumed = False

    def consume_fast_answer(*, handoff=False):
        nonlocal fast_consumed
        # Polling and reviewer handoff share one result owner. Closing the
        # gate before Main work also prevents a late response overwriting it.
        with fast_lock:
            if fast_future is None or fast_consumed:
                return traj.stop_reason == "fast_main_answer"
            if not handoff and not fast_future.done():
                return False
            try:
                result = fast_future.result(
                    timeout=(
                        _fast_answer_handoff_grace_seconds()
                        if handoff else None))
            except TimeoutError:
                fast_consumed = True
                if fast_request_cancel is not None:
                    fast_request_cancel.cancel()
                fast_future.cancel()
                return False
            except Exception as exc:
                fast_consumed = True
                _boundary(f"fast_answer_exception type={type(exc).__name__}: {exc}")
                return False
            fast_consumed = True
            # This request sees only original inputs, never pool evidence.
            result = dict(result, evidence_count=0)
            accepted = self._apply_fast_answer_result(
                result, user_text, pool, traj)
            if accepted:
                self._stop_draft(draft_future, cancel)
            return accepted

    loop_completed_normally = False

    # Run main reviewer in separate thread — does NOT block draft
    review_future: Future = self._executor.submit(
        self._review_loop, input_images, user_text, pool, traj, cancel,
        draft_future, fallback_coordinator,
        fast_future, fast_request_cancel, consume_fast_answer,
    )

    # Do not wait for an already-running Draft HTTP request after Main has
    # safely accepted a direct answer; Draft cancellation is cooperative.
    fast_answer_taken = False
    try:
        while not draft_future.done():
            if consume_fast_answer():
                fast_answer_taken = True
                break
            time.sleep(0.01)
        if not fast_answer_taken:
            draft_future.result()
    except Exception as e:
        _boundary(f"draft_future_exception type={type(e).__name__}: {e}")
        traceback.print_exc()
    # Draft completion must not skip an already-finished initial result.
    fast_answer_taken = consume_fast_answer(handoff=True) or fast_answer_taken
    self._merge_draft_pool_into_traj(pool, traj)

    if fast_answer_taken:
        try:
            review_future.result(timeout=5.0)
        except TimeoutError:
            _boundary("review_cancel_timeout after fast answer")
        except Exception as exc:
            _boundary(f"review_cancel_exception type={type(exc).__name__}: {exc}")
        traj.total_time = time.perf_counter() - t_run_start
        traj.finalize_timing()
        return traj

    if fast_future is not None and not fast_future.done():
        fast_request_cancel.cancel()
        fast_future.cancel()

    # If review hasn't approved yet, wait for it
    if traj.final_answer is None:
        cancel.cancel()  # tell reviewer to do final review
        # The reviewer may already own a long-running fallback. Waiting
        # for it avoids launching a second Main/tool trajectory after a
        # short timeout while the first one is still active.
        try:
            review_future.result()
        except Exception as e:
            _boundary(f"review_future_exception type={type(e).__name__}: {e}")
            traceback.print_exc()
        self._merge_draft_pool_into_traj(pool, traj)

    loop_completed_normally = traj.final_answer is not None

    # If speculative work ended without an answer, let Main take over the
    # validated state with only the remaining shared step budget.
    if not loop_completed_normally:
        # `cancel` has already been used to stop the reviewer thread.
        # Reusing the same cancelled token would make fallback exit
        # immediately, yielding an empty trajectory for every sample.
        fallback_cancel = CancellableTask()
        fallback_trigger = (
            "draft_answer_rejected"
            if traj.stop_reason == "draft_answer_evidence_rejected"
            else "speculative_exhausted")
        self._run_fallback_once(
            fallback_coordinator, input_images, user_text, pool, traj,
            fallback_cancel, trigger=fallback_trigger)

    traj.total_time = time.perf_counter() - t_run_start
    traj.finalize_timing()
    return traj
