"""
VDR（Vision-DeepResearch）推测性执行 Agent 循环 — 隔离版本。

此模块是 spectool_agent.speculative_agent 的派生版本，用于驱动 VDR 模型对。
使用 VDR 特有的提示词、解析器和 Draft 模型，确保 VDR 的格式处理
完全隔离，不影响 Qwen3.5 路径。

格式无关的组件（VLLMModel、ToolRegistry、DraftPool、数据类型等）
仍从共享模块导入。
"""
import copy
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, Future
from typing import Optional

from PIL import Image

from .draft_model_vdr import DraftModelVDR
# 格式无关的传输层/辅助函数，与 Qwen3.5 路径共享：
from ..model import VLLMModel, pil_to_data_url, strip_reasoning_tags
from ..pool import DraftPool
from ..tools import ToolRegistry
from ..types import AgentConfig, PoolEntry, StepResult, ToolCall, Trajectory
# VDR 专属的格式层（提示词 + 解析器）— 可独立演化，不影响其他路径：
from .parsers_vdr import parse_answer, parse_fallback_answer, parse_tool_call
from .prompts_vdr import MAIN_SYSTEM_PROMPT, SYSTEM_PROMPT, USER_PROMPT_TEMPLATE


# 展示给 Main 审阅器的 Draft 证据字符预算。
_POOL_REASONING_CHARS = 1500     # Draft 推理文本的最大展示字符数
_POOL_OBSERVATION_CHARS = 2000   # Draft 工具观察结果的最大展示字符数
_POOL_RESPONSE_CHARS = 1000      # Draft 原始响应的最大展示字符数

# Draft 连续截断步数阈值。超过此值说明 draft 失控（如重复循环），
# 提前终止，让 Main 模型立即接管。
_MAX_TRUNCATED_DRAFT_STEPS = 2

# A single weak step should not discard an otherwise useful exploration.
_MAX_CONSECUTIVE_QUALITY_BAD = 2


def _truncate_image_history(messages: list, max_images: int) -> None:
    """截断消息历史中的图片数量，保留原始输入图片。"""
    image_positions = []
    for mi, msg in enumerate(messages):
        content = msg.get("content")
        if not isinstance(content, list):
            continue
        for ci, part in enumerate(content):
            if isinstance(part, dict) and part.get("type") == "image_url":
                if mi == 1 and ci == 0:
                    continue
                image_positions.append((mi, ci))
    n_excess = (len(image_positions) + 1) - max_images
    if n_excess <= 0:
        return
    for mi, ci in image_positions[:n_excess]:
        messages[mi]["content"][ci] = {
            "type": "text", "text": "[之前图片已省略]"}


def _diagnose_tool_call(text: str) -> str:
    """诊断工具调用解析失败的原因，返回中文提示"""
    if "<tool_call>" in text and "</tool_call>" in text:
        return (
            "工具调用 JSON 解析错误：您的 <tool_call> 块不是有效的 "
            'JSON 格式（需要 {"name": ..., "arguments": {...}}）。'
            "请重试或提供 <answer>。")
    return "请调用工具或提供 <answer>。"


class CancellableTask:
    """可取消的任务标记。使用 threading.Event 实现线程安全的取消信号。"""

    def __init__(self):
        self._event = threading.Event()

    def cancel(self):
        """设置取消信号"""
        self._event.set()

    @property
    def cancelled(self) -> bool:
        """是否已被取消"""
        return self._event.is_set()


class VDRSpeculativeAgentLoop:
    """VDR 推测性执行 Agent 循环。

    与 SpeculativeAgentLoop 类似，但使用 VDR 格式的组件：
    DraftModelVDR、VDR 格式的解析器和提示词。
    """

    def __init__(
        self,
        main_model: VLLMModel,
        draft_model: Optional[DraftModelVDR],
        tools: ToolRegistry,
        config: AgentConfig,
    ):
        """
        参数:
            main_model: Main 模型（VDR-30B 等）
            draft_model: VDR Draft 模型（VDR-8B 等）；为 None 时禁用推测性
                执行，Main 模型独立运行完整 Agent 循环（baseline）
            tools: 工具注册表
            config: Agent 配置
        """
        self.main_model = main_model
        self.draft_model = draft_model
        self.tools = tools
        self.config = config
        self._executor = ThreadPoolExecutor(max_workers=3)

    def shutdown(self):
        """关闭线程池"""
        self._executor.shutdown(wait=False)

    def run(self, image: Image.Image, question: str, sample_id: str = "",
            extra_prompt: str = "") -> Trajectory:
        """运行 VDR 推测性执行推理循环。

        Draft 模型在后台执行工具探索；Main 模型审阅并给出答案。
        VDR-Bench 的开放题格式在 Main 模型未给出答案时使用
        _force_open_ended_answer 强制提取。

        参数:
            image: 输入图片
            question: 问题文本
            sample_id: 样本 ID
            extra_prompt: 附加提示词（VDR 不使用）
        返回:
            推理轨迹 Trajectory
        """
        t_run_start = time.perf_counter()

        self.tools.set_input_image(image)

        user_text = USER_PROMPT_TEMPLATE.format(question=question)
        if extra_prompt:
            user_text = extra_prompt + "\n\n" + user_text

        if self.draft_model is None:
            # Baseline（无 draft）：不启动 draft 线程，Main 模型独立运行完整
            # Agent 循环（可调用工具），等价于 Agent-Qwen3.5-27B 的非推测基线。
            # 由于不创建 draft 客户端，也就不会再向不存在的 draft 服务发请求。
            pool = DraftPool()
            traj = Trajectory(sample_id=sample_id)
            cancel = CancellableTask()
            self._run_fallback_agent(image, user_text, pool, traj, cancel)
            if traj.final_answer in (None, "error"):
                # 开放题兜底：用一次受约束的推理提交最终答案，
                # 而不是把样本判为 error。
                self._force_open_ended_answer(image, user_text, pool, traj)
                if traj.final_answer is None:
                    traj.stop_reason = "no_main_answer"
            traj.total_time = time.perf_counter() - t_run_start
            return traj

        draft_initial = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": [
                {"type": "image_url",
                 "image_url": {"url": pil_to_data_url(image)}},
                {"type": "text", "text": user_text},
            ]},
        ]

        pool = DraftPool()
        traj = Trajectory(sample_id=sample_id)
        cancel = CancellableTask()

        draft_future: Future = self._executor.submit(
            self._draft_loop, draft_initial, image, pool, cancel,
        )

        pool_check_index = 0
        loop_completed_normally = False
        consecutive_quality_bad = 0

        # Monitor pool until draft finishes or an answer is accepted
        while not draft_future.done() or pool.entry_count > pool_check_index:
            new_entries = pool.get_new_since(pool_check_index)
            pool_check_index = pool.entry_count

            for entry in new_entries:
                if entry.answer is not None or entry.tool_call is not None:
                    # Judge tool exploration with the VDR-native prompt before
                    # allowing the normal Main review to consume it. Answer
                    # entries are handled by the existing answer path.
                    if (entry.tool_call is not None
                            and self.config.quality_judge_enabled):
                        judge = self._quality_judge(
                            image, user_text, entry, pool.entry_count)
                        self._record_quality_judge(traj, judge)
                        if judge["passed"]:
                            consecutive_quality_bad = 0
                        else:
                            consecutive_quality_bad += 1
                            if consecutive_quality_bad < _MAX_CONSECUTIVE_QUALITY_BAD:
                                continue
                            traj.stop_reason = "draft_quality_bad_streak"
                            cancel.cancel()
                            self._run_fallback_agent(
                                image, user_text, pool, traj,
                                CancellableTask())
                            loop_completed_normally = True
                            break

                    # Run main model to review all pool evidence and produce answer
                    main_msgs = self._build_main_context(
                        image, user_text, pool, pool_check_index)
                    response_text = self.main_model.generate(main_msgs)
                    main_reasoning = strip_reasoning_tags(response_text)
                    main_answer = parse_answer(response_text)
                    if main_answer is None:
                        main_answer = parse_fallback_answer(response_text)
                    if main_answer is not None:
                        traj.final_answer = main_answer
                        traj.stop_reason = "main_answer"
                        traj.steps.append(StepResult(
                            step_idx=pool.entry_count,
                            reasoning=main_reasoning,
                            answer=traj.final_answer,
                        ))
                    # If main model didn't produce an answer, continue waiting
                    if traj.final_answer is not None:
                        cancel.cancel()
                        loop_completed_normally = True
                        break

            if traj.final_answer is not None:
                break

            # Wait a bit for draft to produce more entries
            if not draft_future.done():
                try:
                    draft_future.result(timeout=0.2)
                except (TimeoutError, Exception):
                    pass

        if not loop_completed_normally:
            # Draft exhausted all steps — run main model on full pool as final attempt
            main_msgs = self._build_main_context(
                image, user_text, pool, pool.entry_count)
            response_text = self.main_model.generate(main_msgs)
            main_reasoning = strip_reasoning_tags(response_text)
            main_answer = parse_answer(response_text)
            if main_answer is None:
                main_answer = parse_fallback_answer(response_text)
            if main_answer is not None:
                traj.final_answer = main_answer
                traj.stop_reason = "main_answer_final"
                traj.steps.append(StepResult(
                    step_idx=pool.entry_count,
                    reasoning=main_reasoning,
                    answer=traj.final_answer,
                ))
            if traj.final_answer is None:
                # Open-ended salvage: VDR-Bench answers are free-form, so the
                # MCQ letter fallback recovers nothing. Commit a final answer
                # via one constrained pass instead of scoring the sample as an
                # error. (VDR-only loop — the Qwen3.5 path is unaffected.)
                self._force_open_ended_answer(image, user_text, pool, traj)
                if traj.final_answer is None:
                    traj.stop_reason = "no_main_answer"

        cancel.cancel()

        # Record draft steps in trajectory for debugging / trajectory tools
        for entry in pool.get_all():
            s = StepResult(
                step_idx=entry.step_idx,
                reasoning=entry.reasoning,
                draft_time=entry.draft_time,
            )
            if entry.answer is not None:
                s.answer = entry.answer
            if entry.tool_call is not None:
                s.tool_call = entry.tool_call
                s.observation = entry.tool_observation
            traj.steps.append(s)

        traj.total_time = time.perf_counter() - t_run_start
        return traj

    @staticmethod
    def _format_pool_entry(entry: PoolEntry) -> str:
        """将 PoolEntry 格式化为可读文本，供 Main 模型审阅。"""
        lines = ["[DRAFT POOL] Draft 模型第 {} 步:".format(entry.step_idx)]
        if entry.answer is not None:
            lines.append("  状态: 暂定答案 "
                         "(Draft 的假设 — 未验证，可能错误)")
            lines.append("  Draft 的暂定答案: {}".format(entry.answer))
            lines.append("  Draft 的推理: {}".format(
                entry.reasoning[:_POOL_REASONING_CHARS]))
        elif entry.tool_call is not None:
            lines.append("  状态: 工具调用")
            lines.append("  工具: {}({})".format(
                entry.tool_call.tool_name, entry.tool_call.params))
            if entry.tool_observation:
                lines.append("  观察结果: {}".format(
                    entry.tool_observation[:_POOL_OBSERVATION_CHARS]))
        else:
            lines.append("  状态: 无工具/答案")
            lines.append("  响应: {}".format(
                entry.response_text[:_POOL_RESPONSE_CHARS]))
        return "\n".join(lines)

    def _build_quality_judge_context(self, image, user_text, entry):
        """Build a VDR-format GOOD/BAD prompt for one draft tool step."""
        messages = [
            {"role": "system", "content": MAIN_SYSTEM_PROMPT},
            {"role": "user", "content": [
                {"type": "image_url",
                 "image_url": {"url": pil_to_data_url(image)}},
                {"type": "text", "text": user_text},
            ]},
            {"role": "user", "content": [{
                "type": "text", "text": self._format_pool_entry(entry),
            }]},
        ]
        for img in (entry.tool_result_images or []):
            messages[-1]["content"].append({
                "type": "image_url",
                "image_url": {"url": pil_to_data_url(img)},
            })
        messages.append({"role": "user", "content": [{"type": "text", "text": (
            "QUALITY EVALUATION. Judge only whether the latest Draft tool "
            "step is useful for solving the question. Do not solve the task. "
            "GOOD means the call is relevant and provides any useful clue, "
            "evidence, visual detail, intermediate result, or progress. "
            "Partial, uncertain, or imperfect results are still GOOD. "
            "BAD means only clearly unrelated, empty, or repetitive work. "
            "A tool execution error alone is not a reason to reject the step. "
            "Output exactly GOOD or BAD and nothing else."
        )}]})
        return messages

    def _quality_judge(self, image, user_text, entry, step_idx):
        """Return the Main GOOD probability for a VDR draft tool step."""
        t0 = time.perf_counter()
        try:
            result = self.main_model.judge_good_bad(
                self._build_quality_judge_context(image, user_text, entry))
            return {
                "passed": float(result.get("score", 0.5))
                >= self.config.quality_judge_threshold,
                "result": result,
                "elapsed": time.perf_counter() - t0,
                "step_idx": step_idx,
            }
        except Exception as exc:
            print(f"  [WARN] VDR quality judge failed: {exc}", file=sys.stderr)
            # A judge transport failure must not silently force fallback.
            return {
                "passed": True,
                "result": None,
                "elapsed": time.perf_counter() - t0,
                "step_idx": step_idx,
            }

    @staticmethod
    def _record_quality_judge(traj, outcome):
        result = outcome.get("result")
        if result is None:
            return
        score = float(result.get("score", 0.5))
        passed = bool(outcome.get("passed", True))
        traj.total_main_time += float(outcome.get("elapsed", 0.0))
        traj.steps.append(StepResult(
            step_idx=int(outcome.get("step_idx", 0)),
            reasoning=(
                f"draft_quality_judge raw={result.get('raw', '')!r} "
                f"log_good={result.get('log_good', 0.0):.3f} "
                f"log_bad={result.get('log_bad', 0.0):.3f}"
            ),
            answer=f"quality_{'good' if passed else 'bad'}:{score:.3f}",
            inference_time=float(outcome.get("elapsed", 0.0)),
            verdict_mode="quality_judge",
            judge_score=score,
        ))

    def _build_main_context(self, image, user_text, pool, pool_check_index):
        """构建 Main 模型的消息上下文：系统提示 + 用户（图片+问题）
        + 到目前为止产生的 [DRAFT POOL] 条目。"""
        messages = [
            {"role": "system", "content": MAIN_SYSTEM_PROMPT},
            {"role": "user", "content": [
                {"type": "image_url",
                 "image_url": {"url": pil_to_data_url(image)}},
                {"type": "text", "text": user_text},
            ]},
        ]
        for entry in pool.get_all():
            content = [{"type": "text",
                        "text": self._format_pool_entry(entry)}]
            if entry.tool_result_images:
                for img in entry.tool_result_images:
                    content.append({
                        "type": "image_url",
                        "image_url": {"url": pil_to_data_url(img)},
                    })
            messages.append({"role": "user", "content": content})
        return messages

    def _run_fallback_agent(
        self, image: Image.Image, user_text: str, pool: DraftPool,
        traj: Trajectory, cancel: CancellableTask,
    ) -> None:
        """推测性执行置信度低时，运行 Main 模型作为完整的 Agent（可调用工具）。

        Draft 模型的池条目被注入为先验上下文，
        Main 模型可以调用 sandbox / web_search 收集额外证据。

        此方法始终设置 ``traj.final_answer`` —— 成功时设为答案，
        失败时设为 ``"error"``。
        """
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": [
                {"type": "image_url",
                 "image_url": {"url": pil_to_data_url(image)}},
                {"type": "text", "text": user_text},
            ]},
        ]
        # Inject draft pool entries as prior context
        for entry in pool.get_all():
            content = [{"type": "text",
                        "text": self._format_pool_entry(entry)}]
            if entry.tool_result_images:
                for img in entry.tool_result_images:
                    content.append({
                        "type": "image_url",
                        "image_url": {"url": pil_to_data_url(img)},
                    })
            messages.append({"role": "user", "content": content})
        # Transition message
        messages.append({
            "role": "user",
            "content": [{"type": "text", "text": (
                "[SYSTEM] The draft model's exploration above is provided "
                "as context. You may now call additional tools (sandbox or "
                "web_search) if you need more evidence, or output your final "
                "answer using <answer>ANSWER</answer>."
            )}],
        })

        max_fallback = self.config.max_steps

        for step_idx in range(max_fallback):
            if cancel.cancelled:
                break

            if step_idx == max_fallback - 1:
                messages.append({
                    "role": "user",
                    "content": [{"type": "text", "text": (
                        "[SYSTEM] This is your LAST step. Output your best "
                        "answer using the <answer> tag. For multiple-choice "
                        "questions, output ONLY the single letter (A/B/C/D/E). "
                        "Do NOT output the option text or computed value."
                    )}],
                })

            t0 = time.perf_counter()
            try:
                response_text = self.main_model.generate(messages)
            except Exception as e:
                print(f"  [WARN] Fallback agent failed: {e}", file=sys.stderr)
                break
            inference_time = time.perf_counter() - t0
            traj.total_inference_time += inference_time

            # 1) Answer via <answer> tag?
            answer = parse_answer(response_text)
            if answer is not None:
                traj.final_answer = answer
                traj.stop_reason = "main_answer_fallback"
                traj.steps.append(StepResult(
                    step_idx=step_idx,
                    reasoning=strip_reasoning_tags(response_text),
                    answer=answer,
                    inference_time=inference_time,
                ))
                return

            # 2) Tool call?
            tool_call = parse_tool_call(response_text)
            messages.append({"role": "assistant",
                             "content": [{"type": "text",
                                          "text": response_text}]})

            if tool_call is not None:
                t1 = time.perf_counter()
                observation, result_images = self.tools.dispatch(
                    tool_call, image)
                tool_time = time.perf_counter() - t1
                traj.total_tool_time += tool_time
                traj.tool_call_count += 1

                user_content = [{"type": "text", "text": observation}]
                for img in result_images:
                    user_content.append({
                        "type": "image_url",
                        "image_url": {"url": pil_to_data_url(img)},
                    })
                messages.append({"role": "user", "content": user_content})

                traj.steps.append(StepResult(
                    step_idx=step_idx,
                    reasoning=strip_reasoning_tags(response_text),
                    tool_call=tool_call,
                    observation=observation,
                    inference_time=inference_time,
                    tool_time=tool_time,
                ))
            else:
                # 2a) Bare MCQ answer fallback
                fb = parse_fallback_answer(response_text)
                if fb is not None:
                    traj.final_answer = fb
                    traj.stop_reason = "main_answer_fallback"
                    traj.steps.append(StepResult(
                        step_idx=step_idx,
                        reasoning=strip_reasoning_tags(response_text),
                        answer=fb,
                        inference_time=inference_time,
                    ))
                    return

                nudge = _diagnose_tool_call(response_text)
                messages.append({"role": "user", "content": [
                    {"type": "text", "text": nudge}]})
                traj.steps.append(StepResult(
                    step_idx=step_idx,
                    reasoning=strip_reasoning_tags(response_text),
                    observation=nudge,
                    inference_time=inference_time,
                ))

            _truncate_image_history(messages, self.config.max_image_history)

        # Fallback exhausted without answer
        if traj.final_answer is None:
            traj.final_answer = "error"
            if traj.stop_reason in ("", "main_answer_low_confidence"):
                traj.stop_reason = "fallback_no_answer"

    def _force_open_ended_answer(
        self, image: Image.Image, user_text: str, pool: DraftPool,
        traj: Trajectory,
    ) -> None:
        """为 VDR-Bench 的开放题强制提取最终答案（兜底策略）。

        当 Main 模型从未输出 <answer> 标签时调用。
        开放题的答案无法通过 MCQ 字母 fallback 恢复，
        因此使用约束生成（generate_forced）要求模型只输出最终答案，
        不进行推理、不调用工具。
        """
        messages = self._build_main_context(
            image, user_text, pool, pool.entry_count)
        messages.append({
            "role": "user",
            "content": [{"type": "text", "text": (
                "[SYSTEM] Stop reasoning and do not call any tools. Output ONLY "
                "your single best final answer to the question — a short phrase, "
                "number, or name — with no explanation. Wrap it as "
                "<answer>ANSWER</answer>. If uncertain, still give your most "
                "likely answer."
            )}],
        })

        t0 = time.perf_counter()
        try:
            response_text = self.main_model.generate_forced(messages)
        except Exception as e:
            print(f"  [WARN] VDR forced open-ended answer failed: {e}",
                  file=sys.stderr)
            return
        inference_time = time.perf_counter() - t0
        traj.total_inference_time += inference_time

        answer = parse_answer(response_text)
        if answer is None:
            answer = strip_reasoning_tags(response_text).strip()
        if not answer:
            return

        traj.final_answer = answer
        traj.stop_reason = "forced_open_answer"
        traj.steps.append(StepResult(
            step_idx=pool.entry_count,
            reasoning=strip_reasoning_tags(response_text),
            answer=answer,
            inference_time=inference_time,
        ))

    def _draft_loop(
        self, initial_messages: list, image: Image.Image,
        pool: DraftPool, cancel: CancellableTask,
    ) -> None:
        """Draft 模型的后台推理循环。

        与 SpeculativeAgentLoop._draft_loop 类似，但使用 VDR 格式的解析器。
        """
        draft_msgs = copy.deepcopy(initial_messages)
        stuck_steps = 0  # 连续截断计数

        for step_idx in range(self.config.max_steps):
            if cancel.cancelled:
                break

            # 最后一步：强制输出答案
            if step_idx == self.config.max_steps - 1:
                draft_msgs.append({
                    "role": "user",
                    "content": [{"type": "text",
                                 "text": (
                                     "[SYSTEM] 这是您的最后一步。您已经在之前的步骤中调用了工具。"
                                     "现在必须使用 <answer> 标签输出您的最佳答案。"
                                     "不要再调用任何工具。对于多选题，只输出单个字母 (A/B/C/D/E)。"
                                 )}],
                })

            t0 = time.perf_counter()
            try:
                response_text = self.draft_model.generate_full(draft_msgs)
            except Exception as e:
                print(f"  [WARN] Draft 模型失败: {e}", file=sys.stderr)
                break
            draft_time = time.perf_counter() - t0
            truncated = self.draft_model.last_truncated

            answer = parse_answer(response_text)
            tool_call = parse_tool_call(response_text)

            # 有答案则写入池并退出
            if answer is not None:
                pool.add(PoolEntry(
                    step_idx=step_idx,
                    response_text=response_text,
                    reasoning=strip_reasoning_tags(response_text),
                    answer=answer,
                    draft_time=draft_time,
                ))
                break

            draft_msgs.append({
                "role": "assistant",
                "content": [{"type": "text", "text": response_text}],
            })

            if tool_call is not None:
                # 有工具调用 — 执行并记录结果
                stuck_steps = 0
                observation, result_images = self.tools.dispatch(tool_call, image)
                pool.add(PoolEntry(
                    step_idx=step_idx,
                    response_text=response_text,
                    reasoning=strip_reasoning_tags(response_text),
                    tool_call=tool_call,
                    tool_observation=observation,
                    tool_result_images=result_images,
                    draft_time=draft_time,
                ))

                user_content = [{"type": "text", "text": observation}]
                for img in result_images:
                    user_content.append({"type": "image_url",
                                         "image_url": {"url": pil_to_data_url(img)}})
                draft_msgs.append({"role": "user", "content": user_content})
            else:
                # 没有工具调用 → 尝试 fallback 答案解析
                fb = parse_fallback_answer(response_text)
                if fb is not None:
                    pool.add(PoolEntry(
                        step_idx=step_idx,
                        response_text=response_text,
                        reasoning=strip_reasoning_tags(response_text),
                        answer=fb,
                        draft_time=draft_time,
                    ))
                    break

                # 既无答案也无工具调用
                nudge = _diagnose_tool_call(response_text)
                draft_msgs.append({"role": "user", "content": [
                    {"type": "text", "text": nudge}]})
                pool.add(PoolEntry(
                    step_idx=step_idx,
                    response_text=response_text,
                    reasoning=strip_reasoning_tags(response_text),
                    draft_time=draft_time,
                ))

                # 没有答案，没有工具调用。如果还被截断（重复循环），
                # 累积计数并提前交给 Main 模型。
                if truncated:
                    stuck_steps += 1
                    if stuck_steps >= _MAX_TRUNCATED_DRAFT_STEPS:
                        print(f"  [WARN] Draft 卡住（被截断且无输出 "
                              f"{stuck_steps} 次）；移交给 Main 模型",
                              file=sys.stderr)
                        break
                else:
                    stuck_steps = 0

            _truncate_image_history(draft_msgs, self.config.max_image_history)
