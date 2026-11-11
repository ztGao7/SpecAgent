"""
Agent 推理循环 — 多轮工具调用循环，带每步时间统计。

AgentLoop 是 spectool 的标准推理模式（非推测性执行）。
模型在每个 step 中可以选择：
  1. 输出 <answer> 标签直接回答（若图片可直接看出答案）
  2. 调用工具（sandbox / web_search）收集更多证据
  3. 未按协议输出时继续下一轮

对于每个样本无状态：每次 `run()` 调用完全独立。
"""
import re
import time

from PIL import Image

from .model import (VLLMModel, build_assistant_tool_call_message,
                    build_tool_result_messages, parse_answer,
                    parse_tool_call, pil_to_data_url, strip_reasoning_tags)
from .prompts import (PromptSet, USER_PROMPT_TEMPLATE, get_prompt_set)
from .tools import ToolRegistry, DEFAULT_TOOL_SCHEMAS
from .types import AgentConfig, StepResult, Trajectory


def _truncate_image_history(messages: list, max_images: int) -> None:
    """截断消息历史中的图片数量。

    将超出 max_images 的最旧 image_url 内容替换为文本占位符。
    保留 messages[1] 中的所有原始输入图片，仅裁剪后续工具结果图片。"""
    image_positions = []  # (消息索引, 内容索引)
    original_image_count = 0
    for mi, msg in enumerate(messages):
        content = msg.get("content")
        if not isinstance(content, list):
            continue
        for ci, part in enumerate(content):
            if isinstance(part, dict) and part.get("type") == "image_url":
                if mi == 1:
                    original_image_count += 1
                    continue
                image_positions.append((mi, ci))
    tool_image_budget = max(0, max_images - original_image_count)
    n_excess = len(image_positions) - tool_image_budget
    if n_excess <= 0:
        return
    for mi, ci in image_positions[:n_excess]:
        messages[mi]["content"][ci] = {
            "type": "text", "text": "[之前图片已省略]"}


def _diagnose_tool_call(text: str) -> str:
    """诊断工具调用解析失败的原因。

    如果模型输出了 <tool_call> 标签但无法解析，
    返回具体的错误反馈；否则返回通用的提示。"""
    if "<tool_call>" in text and "</tool_call>" in text:
        return (
            "工具调用 JSON 解析错误：您的 <tool_call> 块不是有效的 "
            'JSON 格式（需要 {"name": ..., "arguments": {...}}）。'
            "请重试或提供 <answer>。")
    return "请调用工具或提供 <answer>。"


def _estimate_message_tokens(messages: list) -> int:
    """Conservatively estimate chat tokens without requiring a tokenizer.

    Search results are mostly English while prompts may contain Chinese, so
    three characters per token is deliberately conservative.  Images are
    counted as a fixed multimodal allowance instead of counting base64 bytes.
    The estimate is only used as an early-finalization guard for MMSearch.
    """
    total = 0
    for message in messages:
        total += 8
        content = message.get("content", "")
        if isinstance(content, str):
            total += (len(content) + 2) // 3
        elif isinstance(content, list):
            for part in content:
                if not isinstance(part, dict):
                    total += (len(str(part)) + 2) // 3
                elif part.get("type") == "text":
                    total += (len(str(part.get("text", ""))) + 2) // 3
                elif part.get("type") == "image_url":
                    total += 4096
        tool_calls = message.get("tool_calls")
        if tool_calls:
            total += (len(str(tool_calls)) + 2) // 3
    return total


def _compact_messages_for_final(
        messages: list, max_context_tokens: int, reserve_tokens: int) -> list:
    """Build a valid, bounded final-answer context from an agent transcript.

    Native assistant/tool call pairs cannot be truncated independently without
    producing an invalid chat template.  Keep the original system + image
    question and flatten prior textual evidence into one user message instead.
    """
    if max_context_tokens <= 0 or len(messages) <= 2:
        return list(messages)

    base = list(messages[:2])
    target_tokens = max(2048, max_context_tokens - max(0, reserve_tokens))
    remaining_tokens = max(256, target_tokens - _estimate_message_tokens(base))
    evidence_parts = []
    for message in messages[2:]:
        role = str(message.get("role", "context")).upper()
        content = message.get("content", "")
        texts = []
        if isinstance(content, str):
            texts.append(content)
        elif isinstance(content, list):
            for part in content:
                if isinstance(part, dict) and part.get("type") == "text":
                    texts.append(str(part.get("text", "")))
        if texts:
            evidence_parts.append(f"[{role}] " + "\n".join(texts))

    evidence = "\n\n".join(evidence_parts)
    max_chars = remaining_tokens * 3
    if len(evidence) > max_chars:
        head_chars = max_chars // 3
        tail_chars = max_chars - head_chars
        evidence = (
            evidence[:head_chars]
            + "\n\n[... middle evidence omitted for context safety ...]\n\n"
            + evidence[-tail_chars:]
        )
    if evidence:
        base.append({
            "role": "user",
            "content": [{
                "type": "text",
                "text": (
                    "[Collected evidence from earlier MMSearch tool calls; "
                    "images and protocol metadata were omitted for context "
                    "safety.]\n\n" + evidence
                ),
            }],
        })
    return base


# 匹配 VLMEvalKit 为选择题数据集构建的提示词中的选项行，
# 如 "A. rubber" / "B) cotton"（见 image_mcq.build_prompt）。
_MCQ_OPTION_RE = re.compile(r"(?m)^\s*\(?([A-Z])[.)]\s+\S")


def _detect_mcq_choices(question: str):
    """检测问题是否为多选题，返回连续的选项字母列表。

    如 ['A','B','C','D']。用于约束强制最终答案为合法字母。"""
    letters = []
    for m in _MCQ_OPTION_RE.finditer(question or ""):
        L = m.group(1)
        if L not in letters:
            letters.append(L)
    expected = [chr(ord("A") + i) for i in range(len(letters))]
    if len(letters) >= 2 and letters == expected:
        return letters
    return None


class AgentLoop:
    """标准 Agent 推理循环（非推测性执行）。

    模型在每个 step 中推理 → 解析 → 决定下一步：
      - 有 <answer> 标签 → 结束并返回答案
      - 有工具调用 → 执行工具，将结果追加到对话
      - 都无 → 提示模型重试
    """

    def __init__(
        self,
        model: VLLMModel,
        tools: ToolRegistry,
        config: AgentConfig,
        prompt_set: PromptSet | None = None,
        tool_schemas: list[dict] | None = None,
    ):
        self.model = model
        self.tools = tools
        self.config = config
        self.prompt_set = prompt_set or get_prompt_set()
        self.tool_schemas = (
            DEFAULT_TOOL_SCHEMAS if tool_schemas is None else tool_schemas
        )

    def run(self, image: Image.Image | list[Image.Image], question: str,
            sample_id: str = "",
            extra_prompt: str = "") -> Trajectory:
        """运行一个样本的推理循环。

        参数:
            image: 一张输入图片，或按题目顺序排列的多张图片
            question: 问题文本
            sample_id: 样本唯一标识
            extra_prompt: 附加的提示词（如 MathVision 专用指令）
        返回:
            Trajectory 推理轨迹
        """
        t_run_start = time.perf_counter()
        input_images = [image] if isinstance(image, Image.Image) else list(image)
        if not input_images or not all(
                isinstance(item, Image.Image) for item in input_images):
            raise ValueError("AgentLoop requires at least one PIL input image")
        primary_image = input_images[0]

        if hasattr(self.tools, "set_input_images"):
            self.tools.set_input_images(input_images)
        else:
            self.tools.set_input_image(primary_image)

        user_text = USER_PROMPT_TEMPLATE.format(question=question)
        if extra_prompt:
            user_text = extra_prompt + "\n\n" + user_text

        user_content = []
        for image_idx, input_image in enumerate(input_images, start=1):
            if len(input_images) > 1:
                user_content.append({
                    "type": "text",
                    "text": f"[输入图片 {image_idx} / sandbox image_{image_idx}]",
                })
            user_content.append({
                "type": "image_url",
                "image_url": {"url": pil_to_data_url(input_image)},
            })
        user_content.append({"type": "text", "text": user_text})

        messages = [
            {"role": "system", "content": self.prompt_set.system_prompt},
            {"role": "user", "content": user_content},
        ]
        traj = Trajectory(sample_id=sample_id)
        # 所有工具调用都基于原始的输入图片。不会将工具输出链式传递给
        # "当前图片"——以往尝试表明，当坐标参考系不断变化时模型会迷失。
        # 工具结果图片仍然追加到对话中以便模型看到刚生成的内容，
        # 但下一次工具调用时仍使用原始 `primary_image` 派发。sandbox
        # 同时可通过 image_1、image_2 等变量访问全部原始输入图片。

        loop_completed_normally = False
        is_mmsearch = self.prompt_set.profile == "mmsearch"
        estimated_output_tokens = 0
        malformed_tool_calls = 0
        answer_only_recovery = False
        for step_idx in range(self.config.max_steps):
            is_last_step = step_idx == self.config.max_steps - 1
            budget_trigger = "max_steps" if is_last_step else ""
            if is_mmsearch and not budget_trigger:
                if (
                    self.config.max_context_tokens > 0
                    and _estimate_message_tokens(messages)
                    + self.config.max_tokens
                    >= self.config.max_context_tokens
                ):
                    budget_trigger = "context_tokens"
                elif (
                    self.config.max_total_output_tokens > 0
                    and estimated_output_tokens + self.config.max_tokens
                    >= self.config.max_total_output_tokens
                ):
                    budget_trigger = "output_tokens"

            must_finalize = bool(budget_trigger)
            if is_mmsearch and must_finalize:
                traj.budget_trigger = budget_trigger
                messages = _compact_messages_for_final(
                    messages,
                    self.config.max_context_tokens,
                    self.config.max_tokens,
                )

            # Only MMSearch disables tools at its budget boundary, so only that
            # profile receives an explicit final-report instruction. Standard
            # Agent runs keep their original conversation on the last step.
            if is_mmsearch and must_finalize:
                final_instruction = (
                    "[SYSTEM] This is your final MMSearch step. Use the "
                    "evidence already gathered to produce a self-contained "
                    "research report. State the direct answer clearly and "
                    "put the entire report inside <answer></answer>. Do not "
                    "call another tool."
                )
                messages.append({
                    "role": "user",
                    "content": [{"type": "text", "text": final_instruction}],
                })

            # === 推理计时 ===
            t0 = time.perf_counter()
            try:
                response = self.model.generate(
                    messages,
                    tools=(None if is_mmsearch and must_finalize
                           or answer_only_recovery else self.tool_schemas),
                )
            except Exception as e:
                if is_mmsearch:
                    traj.budget_trigger = traj.budget_trigger or "model_error"
                    messages = _compact_messages_for_final(
                        messages,
                        self.config.max_context_tokens,
                        self.config.max_tokens,
                    )
                    break
                traj.stop_reason = f"model_error: {type(e).__name__}: {e}"
                loop_completed_normally = True  # 防止被 max_steps 覆盖
                break
            inference_time = time.perf_counter() - t0
            traj.total_inference_time += inference_time
            if isinstance(response, str):
                estimated_output_tokens += max(1, (len(response) + 2) // 3)
                traj.estimated_output_tokens = estimated_output_tokens

            # 原生工具调用优先（vLLM --tool-call-parser）；无则回退 <tool_call> 文本解析
            tool_calls = (
                [] if is_mmsearch and must_finalize
                else list(self.model.last_tool_calls)
            )
            call_ids = (
                [] if is_mmsearch and must_finalize
                else list(self.model.last_tool_call_ids)
            )
            fallback_assistant_msg = None
            if (not tool_calls and not (is_mmsearch and must_finalize)
                    and not answer_only_recovery):
                tc = parse_tool_call(response)
                if tc is not None:
                    tool_calls = [tc]
                    call_ids = [f"call_fallback_{step_idx}_0"]
                    fallback_assistant_msg = build_assistant_tool_call_message(
                        response, tool_calls, call_ids)

            # 1) 无工具调用且有 <answer> → 最终答案
            if not tool_calls:
                answer = parse_answer(
                    response,
                    normalize_mcq=self.prompt_set.normalize_mcq_answers,
                    choices=(_detect_mcq_choices(user_text)
                             if self.prompt_set.normalize_mcq_answers else None),
                    question=user_text)
                if answer is not None:
                    traj.steps.append(StepResult(
                        step_idx=step_idx,
                        reasoning=strip_reasoning_tags(response),
                        raw_response=response,
                        answer=answer,
                        inference_time=inference_time,
                    ))
                    traj.final_answer = answer
                    traj.stop_reason = "answer"
                    loop_completed_normally = True
                    break

            if is_mmsearch and must_finalize:
                # Tools are disabled for this request.  If the model still
                # emits a tool-like or untagged response, do not execute it;
                # the answer-only fallback below gets one bounded attempt.
                messages.append({"role": "assistant", "content": response})
                traj.steps.append(StepResult(
                    step_idx=step_idx,
                    reasoning=strip_reasoning_tags(response),
                    raw_response=response,
                    observation="MMSearch budget reached; tool execution disabled.",
                    inference_time=inference_time,
                ))
                break

            # 回填 assistant 回合（原生模式带 tool_calls，供模板正确格式化）
            messages.append(
                fallback_assistant_msg or self.model.last_assistant_msg)

            if not tool_calls:
                # 无法解析 — 给出提示
                nudge = _diagnose_tool_call(response)
                messages.append({"role": "user", "content": [
                    {"type": "text", "text": nudge}]})
                traj.steps.append(StepResult(
                    step_idx=step_idx,
                    reasoning=strip_reasoning_tags(response),
                    raw_response=response,
                    observation=nudge,
                    inference_time=inference_time,
                ))
                continue

            # 2) Execute only calls with their required arguments. Native
            # parsers can emit sandbox({}) for malformed arguments; preserve
            # the assistant/tool protocol but never invoke the sandbox itself.
            rejected_calls = 0
            for tc, cid in zip(tool_calls, call_ids):
                validation_error = self.tools.validate_call(tc)
                if validation_error is not None:
                    rejected_calls += 1
                    malformed_tool_calls += 1
                    messages.extend(build_tool_result_messages(cid, validation_error, []))
                    traj.steps.append(StepResult(
                        step_idx=step_idx,
                        reasoning=strip_reasoning_tags(response),
                        raw_response=response,
                        tool_call=tc,
                        observation=validation_error,
                        inference_time=inference_time,
                    ))
                    continue
                t1 = time.perf_counter()
                observation, result_imgs = self.tools.dispatch(tc, primary_image)
                tool_time = time.perf_counter() - t1
                traj.total_tool_time += tool_time
                traj.tool_call_count += 1
                messages.extend(
                    build_tool_result_messages(cid, observation, result_imgs))
                traj.steps.append(StepResult(
                    step_idx=step_idx,
                    reasoning=strip_reasoning_tags(response),
                    raw_response=response,
                    tool_call=tc,
                    observation=observation,
                    inference_time=inference_time, tool_time=tool_time,
                    ))

            if rejected_calls:
                if malformed_tool_calls >= 2:
                    answer_only_recovery = True
                    recovery = (
                        "Two sandbox calls were rejected because they lacked "
                        "usable code. Tools are now disabled for this sample. "
                        "Use the original image and question, then submit your "
                        "best final answer strictly as <answer>...</answer>."
                    )
                else:
                    recovery = (
                        "If you need sandbox, retry it using a native function "
                        "call with a non-empty Python 'code' argument. Otherwise "
                        "submit the final answer as <answer>...</answer>."
                    )
                messages.append({"role": "user", "content": [
                    {"type": "text", "text": recovery}]})

            _truncate_image_history(messages, self.config.max_image_history)

        if not loop_completed_normally:
            traj.stop_reason = "max_steps"

        traj.total_time = time.perf_counter() - t_run_start
        return traj
