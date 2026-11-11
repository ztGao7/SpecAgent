"""Shared agent-loop protocol messages.

Keep correction and final-step instructions byte-identical across the full
agent and Spectool's clean Main fallback so they are directly comparable.
"""


def protocol_retry_nudge(text: str) -> str:
    """Return the single corrective prompt for an unparsable model turn."""
    answer_rule = (
        "最终答案必须写成 <answer>答案</answer>；"
        "多选题不能裸输出选项字母。"
    )
    if "<tool_call>" in text and "</tool_call>" in text:
        return (
            "工具调用 JSON 解析错误：您的 <tool_call> 块不是有效的 "
            'JSON 格式（需要 {"name": ..., "arguments": {...}}）。'
            "请重试或提供 <answer>。" + answer_rule
        )
    return "请调用工具或提供 <answer>。" + answer_rule


def final_step_instruction(*, tool_call_count: int,
                           allow_early_answer: bool) -> str:
    """Return the common final regular-step instruction for an agent loop."""
    if not allow_early_answer:
        return (
            "[SYSTEM] This is your final MMSearch step. Use the "
            "evidence already gathered to produce a self-contained "
            "research report. State the direct answer clearly and "
            "put the entire report inside <answer></answer>. Do not "
            "call another tool."
        )
    evidence_note = (
        "请根据之前工具调用收集到的证据作答。"
        if tool_call_count
        else "如果没有工具证据，请根据原始输入作出最佳判断。"
    )
    return (
        "[SYSTEM] 这是您的最后一步。"
        f"{evidence_note}"
        "现在必须使用 <answer> 标签输出您的最佳答案。"
        "不要再调用任何工具。"
    )
