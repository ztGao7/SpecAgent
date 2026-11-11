"""
VDR 专用的 Draft 模型封装。

与 Qwen3.5 的 DraftModel 不同，VDR 的 DraftModel 额外支持：
  - predict_next_tool: 预测 Main 模型的下一个工具调用（用于推测性执行）
  - 使用 structured_outputs 约束输出符合 JSON Schema 的工具调用格式
"""
import json
from typing import Optional

from openai import OpenAI

from ..model import downscale_image_messages  # 格式无关的共享函数
from .parsers_vdr import parse_answer, parse_tool_call, parse_fallback_answer
from .prompts_vdr import DRAFT_SYSTEM_PROMPT, SYSTEM_PROMPT
from ..types import ToolCall


class DraftModelVDR:
    """VDR 专用的 Draft 模型封装。

    除了标准的 generate_full 方法外，还提供 predict_next_tool 方法，
    用于在推测性执行中预测 Main 模型的下一个工具调用。
    """

    def __init__(self, api_base: str, model_name: str, api_key: str = "EMPTY",
                 timeout: int = 200, max_image_size: int = 448):
        """
        参数:
            api_base: OpenAI 兼容 API 的基础 URL
            model_name: 模型名称
            api_key: API 密钥
            timeout: 请求超时（秒）
            max_image_size: 图片降采样的最大尺寸（0=不降采样）
        """
        self._client = OpenAI(base_url=api_base, api_key=api_key)
        self._model_name = model_name
        self._timeout = timeout
        self._max_image_size = max_image_size

    def generate_full(self, messages: list[dict], max_tokens: int = 4096) -> str:
        """生成完整的模型响应（推理 + 工具调用/答案标签）。

        使用与 Main 模型相同的 SYSTEM_PROMPT 保持行为一致。
        可选的图片降采样加速推理。
        轻微的 repetition_penalty 防止贪心解码时的重复循环。
        """
        full_msgs = [{"role": "system", "content": SYSTEM_PROMPT}]
        start_idx = 1 if (messages and messages[0].get("role") == "system") else 0
        full_msgs.extend(messages[start_idx:])

        if self._max_image_size > 0:
            full_msgs = downscale_image_messages(full_msgs, self._max_image_size)

        resp = self._client.chat.completions.create(
            model=self._model_name,
            messages=full_msgs,
            max_tokens=max_tokens,
            temperature=0.0,
            timeout=self._timeout,
            # 贪心解码在困难样本上可能退化为重复循环，
            # repetition_penalty 可在 GPU 上打破循环且无额外延迟。
            extra_body={"repetition_penalty": 1.1},
        )
        # 检测是否因 max_tokens 被截断
        self._last_truncated = (
            resp.choices[0].finish_reason == "length")
        msg = resp.choices[0].message
        raw = msg.model_dump() if hasattr(msg, "model_dump") else {}
        reasoning = raw.get("reasoning") or raw.get("reasoning_content") or ""
        content = msg.content or ""
        if reasoning:
            return reasoning + content
        return content

    @property
    def last_truncated(self) -> bool:
        """最近一次 generate_full() 是否因 max_tokens 限制被截断"""
        return getattr(self, "_last_truncated", False)

    def predict_next_tool(self, main_messages: list[dict]) -> tuple:
        """预测 Main 模型的下一个工具调用。

        使用 DRAFT_SYSTEM_PROMPT（工具调用预测器提示词）引导 Draft 模型
        基于对话历史预测 Main 模型下一步的工具调用。

        返回:
            (ToolCall or None, 原始响应文本)
        """
        draft_msgs = [{"role": "system", "content": DRAFT_SYSTEM_PROMPT}]
        start_idx = 1 if (main_messages and main_messages[0].get("role") == "system") else 0
        draft_msgs.extend(main_messages[start_idx:])
        draft_msgs.append({
            "role": "user",
            "content": [{"type": "text", "text": (
                "基于历史记录，尤其是上一轮的记录，预测下一个工具调用。"
            )}],
        })

        if self._max_image_size > 0:
            draft_msgs = downscale_image_messages(draft_msgs, self._max_image_size)

        response = self._generate(draft_msgs)
        return self._parse_prediction(response), response

    def _generate(self, messages: list[dict]) -> str:
        """使用 structured_outputs 约束生成工具调用 JSON。"""
        resp = self._client.chat.completions.create(
            model=self._model_name,
            messages=messages,
            max_tokens=2048,
            temperature=0.0,
            timeout=self._timeout,
            extra_body={
                "chat_template_kwargs": {"enable_thinking": False},
                "structured_outputs": {
                    "json": {
                        "type": "object",
                        "properties": {
                            "tool_name": {"type": "string", "enum": ["sandbox", "web_search"]},
                            "params": {
                                "type": "object",
                                "properties": {
                                    "box": {"type": "array", "items": {"type": "number"}, "minItems": 4, "maxItems": 4},
                                    "code": {"type": "string"},
                                    "query": {"type": "string"},
                                },
                            },
                        },
                        "required": ["tool_name", "params"],
                        "additionalProperties": False,
                    },
                },
            },
        )
        content = resp.choices[0].message.content
        if not content:
            extra = resp.choices[0].message.model_dump() if hasattr(
                resp.choices[0].message, "model_dump") else {}
            content = extra.get("reasoning") or extra.get("reasoning_content") or ""
        return content.strip()

    def _parse_prediction(self, response: str) -> Optional[ToolCall]:
        """解析工具调用预测结果。

        尽管使用了结构化输出，某些 Draft 模型仍然输出 <tool_call> 标签格式。
        优先从标签提取，回退到直接 JSON 解析。
        支持 draft 格式（tool_name/params）和 main 格式（name/arguments）。
        """
        # 优先检查 <tool_call> 标签
        tc_start = response.find("<tool_call>")
        tc_end = response.find("</tool_call>")
        if tc_start != -1 and tc_end != -1:
            json_str = response[tc_start + len("<tool_call>"):tc_end].strip()
        else:
            start = response.find("{")
            end = response.rfind("}")
            if start == -1 or end == -1:
                return None
            json_str = response[start:end + 1]
        if not json_str:
            return None
        try:
            data = json.loads(json_str)
        except json.JSONDecodeError:
            return None
        # 兼容两种格式：draft 的 tool_name/params 和 main 的 name/arguments
        tool_name = data.get("tool_name") or data.get("name") or ""
        params = data.get("params") or data.get("arguments") or {}
        if not tool_name or not isinstance(tool_name, str):
            return None
        if not isinstance(params, dict):
            return None
        return ToolCall(tool_name=tool_name, params=params)
