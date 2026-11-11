"""
Draft 模型封装 — 推测性执行中的轻量级探索模型。

Draft 模型是一个较小的模型（如 Qwen3.5-9B），与 Main 模型并行运行。
它在后台执行工具调用和推理，将中间结果写入 DraftPool。
Main 模型（较大的模型，如 Qwen3.5-27B）审阅 Draft 的产出并做出最终决策。

设计要点：
  - 使用与 Main 模型相同的 SYSTEM_PROMPT，保持行为一致
  - 可选的图片降采样（draft_max_image_size > 0）以加速推理
  - 轻微的 repetition_penalty（1.1）防止贪心解码时的重复循环
  - 检测 max_tokens 截断以识别失控的推理
"""
import asyncio
import threading

from openai import AsyncOpenAI, OpenAI

from .model import (downscale_image_messages, _parse_native_tool_calls,
                    _assistant_message_dict, _bounded_max_tokens, _env_int,
                    RequestCancellation, RequestCancelled, _model_response)
from .prompts import SYSTEM_PROMPT
from .types import DEFAULT_MAX_TOKENS


class DraftModel:
    """Draft 模型封装，用于推测性执行中的后台探索。

    Draft 模型异步运行，执行工具调用收集证据，
    并将每一步的产出（PoolEntry）写入共享的 DraftPool。
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
        self._api_base = api_base
        self._api_key = api_key
        self._max_retries = max(0, _env_int("SPECTOOL_OPENAI_MAX_RETRIES", 0))
        self._client = OpenAI(
            base_url=api_base, api_key=api_key, max_retries=self._max_retries)
        self._model_name = model_name
        self._timeout = timeout
        self._max_image_size = max_image_size
        # EvalKit shares one DraftModel across multiple sample worker threads.
        # Generation metadata must stay with the thread that produced it.
        self._generation_state = threading.local()

    def _reset_generation_state(self) -> None:
        state = self._generation_state
        state.truncated = False
        state.tool_calls = []
        state.tool_call_ids = []
        state.assistant_msg = {"role": "assistant", "content": ""}

    async def _create_cancellable_completion(
        self, request_kwargs: dict, cancel_token: RequestCancellation,
    ):
        """Issue a Draft request whose HTTP task can be aborted in flight."""
        client = AsyncOpenAI(
            base_url=self._api_base,
            api_key=self._api_key,
            max_retries=self._max_retries,
        )
        loop = asyncio.get_running_loop()
        task = asyncio.current_task()

        def cancel_task() -> None:
            if task is not None:
                loop.call_soon_threadsafe(task.cancel)

        cancel_token.register(cancel_task)
        try:
            response = await client.chat.completions.create(**request_kwargs)
            if cancel_token.cancelled:
                raise RequestCancelled("Draft request cancelled")
            return response
        except asyncio.CancelledError as exc:
            raise RequestCancelled("Draft HTTP task cancelled") from exc
        finally:
            cancel_token.unregister(cancel_task)
            try:
                await client.close()
            except asyncio.CancelledError:
                pass

    def generate_full(self, messages: list[dict], max_tokens: int = DEFAULT_MAX_TOKENS,
                      tools: list | None = None,
                      cancel_token: RequestCancellation | None = None) -> str:
        """生成完整的模型响应。

        保留调用方提供的 system prompt；当调用方没有提供时，使用默认
        SYSTEM_PROMPT。
        返回包含推理文本和可选的 <tool_call> / <answer> 标签的完整文本。

        参数:
            messages: 对话消息列表
            max_tokens: 最大生成 token 数
        返回:
            模型输出的完整文本（推理 + 内容）
        """
        self._reset_generation_state()
        full_msgs = list(messages)
        if not any(message.get("role") == "system" for message in full_msgs):
            full_msgs.insert(0, {"role": "system", "content": SYSTEM_PROMPT})

        if self._max_image_size > 0:
            full_msgs = downscale_image_messages(full_msgs, self._max_image_size)

        create_kwargs = dict(
            model=self._model_name,
            messages=full_msgs,
            max_tokens=_bounded_max_tokens(
                max_tokens, default=1024,
                cap_env="SPECTOOL_DRAFT_MAX_TOKENS_CAP", default_cap=0),
            temperature=0.0,
            timeout=self._timeout,
            # 贪心（temp=0）解码在困难的视觉样本上可能退化为 token 级
            # 重复循环，消耗全部 max_tokens 预算。轻微的 repetition_penalty
            # 可在 GPU 采样器上打破循环且无额外延迟。
            # （注：曾试 enable_thinking=False 想逼 draft 多调工具，A/B 实测
            #  工具调用未升 0.49→0.46、acc 未动，已回退；改用强制首步网格检视。）
            extra_body={"repetition_penalty": 1.1},
        )
        if tools:
            create_kwargs["tools"] = tools
            create_kwargs["tool_choice"] = "auto"
        if cancel_token is None:
            resp = self._client.chat.completions.create(**create_kwargs)
        else:
            resp = asyncio.run(self._create_cancellable_completion(
                create_kwargs, cancel_token))
        # 检测响应是否因 max_tokens 被截断。
        # 截断意味着 draft 失控（如重复循环）且未生成可用的答案/工具调用。
        state = self._generation_state
        state.truncated = (
            resp.choices[0].finish_reason == "length")
        msg = resp.choices[0].message
        _native = _parse_native_tool_calls(msg)
        state.tool_calls = [tc for tc, _ in _native]
        state.tool_call_ids = [cid for _, cid in _native]
        state.assistant_msg = _assistant_message_dict(msg)
        return _model_response(resp.choices[0])

    @property
    def last_truncated(self) -> bool:
        """最近一次 generate_full() 是否因 max_tokens 限制被截断"""
        state = getattr(self, "_generation_state", None)
        return getattr(state, "truncated", False) if state is not None else False

    @property
    def last_tool_calls(self) -> list:
        """最近一次 generate_full() 的原生工具调用（list[ToolCall]）。"""
        state = getattr(self, "_generation_state", None)
        return getattr(state, "tool_calls", []) if state is not None else []

    @property
    def last_tool_call_ids(self) -> list:
        """与 last_tool_calls 对应的 tool_call_id（list，元素可为 None）。"""
        state = getattr(self, "_generation_state", None)
        return getattr(state, "tool_call_ids", []) if state is not None else []

    @property
    def last_assistant_msg(self) -> dict:
        """最近一次响应转成的 assistant 消息 dict（含 tool_calls）。"""
        state = getattr(self, "_generation_state", None)
        if state is None:
            return {"role": "assistant", "content": ""}
        return getattr(state, "assistant_msg",
                       {"role": "assistant", "content": ""})
