"""
vLLM 客户端 + 无状态解析器。

提供与 vLLM（兼容 OpenAI API）的通信层，
以及纯函数的文本解析工具（答案提取、工具调用解析、JSON 修复等）。
解析器设计为模块级函数，以便无需模型服务即可轻松测试。

核心功能：
  - VLLMModel: vLLM 模型的封装，支持常规生成和约束生成
  - 答案解析：从模型输出中提取 <answer> 标签内内容
  - 工具调用解析：支持 Qwen3 XML 和历史 JSON fallback
  - QJ/一致性评估：基于 logprobs 的质量与答案一致性判断
  - 图片处理：模型媒体 URL、缓存、降采样、消息级降采样
"""
import asyncio
import ast
import base64
import hashlib
import io
import json
import math
import os
import re
import sys
import tempfile
import threading
import time
from pathlib import Path
from typing import Callable, Optional
from urllib.parse import unquote, urlparse

from PIL import Image

from .types import AgentConfig, ToolCall


def _boundary(message: str) -> None:
    if os.environ.get("SPECTOOL_BOUNDARY_DEBUG") == "1":
        print(f"[BOUNDARY] t={time.monotonic():.6f} pid={os.getpid()} {message}", file=sys.stderr, flush=True)


# 匹配 <answer>...</answer> 标签内的答案内容
_ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)
# 匹配 <tool_call>...</tool_call> 标签内的工具调用载荷
_TOOL_CALL_RE = re.compile(r"<tool_call>(.*?)</tool_call>", re.DOTALL)
_XML_FUNCTION_RE = re.compile(
    r"<function=([^>\s]+)>\s*(.*?)</function>", re.DOTALL)
_XML_PARAMETER_RE = re.compile(
    r"<parameter=([^>\s]+)>\s*(.*?)\s*</parameter>", re.DOTALL)
# 剥离 <answer> 和 <tool_call> 标签，保留纯推理文本
_STRIP_TAGS_RE = re.compile(r"<(?:answer|tool_call)>.*?</(?:answer|tool_call)>\s*",
                            re.DOTALL)
# 匹配模型输出的纯 MCQ 答案（未用 <answer> 标签包裹）。
# 接受单个大写字母 A-Z，可选地后跟句号/括号/空格和简短理由。
# 故意不使用 re.DOTALL 并且限制了尾部理由长度：
# 之前使用 DOTALL + 无界 ``.*$`` 时会误匹配任何文本的首个大写字母
# （如 "I need to figure out..." 中的 "I"），然后将整个多段响应
# 当作"理由"吞掉，将自由推理错误识别为确定性的 MCQ 答案。
_FALLBACK_ANSWER_RE = re.compile(
    r"^\s*\(?([A-Z])\)?(?:[.\s):：-].{0,160})?$")
# 匹配多行响应末尾的 "(X)" 或 "(X)." 模式
_ANSWER_TAIL_RE = re.compile(r'\(([A-Z])\)[.)\s]*$')
# 匹配显式答案短语："answer is A" / "option: B" / "final answer C"
# 关键：捕获的字母后面不能继续接字母，避免把 "answer is Since..."
# 里的 S 或 "answer is It..." 里的 I 误识别成选项。
_ANSWER_PHRASE_RE = re.compile(
    r'(?:(?:final\s+)?answer|option)\s*(?:is|:|：)?\s*\(?([A-Z])\)?'
    r'(?=\s*[\).,:：，-]?(?:\s|$))', re.IGNORECASE)
# 匹配答案开头就是孤立选项字母的情形："A" / "(B)" / "C. ..."
# 不再允许缺省地吞掉任意首字母开头的自由文本。
_ANSWER_PREFIX_RE = re.compile(
    r"^\s*\(?([A-Z])\)?(?=\s*[\).,:：，-]?(?:\s|$))")
_FALLBACK_PREFIX_RE = re.compile(
    r"^\s*\(?([A-Z])\)?(?=\s*(?:[\).,:：，-]|$))")


def _bounded_max_tokens(value: int, *, default: int = 2048,
                        cap_env: str = "SPECTOOL_MAX_TOKENS_CAP",
                        default_cap: int = 4096) -> int:
    """Apply an explicit role cap; legacy forced generation retains its cap."""
    try:
        tokens = int(value)
    except (TypeError, ValueError):
        tokens = default
    if tokens < 1:
        tokens = default
    try:
        cap = int(os.environ.get(cap_env, str(default_cap)))
    except ValueError:
        cap = default_cap
    if cap > 0:
        tokens = min(tokens, cap)
    return max(1, tokens)


def _extract_mcq_letter(text: str) -> Optional[str]:
    """Extract an explicit MCQ option letter from answer-like text.

    This intentionally only accepts letters at the beginning of the answer
    payload or after an explicit "answer/option" phrase, so ordinary reasoning
    sentences do not accidentally become choices.
    """
    if not text:
        return None
    stripped = str(text).strip()

    m = _ANSWER_PREFIX_RE.match(stripped)
    if m:
        return m.group(1).upper()

    m = _ANSWER_TAIL_RE.search(stripped)
    if m:
        return m.group(1).upper()

    m = _ANSWER_PHRASE_RE.search(stripped)
    if m:
        return m.group(1).upper()

    return None


class ModelResponse(str):
    """String-compatible trace with separate, authoritative final content.

    Keep the historical full text for logging/context. Protocol parsers must
    use ``content``, never tags quoted in the private reasoning channel.
    """

    def __new__(cls, content: str, reasoning: str = "", finish_reason=None):
        content = content if isinstance(content, str) else ""
        reasoning = reasoning if isinstance(reasoning, str) else ""
        obj = super().__new__(cls, reasoning + content)
        obj.content = content
        obj.reasoning = reasoning
        obj.finish_reason = finish_reason if isinstance(finish_reason, str) else None
        return obj


def _model_response(choice) -> ModelResponse:
    message = choice.message
    raw = message.model_dump() if hasattr(message, "model_dump") else {}
    return ModelResponse(
        message.content or "",
        raw.get("reasoning") or raw.get("reasoning_content") or "",
        getattr(choice, "finish_reason", None),
    )


def _protocol_content(text: str) -> str:
    text = text.content if isinstance(text, ModelResponse) else text
    if not isinstance(text, str):
        return ""
    # Also support endpoints that prefix content with explicit think blocks.
    # Do not rewrite tag literals inside a tool's JSON/code payload.
    while re.match(r"\s*<think>", text, re.IGNORECASE):
        block = re.match(r"\s*<think>.*?</think>", text, re.DOTALL | re.IGNORECASE)
        if block is None:
            return ""  # An unfinished reasoning block is not final output.
        text = text[block.end():]
    return text


def validate_final_answer(answer, *, choices=None, question: str = "") -> Optional[str]:
    """Validate output syntax, using only the question/option schema, not gold.

    Numeric/list constraints are taken only from explicit MathVista-style
    answer-format hints. Never infer a type merely because a question contains
    numbers, round a model value, or extract one number from multiple values.
    """
    if not isinstance(answer, str):
        return None
    answer = answer.strip()
    if (not answer or answer in {"...", "…", "YOUR ANSWER", "YOUR COMPLETE ANSWER",
                                 "INSUFFICIENT_EVIDENCE"}
            or re.search(r"</?(?:answer|think|tool_call)\b", answer, re.IGNORECASE)):
        return None
    if choices:
        match = re.fullmatch(r"(?:([A-Za-z])[.)]?|\(([A-Za-z])\)\.?)", answer)
        letter = (match.group(1) or match.group(2)).upper() if match else None
        if letter is None:
            # Preserve harmless "B. Blue" formatting only when the complete
            # suffix matches B's actual option text. Never turn "A or B" or
            # arbitrary reasoning starting with a letter into a submission.
            labeled = re.fullmatch(r"(?:([A-Za-z])[.)]|\(([A-Za-z])\))\s+(.+)", answer)
            if labeled:
                candidate = (labeled.group(1) or labeled.group(2)).upper()
                texts = re.findall(
                    r"(?m)^\s*\(?" + re.escape(candidate) + r"[.)]\s+([^\r\n]+)", question or "")
                if len(texts) == 1 and labeled.group(3).strip().casefold() == texts[0].strip().casefold():
                    letter = candidate
        return letter if letter in choices else None
    hint = re.search(
        r"Hint:\s*Please answer the question requiring (?:an?|the) "
        r"(integer answer|floating-point number|Python list)", question or "", re.IGNORECASE)
    if hint:
        kind = hint.group(1).lower()
        if kind == "integer answer":
            if not re.fullmatch(r"[+-]?\d+", answer):
                return None
        elif kind == "floating-point number":
            if not re.fullmatch(r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?", answer):
                return None
            if not math.isfinite(float(answer)):
                return None
        else:
            try:
                value = ast.literal_eval(answer)
            except (ValueError, SyntaxError, RecursionError):
                return None
            if not isinstance(value, list):
                return None
    return answer


def parse_answer(text: str, normalize_mcq: bool = True, *,
                 choices=None, question: Optional[str] = None) -> Optional[str]:
    """Extract exactly one well-formed answer from final content only."""
    if isinstance(text, ModelResponse) and text.finish_reason == "length":
        return None
    content = _protocol_content(text)
    if content.count("<answer>") != 1 or content.count("</answer>") != 1:
        return None
    m = _ANSWER_RE.search(content)
    if m is None:
        return None
    answer = m.group(1).strip()
    if choices:
        return validate_final_answer(answer, choices=choices, question=question or "")
    answer = validate_final_answer(answer, question=question or "")
    # Preserve the legacy standalone parser contract. Task-aware callers do
    # not normalize prose/numeric answers into an arbitrary option letter.
    if answer is not None and normalize_mcq and question is None:
        return _extract_mcq_letter(answer) or answer
    return answer


def parse_fallback_answer(text: str) -> Optional[str]:
    """Bare answers are intentionally ignored; only <answer> tags count."""
    return None


def strip_reasoning_tags(text: str) -> str:
    """移除文本中的所有 <answer> 和 <tool_call> 标签，保留纯推理文本"""
    return _STRIP_TAGS_RE.sub("", text).strip()


def _repair_json(text: str) -> str:
    """修复被截断的 JSON 字符串：补充缺失的闭合括号和方括号。
    会考虑字符串字面量和转义序列。"""
    depth = {"{": 0, "[": 0}
    in_string = False
    escape = False
    for ch in text:
        if escape:
            escape = False
            continue
        if ch == "\\":
            escape = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch == "{":
            depth["{"] += 1
        elif ch == "}":
            depth["{"] -= 1
        elif ch == "[":
            depth["["] += 1
        elif ch == "]":
            depth["["] -= 1
    result = text
    result += "]" * max(0, depth["["])
    result += "}" * max(0, depth["{"])
    return result


def parse_tool_call(text: str) -> Optional[ToolCall]:
    """解析 Qwen3 XML 工具调用，并兼容旧的 JSON 载荷。"""
    text = _protocol_content(text)
    m = _TOOL_CALL_RE.search(text)
    if not m:
        return None
    payload = m.group(1).strip()

    xml_match = _XML_FUNCTION_RE.search(payload)
    if xml_match:
        name = xml_match.group(1).strip()
        params = {
            param.group(1).strip(): param.group(2).strip()
            for param in _XML_PARAMETER_RE.finditer(xml_match.group(2))
        }
        return ToolCall(tool_name=name, params=params)

    try:
        data = json.loads(payload, strict=False)
    except json.JSONDecodeError:
        repaired = _repair_json(payload)
        if repaired == payload:
            return None
        try:
            data = json.loads(repaired, strict=False)
        except json.JSONDecodeError:
            return None
    name = data.get("name")
    if not isinstance(name, str):
        return None
    args = data.get("arguments", {})
    if not isinstance(args, dict):
        args = {}
    return ToolCall(tool_name=name, params=args)


def build_assistant_tool_call_message(
        response: str, tool_calls: list[ToolCall], call_ids: list[str]) -> dict:
    """Convert text-parsed tool calls back to canonical OpenAI messages.

    Qwen3's chat template renders this structure using its trained XML format.
    This avoids feeding legacy JSON ``<tool_call>`` text back into the next turn.
    """
    if len(tool_calls) != len(call_ids):
        raise ValueError("tool_calls and call_ids must have the same length")
    calls = []
    for tool_call, call_id in zip(tool_calls, call_ids):
        calls.append({
            "id": call_id,
            "type": "function",
            "function": {
                "name": tool_call.tool_name,
                "arguments": json.dumps(
                    tool_call.params, ensure_ascii=False, separators=(",", ":")),
            },
        })
    return {
        "role": "assistant",
        "content": strip_reasoning_tags(response),
        "tool_calls": calls,
    }


def _parse_native_tool_calls(message):
    """从 OpenAI 响应 message 提取原生 tool_calls → [(ToolCall, call_id), ...]。
    供 vLLM --tool-call-parser 的结构化工具调用使用；无则返回空列表。"""
    out = []
    tcs = getattr(message, "tool_calls", None) or []
    for tc in tcs:
        fn = getattr(tc, "function", None)
        if fn is None:
            continue
        name = getattr(fn, "name", None)
        if not isinstance(name, str):
            continue
        raw_args = getattr(fn, "arguments", "") or "{}"
        try:
            args = json.loads(raw_args, strict=False)
        except json.JSONDecodeError:
            try:
                args = json.loads(_repair_json(raw_args), strict=False)
            except json.JSONDecodeError:
                args = {}
        if not isinstance(args, dict):
            args = {}
        out.append((ToolCall(tool_name=name, params=args), getattr(tc, "id", None)))
    return out


def _assistant_message_dict(message):
    """把响应 message 转成可回填到对话的 assistant 消息 dict；
    若有原生 tool_calls 则带上（供 chat template 正确格式化工具调用回合）。"""
    raw = message.model_dump() if hasattr(message, "model_dump") else {}
    reasoning = (
        getattr(message, "reasoning", None)
        or getattr(message, "reasoning_content", None)
        or raw.get("reasoning")
        or raw.get("reasoning_content")
        or ""
    )
    d = {"role": "assistant", "content": message.content or reasoning}
    tcs = getattr(message, "tool_calls", None)
    if tcs:
        calls = []
        for tc in tcs:
            fn = getattr(tc, "function", None)
            if fn is None:
                continue
            calls.append({
                "id": getattr(tc, "id", None),
                "type": "function",
                "function": {"name": getattr(fn, "name", ""),
                             "arguments": getattr(fn, "arguments", "") or "{}"},
            })
        if calls:
            d["tool_calls"] = calls
    return d


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, str(default)))
    except ValueError:
        return default


_IMAGE_URL_CACHE_KEY = "_spectool_image_url_cache"
_IMAGE_URL_CACHE_LOCK = threading.Lock()


def _scaled_image_size(width: int, height: int, *, max_size: int,
                       max_pixels: int, factor: int,
                       min_pixels: int = 0,
                       resize_policy: str = "downscale") -> tuple[int, int]:
    """Return a bounded size while preserving aspect ratio.

    ``downscale`` is Spectool's historical policy.  ``speceyes`` reproduces
    the official SpecEyes ``smart_resize`` rule (nearest factor rounding plus
    its minimum and maximum pixel budgets).  It is opt-in through environment
    variables, so ordinary Spectool runs retain their existing preprocessing.
    """
    if resize_policy == "speceyes":
        # This is the official SpecEyes smart_resize calculation, expressed in
        # width/height order to match PIL.Image.resize.  It intentionally uses
        # Python's round/floor/ceil semantics from the source repository.
        if factor <= 1:
            return width, height
        scaled_width = max(factor, round(width / factor) * factor)
        scaled_height = max(factor, round(height / factor) * factor)
        area = scaled_width * scaled_height
        if max_pixels > 0 and area > max_pixels:
            beta = math.sqrt((width * height) / max_pixels)
            scaled_width = max(factor, math.floor(width / beta / factor) * factor)
            scaled_height = max(factor, math.floor(height / beta / factor) * factor)
        elif min_pixels > 0 and area < min_pixels:
            beta = math.sqrt(min_pixels / (width * height))
            scaled_width = max(factor, math.ceil(width * beta / factor) * factor)
            scaled_height = max(factor, math.ceil(height * beta / factor) * factor)
        return scaled_width, scaled_height

    ratio = 1.0
    if max_size > 0 and max(width, height) > max_size:
        ratio = min(ratio, max_size / max(width, height))
    if max_pixels > 0 and width * height > max_pixels:
        ratio = min(ratio, math.sqrt(max_pixels / (width * height)))
    if ratio >= 1.0:
        return width, height

    scaled_width = max(1, int(width * ratio))
    scaled_height = max(1, int(height * ratio))
    if factor > 1:
        if scaled_width >= factor:
            scaled_width = max(factor, scaled_width // factor * factor)
        if scaled_height >= factor:
            scaled_height = max(factor, scaled_height // factor * factor)
    return scaled_width, scaled_height


def _write_cached_image(payload: bytes, *, root: str, suffix: str) -> str:
    """Write content-addressed media atomically and return its file URI."""
    media_root = Path(root).expanduser().resolve()
    media_root.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256(payload).hexdigest()
    destination = media_root / f"{digest}.{suffix}"
    if not destination.exists():
        temp_name = None
        try:
            with tempfile.NamedTemporaryFile(
                    mode="wb", dir=media_root, prefix=".spectool-",
                    suffix=f".{suffix}.tmp", delete=False) as temp_file:
                temp_file.write(payload)
                temp_name = temp_file.name
            os.replace(temp_name, destination)
        finally:
            if temp_name and os.path.exists(temp_name):
                os.unlink(temp_name)
    return destination.as_uri()


def build_tool_result_messages(call_id, observation, images):
    """构造工具结果消息 list[dict]。原生(有 call_id) → role:tool 文本 + 图片单独
    role:user 跟随；文本回退(call_id 空) → 直接 role:user 带文本与图片。"""
    max_size = _env_int(
        "SPECTOOL_TOOL_IMAGE_MAX_SIZE",
        _env_int("SPECTOOL_IMAGE_MAX_SIZE", 768))
    img_parts = [{"type": "image_url",
                  "image_url": {"url": pil_to_data_url(im, max_size=max_size)}}
                 for im in images]
    if call_id:
        msgs = [{"role": "tool", "tool_call_id": call_id, "content": observation}]
        if img_parts:
            msgs.append({"role": "user",
                         "content": [{"type": "text", "text": "工具返回的图片如下："}]
                                    + img_parts})
        return msgs
    return [{"role": "user",
             "content": [{"type": "text", "text": observation}] + img_parts}]


def pil_to_data_url(image: Image.Image, fmt: str = "JPEG",
                    quality: Optional[int] = None,
                    max_size: Optional[int] = None) -> str:
    """Build a cached data or local-file URL for a model input image.

    The historical function name is kept for compatibility. Model-side image
    resizing never mutates the source image, so sandbox tools retain the full
    resolution original.
    """
    if max_size is None:
        max_size = _env_int("SPECTOOL_IMAGE_MAX_SIZE", 1024)
    if quality is None:
        quality = _env_int("SPECTOOL_IMAGE_QUALITY", 90)
    max_pixels = _env_int("SPECTOOL_IMAGE_MAX_PIXELS", 0)
    min_pixels = _env_int("SPECTOOL_IMAGE_MIN_PIXELS", 0)
    factor = max(1, _env_int("SPECTOOL_IMAGE_RESIZE_FACTOR", 1))
    resize_policy = os.environ.get(
        "SPECTOOL_IMAGE_RESIZE_POLICY", "downscale").strip().lower()
    transport = os.environ.get("SPECTOOL_IMAGE_TRANSPORT", "data").strip().lower()
    media_root = os.environ.get("SPECTOOL_LOCAL_MEDIA_ROOT", "").strip()
    fmt = os.environ.get("SPECTOOL_IMAGE_FORMAT", fmt).strip().upper() or fmt.upper()
    if transport not in {"data", "file"}:
        transport = "data"
    if resize_policy not in {"downscale", "speceyes"}:
        resize_policy = "downscale"
    if transport == "file" and not media_root:
        transport = "data"

    cache_key = (fmt, quality, max_size, max_pixels, min_pixels, factor,
                 resize_policy, transport, media_root)
    with _IMAGE_URL_CACHE_LOCK:
        cache = image.info.setdefault(_IMAGE_URL_CACHE_KEY, {})
        cached = cache.get(cache_key)
        if cached:
            return cached

        prepared = image if image.mode == "RGB" else image.convert("RGB")
        target_size = _scaled_image_size(
            *prepared.size, max_size=max_size, max_pixels=max_pixels,
            factor=factor, min_pixels=min_pixels, resize_policy=resize_policy)
        if target_size != prepared.size:
            prepared = prepared.resize(target_size, Image.LANCZOS)

        buf = io.BytesIO()
        prepared.save(buf, format=fmt, quality=quality)
        payload = buf.getvalue()
        mime = "jpeg" if fmt == "JPEG" else fmt.lower()
        if transport == "file":
            result = _write_cached_image(
                payload, root=media_root,
                suffix="jpg" if fmt == "JPEG" else fmt.lower())
        else:
            b64 = base64.b64encode(payload).decode("ascii")
            result = f"data:image/{mime};base64,{b64}"
        cache[cache_key] = result
        _boundary(
            f"image_transport mode={transport} source={image.size} "
            f"model={target_size} bytes={len(payload)}")
        return result


def downscale_data_url(url: str, max_size: int = 448, quality: int = 60) -> str:
    """将 data/file URL 中的图片降采样到指定最大尺寸。
    用于 Draft 模型接收缩小的图片以加速推理。"""
    if url.startswith("data:image/"):
        _, b64_data = url.split(",", 1)
        img = Image.open(io.BytesIO(base64.b64decode(b64_data)))
    elif url.startswith("file://"):
        image_path = unquote(urlparse(url).path)
        with Image.open(image_path) as source:
            img = source.copy()
    else:
        return url
    if img.mode != "RGB":
        img = img.convert("RGB")
    w, h = img.size
    if max(w, h) <= max_size:
        return url
    return pil_to_data_url(img, quality=quality, max_size=max_size)


def downscale_image_messages(messages: list, max_size: int = 448) -> list:
    """对消息列表中所有 image_url 进行降采样。
    用于 Draft 模型以较小图片快速推理。"""
    import copy
    result = copy.deepcopy(messages)
    for msg in result:
        content = msg.get("content")
        if not isinstance(content, list):
            continue
        for part in content:
            if isinstance(part, dict) and part.get("type") == "image_url":
                url = part.get("image_url", {}).get("url", "")
                part["image_url"]["url"] = downscale_data_url(url, max_size)
    return result


from openai import AsyncOpenAI, OpenAI


class RequestCancelled(RuntimeError):
    """Raised when a caller intentionally disconnects an in-flight request."""


class RequestCancellation:
    """Thread-safe handle that closes only the registered HTTP response."""

    def __init__(self):
        self._event = threading.Event()
        self._lock = threading.Lock()
        self._closer: Optional[Callable[[], None]] = None

    @property
    def cancelled(self) -> bool:
        return self._event.is_set()

    def register(self, closer: Callable[[], None]) -> None:
        close_now = False
        with self._lock:
            if self._event.is_set():
                close_now = True
            else:
                self._closer = closer
        if close_now:
            closer()

    def unregister(self, closer: Callable[[], None]) -> None:
        with self._lock:
            if self._closer is closer:
                self._closer = None

    def cancel(self) -> None:
        closer = None
        with self._lock:
            self._event.set()
            closer = self._closer
            self._closer = None
        if closer is not None:
            try:
                closer()
            except Exception:
                pass


def _quality_label(s):
    letters = "".join(ch for ch in (s or "").strip().upper() if ch.isalpha())
    if letters.startswith("GOOD"):
        return "GOOD"
    if letters.startswith("BAD"):
        return "BAD"
    return None


def _good_bad_distribution(logprobs_content, floor_logprob: float = -20.0):
    """Extract GOOD/BAD logprobs from the first generated judge token."""
    for t in logprobs_content or []:
        dist = {}
        label = _quality_label(getattr(t, "token", ""))
        if label is not None and getattr(t, "logprob", None) is not None:
            dist[label] = float(t.logprob)
        for cand in getattr(t, "top_logprobs", None) or []:
            label = _quality_label(getattr(cand, "token", ""))
            if label is None or getattr(cand, "logprob", None) is None:
                continue
            lp = float(cand.logprob)
            if label not in dist or lp > dist[label]:
                dist[label] = lp
        if dist:
            return {
                "GOOD": dist.get("GOOD", floor_logprob),
                "BAD": dist.get("BAD", floor_logprob),
            }
    return {}


def _binary_score_from_logprobs(log_good: float, log_bad: float) -> float:
    m = max(log_good, log_bad)
    good = math.exp(log_good - m)
    bad = math.exp(log_bad - m)
    denom = good + bad
    return good / denom if denom > 0 else 0.5


def _zero_one_label(token_text: str) -> Optional[str]:
    """Extract the first structured 0/1 decision from a generated token."""
    text = (token_text or "").strip()
    for char in text:
        if char in "01":
            return char
        if char.isalpha():
            continue
    return None


def _zero_one_distribution(logprobs_content, floor_logprob: float = -20.0):
    """Read P(answerable=1) and P(insufficient=0) at the decision field."""
    for token in logprobs_content or []:
        label = _zero_one_label(getattr(token, "token", ""))
        # Structured decoding exposes masked 0/1 candidates (logprob=-9999)
        # on prefix and separator tokens. Only inspect candidates once the
        # generated token itself is the actual binary decision field.
        if label is None:
            continue
        dist = {}
        if getattr(token, "logprob", None) is not None:
            dist[label] = float(token.logprob)
        for candidate in getattr(token, "top_logprobs", None) or []:
            label = _zero_one_label(getattr(candidate, "token", ""))
            if label is None or getattr(candidate, "logprob", None) is None:
                continue
            logprob = float(candidate.logprob)
            if label not in dist or logprob > dist[label]:
                dist[label] = logprob
        if dist:
            return {"0": dist.get("0", floor_logprob),
                    "1": dist.get("1", floor_logprob)}
    return {}


_UNIFIED_ASSESSMENT_RE = re.compile(
    r"^\s*(?P<quality>NA|GOOD|BAD)\|(?P<decision>[01])"
    r"(?:\|(?P<answer><answer>.*?</answer>))?\s*$", re.DOTALL)

_ANSWER_VERIFICATION_RE = re.compile(
    r"\s*(?P<quality>NA|GOOD|BAD)\|"
    r"(?:<evidence>(?P<evidence>[^<>\r\n]{1,400})</evidence>"
    r"(?P<answer><answer>[^<>]+</answer>))?"
    r"<verified>(?P<decision>[01])</verified>\s*", re.DOTALL)

def _verification_distribution(content, logprobs_content):
    """Read both real verdict logprobs, never a digit in evidence."""
    match = _ANSWER_VERIFICATION_RE.fullmatch(content)
    if match is None or match.group("decision") is None:
        return {}
    tokens = list(logprobs_content or [])
    # vLLM includes EOS in logprobs but omits it from message.content.
    while tokens and getattr(tokens[-1], "token", "") in {
            "<|im_end|>", "<|endoftext|>", "<|eot_id|>"}:
        tokens.pop()
    if "".join(getattr(t, "token", "") for t in tokens) != content:
        return {}
    decision_offset = match.start("decision")
    offset = 0
    for token in tokens:
        text = getattr(token, "token", "")
        if offset == decision_offset and text in {"0", "1"}:
            dist = {}
            for candidate in [token, *(getattr(token, "top_logprobs", None) or [])]:
                label = getattr(candidate, "token", "")
                lp = getattr(candidate, "logprob", None)
                if label in {"0", "1"} and lp is not None:
                    lp = float(lp)
                    if math.isfinite(lp) and -999 < lp <= 0:
                        dist[label] = max(lp, dist.get(label, -math.inf))
            return dist if len(dist) == 2 else {}
        offset += len(text)
    return {}


def _answer_verification_regex(choices=None, *, quality_enabled=True):
    quality = r"(GOOD|BAD)" if quality_enabled else "NA"
    answer = "(" + "|".join(re.escape(c) for c in choices) + ")" if choices else r"[^<>]{1,2000}"
    candidate = (
        r"<evidence>[^<>\r\n]{1,400}</evidence><answer>" + answer
        + r"</answer><verified>[01]</verified>"
    )
    # Both decisions remain legal for EVERY candidate, including after BAD.
    return quality + r"\|(<verified>0</verified>|" + candidate + ")"


class VLLMModel:
    """vLLM 模型的精简封装，兼容 OpenAI Chat Completions API。

    提供两种生成模式：
      - generate: 标准推理，支持 QJ 所需的 logprobs
      - generate_forced: 约束生成（用于 MC 题目的 guided choice 解码）
    """

    def __init__(self, config: AgentConfig):
        self.config = config
        # EvalKit dispatches API requests through a ThreadPoolExecutor. Keep
        # per-generation metadata isolated so one sample cannot consume
        # another sample's native tool calls or assistant message.
        self._generation_state = threading.local()
        self.max_retries = max(0, _env_int("SPECTOOL_OPENAI_MAX_RETRIES", 0))
        self.client = OpenAI(
            base_url=config.api_base,
            api_key=config.api_key,
            max_retries=self.max_retries,
        )

    def generate(self, messages: list[dict], tools: list | None = None) -> str:
        """标准推理生成。返回完整响应文本（推理 + 内容）。"""
        enable_thinking = os.environ.get(
            "SPECTOOL_ENABLE_THINKING", "0").strip().lower() in {
                "1", "true", "yes", "on"
            }
        create_kwargs = dict(
            model=self.config.model_name,
            messages=messages,
            temperature=self.config.temperature,
            max_tokens=_bounded_max_tokens(self.config.max_tokens, default_cap=0),
            # A full-length Main step (fallback) can take longer than the
            # short judge requests' API timeout under load.
            timeout=float(os.environ.get(
                "SPECTOOL_GENERATE_TIMEOUT", max(float(self.config.timeout), 600.0))),
            # repetition_penalty 打断 27B 贪心解码(temp=0)下的退化复读循环：
            # agent baseline 上 ~12% 样本会复读同一段裁图代码直到烧光 max_tokens
            # (单步 >600s),与 draft 的 1.1 对齐(见 draft_model.py)。
            extra_body={"chat_template_kwargs": {
                            "enable_thinking": enable_thinking},
                        "repetition_penalty": 1.1},
        )
        if tools:
            create_kwargs["tools"] = tools
            create_kwargs["tool_choice"] = "auto"
        resp = self.client.chat.completions.create(**create_kwargs)
        msg = resp.choices[0].message
        # 原生工具调用（vLLM --tool-call-parser）：优先用结构化 tool_calls
        _native = _parse_native_tool_calls(msg)
        self._generation_state.tool_calls = [tc for tc, _ in _native]
        self._generation_state.tool_call_ids = [cid for _, cid in _native]
        self._generation_state.assistant_msg = _assistant_message_dict(msg)

        return _model_response(resp.choices[0])

    def generate_text(
        self, messages: list[dict], *, max_tokens: int | None = None,
        logprobs: bool = False, top_logprobs: int | None = None,
    ) -> dict:
        """Generate one non-tool response, optionally retaining token logprobs.

        This deliberately does not mutate ``last_tool_calls``.  Lightweight
        routing and direct-answer gates should not leak a stale native tool
        call into a later AgentLoop request in the same worker thread.
        """
        request_kwargs = dict(
            model=self.config.model_name,
            messages=messages,
            temperature=0.0,
            max_tokens=_bounded_max_tokens(
                self.config.max_tokens if max_tokens is None else max_tokens,
                default_cap=0),
            timeout=self.config.timeout,
            extra_body={"chat_template_kwargs": {"enable_thinking": False}},
        )
        if logprobs:
            request_kwargs["logprobs"] = True
            request_kwargs["top_logprobs"] = int(
                top_logprobs if top_logprobs is not None else
                os.environ.get("SPECTOOL_TOP_LOGPROBS", "20"))
        resp = self.client.chat.completions.create(**request_kwargs)
        msg = resp.choices[0].message
        raw = msg.model_dump() if hasattr(msg, "model_dump") else {}
        text = (msg.content or raw.get("reasoning") or
                raw.get("reasoning_content") or "").strip()
        content_logprobs = []
        if logprobs:
            try:
                response_logprobs = resp.choices[0].logprobs
                content_logprobs = list(
                    getattr(response_logprobs, "content", None) or [])
            except Exception:
                # The caller treats an absent distribution as a conservative
                # fallback condition rather than trusting a guessed score.
                content_logprobs = []
        return {"text": text, "content_logprobs": content_logprobs}

    def judge_good_bad(self, messages: list[dict],
                       cancel_token: Optional[RequestCancellation] = None) -> dict:
        """Return P(GOOD) for a narrow GOOD/BAD quality-judge prompt."""
        request_kwargs = dict(
            model=self.config.model_name,
            messages=messages,
            temperature=0.0,
            max_tokens=4,
            timeout=self.config.timeout,
            logprobs=True,
            top_logprobs=int(os.environ.get("SPECTOOL_TOP_LOGPROBS", "20")),
            extra_body={
                "chat_template_kwargs": {"enable_thinking": False},
                "structured_outputs": {"choice": ["GOOD", "BAD"]},
            },
        )
        if cancel_token is None:
            resp = self.client.chat.completions.create(**request_kwargs)
        else:
            resp = asyncio.run(self._create_cancellable_completion(
                request_kwargs, cancel_token))
        msg = resp.choices[0].message
        raw = msg.model_dump() if hasattr(msg, "model_dump") else {}
        content = (msg.content or raw.get("reasoning") or
                   raw.get("reasoning_content") or "").strip()
        dist = {}
        try:
            logprobs = resp.choices[0].logprobs
            content_lp = getattr(logprobs, "content", None) if logprobs else None
            dist = _good_bad_distribution(content_lp)
        except Exception:
            dist = {}

        if not dist:
            label = _quality_label(content)
            if label == "GOOD":
                dist = {"GOOD": 0.0, "BAD": -20.0}
            elif label == "BAD":
                dist = {"GOOD": -20.0, "BAD": 0.0}
            else:
                raise ValueError(
                    "Structured GOOD/BAD response was not parseable: "
                    f"{content[:200]!r}")

        score = _binary_score_from_logprobs(dist["GOOD"], dist["BAD"])
        return {
            "score": score,
            "label": "GOOD" if score >= 0.5 else "BAD",
            "log_good": dist["GOOD"],
            "log_bad": dist["BAD"],
            "raw": content,
        }

    def answer_if_sufficient(self, messages: list[dict],
                             choices: Optional[list[str]] = None) -> str:
        """Return a short answer or INSUFFICIENT_EVIDENCE without mutating state."""
        extra_body = {
            "chat_template_kwargs": {"enable_thinking": False},
        }
        if choices:
            extra_body["structured_outputs"] = {"choice": [
                *(f"<answer>{choice}</answer>" for choice in choices),
                "INSUFFICIENT_EVIDENCE",
            ]}
        resp = self.client.chat.completions.create(
            model=self.config.model_name,
            messages=messages,
            temperature=0.0,
            max_tokens=32,
            timeout=self.config.timeout,
            extra_body=extra_body,
        )
        return _model_response(resp.choices[0])

    def fast_answer_gate(
        self, messages: list[dict], choices: Optional[list[str]] = None,
        cancel_token: Optional[RequestCancellation] = None,
    ) -> dict:
        """Compatibility wrapper for the shared answerability mechanism."""
        return self.assess_draft_step(
            messages, choices=choices, cancel_token=cancel_token,
            quality_enabled=False, include_logprobs=True)

    async def _create_cancellable_completion(
        self, request_kwargs: dict, cancel_token: RequestCancellation,
    ):
        """Create a request that can be cancelled before response headers.

        The synchronous OpenAI client does not expose a response object while
        vLLM is still doing multimodal prefill. Cancelling its eventual stream
        therefore cannot interrupt that phase. An asyncio task can be
        cancelled while awaiting headers, which closes the underlying HTTP
        request and lets vLLM abort the engine request immediately.
        """
        client = AsyncOpenAI(
            base_url=self.config.api_base,
            api_key=self.config.api_key,
            max_retries=self.max_retries,
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
                raise RequestCancelled("QJ superseded by Draft answer")
            return response
        except asyncio.CancelledError as exc:
            raise RequestCancelled(
                "QJ HTTP task cancelled after Draft answer") from exc
        finally:
            cancel_token.unregister(cancel_task)
            try:
                await client.close()
            except asyncio.CancelledError:
                pass

    def assess_draft_step(
        self, messages: list[dict], choices: Optional[list[str]] = None,
        cancel_token: Optional[RequestCancellation] = None,
        quality_enabled: bool = True, include_logprobs: bool = True,
    ) -> dict:
        """Assess tool quality and answerability in a short Main request."""
        extra_body = {
            "chat_template_kwargs": {"enable_thinking": False},
        }
        extra_body["structured_outputs"] = {"regex": _answer_verification_regex(
            choices, quality_enabled=quality_enabled)}
        _boundary(f"main_assessment_start messages={len(messages)} prompt_chars={sum(len(str(m.get('content', ''))) for m in messages)}")
        assessment_t0 = time.perf_counter()
        request_kwargs = dict(
            model=self.config.model_name,
            messages=messages,
            temperature=0.0,
            # A short evidence clause precedes the candidate and final verdict.
            max_tokens=128 if choices else 256,
            timeout=self.config.timeout,
            extra_body=extra_body,
        )
        if include_logprobs:
            request_kwargs["logprobs"] = True
            request_kwargs["top_logprobs"] = int(os.environ.get(
                "SPECTOOL_TOP_LOGPROBS", "20"))
        content_lp = []
        if cancel_token is None:
            resp = self.client.chat.completions.create(**request_kwargs)
        else:
            try:
                resp = asyncio.run(self._create_cancellable_completion(
                    request_kwargs, cancel_token))
            except RequestCancelled:
                _boundary(
                    "main_assessment_cancelled "
                    f"elapsed={time.perf_counter() - assessment_t0:.3f}")
                raise

        msg = resp.choices[0].message
        dumped = msg.model_dump() if hasattr(msg, "model_dump") else {}
        content = msg.content or ""
        try:
            logprobs = resp.choices[0].logprobs
            content_lp = (
                list(getattr(logprobs, "content", None) or [])
                if logprobs else [])
        except Exception:
            content_lp = []

        _boundary(f"main_assessment_response elapsed={time.perf_counter() - assessment_t0:.3f}")
        match = (None if getattr(resp.choices[0], "finish_reason", None) == "length"
                 else _ANSWER_VERIFICATION_RE.fullmatch(content))
        quality = match.group("quality") if match else "NA"
        decision = match.group("decision") if match else None
        answer = (parse_answer(match.group("answer"), normalize_mcq=False, choices=choices)
                  if match and match.group("answer") else None)
        decision_dist = _verification_distribution(content, content_lp)
        verification_score = (
            _binary_score_from_logprobs(decision_dist["1"], decision_dist["0"])
            if decision_dist else None)
        answerability_score = verification_score

        quality_dist = _good_bad_distribution(content_lp) if quality_enabled else {}
        quality_valid = not quality_enabled
        if quality_enabled and not quality_dist:
            if quality == "GOOD":
                quality_dist = {"GOOD": 0.0, "BAD": -20.0}
            elif quality == "BAD":
                quality_dist = {"GOOD": -20.0, "BAD": 0.0}
        if quality_enabled:
            quality_valid = (
                quality in {"GOOD", "BAD"}
                and "GOOD" in quality_dist
                and "BAD" in quality_dist)
        quality_score = (_binary_score_from_logprobs(
            quality_dist["GOOD"], quality_dist["BAD"])
            if quality_valid and quality_enabled else None)
        if match is None:
            reason = "invalid_format"
        elif decision is None:
            reason = "invalid_format"
        elif decision == "0":
            reason = "insufficient"
        elif answer is None:
            reason = "missing_answer"
        elif answerability_score is None:
            reason = "missing_logprobs"
        else:
            reason = "answer"
        return {
            "score": quality_score,
            "label": quality if quality_enabled else "NA",
            "quality_valid": quality_valid,
            "log_good": quality_dist.get("GOOD"),
            "log_bad": quality_dist.get("BAD"),
            "answerable": decision == "1",
            "answerability_score": answerability_score,
            "answerability_reason": reason,
            "answer": answer,
            "evidence": match.group("evidence") if match else None,
            "verification_log_1": decision_dist.get("1"),
            "verification_log_0": decision_dist.get("0"),
            "verification_score": verification_score,
            "verification_protocol": "answer_conditioned_v1",
            "raw": content,
        }

    def generate_forced(self, messages: list[dict],
                        choices: Optional[list[str]] = None,
                        max_tokens: int = 8192) -> str:
        """约束最终答案生成，用于 agentic 循环步数耗尽后强制提取答案。

        对于多选题（提供了 choices，如 ["A","B",...]），使用 vLLM 的
        structured_outputs.choice 约束输出完整 <answer>X</answer>，与解析器一致。
        对于开放题进行贪心生成并要求 <answer> 标签。

        关键：MCQ 约束解码时必须关闭 thinking。开 thinking 时模型会先进
        <think> 狂想，反而压过结构化约束——撞 max_tokens 截断也吐不出
        受约束的单字母，调用方只能把整段推理当答案返回（实测导致 forced 样本
        0% 正确）。故有 choices 时直接关闭 thinking；开放题继承普通生成的
        SPECTOOL_ENABLE_THINKING 设置，避免同一 baseline 中途切换推理模式。
        """
        enable_thinking = (
            not choices
            and os.environ.get(
                "SPECTOOL_ENABLE_THINKING", "0").strip().lower()
            in {"1", "true", "yes", "on"}
        )
        extra_body = {"chat_template_kwargs": {"enable_thinking": enable_thinking}}
        if choices:
            extra_body["structured_outputs"] = {
                "choice": [f"<answer>{choice}</answer>" for choice in choices]}
        resp = self.client.chat.completions.create(
            model=self.config.model_name,
            messages=messages,
            temperature=0.0,
            max_tokens=_bounded_max_tokens(
                max_tokens, default=2048,
                cap_env="SPECTOOL_FORCED_MAX_TOKENS_CAP"),
            timeout=self.config.timeout,
            extra_body=extra_body,
        )
        return _model_response(resp.choices[0])

    @property
    def last_tool_calls(self) -> list:
        """最近一次 generate() 的原生工具调用（list[ToolCall]）。"""
        state = getattr(self, "_generation_state", None)
        return getattr(state, "tool_calls", []) if state is not None else []

    @property
    def last_tool_call_ids(self) -> list:
        """与 last_tool_calls 对应的 tool_call_id（list，元素可为 None）。"""
        state = getattr(self, "_generation_state", None)
        return getattr(state, "tool_call_ids", []) if state is not None else []

    @property
    def last_assistant_msg(self) -> dict:
        """最近一次 generate() 响应转成的 assistant 消息 dict（含 tool_calls）。"""
        state = getattr(self, "_generation_state", None)
        if state is None:
            return {"role": "assistant", "content": ""}
        return getattr(state, "assistant_msg",
                       {"role": "assistant", "content": ""})
