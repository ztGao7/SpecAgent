"""
VDR 专用的模型输出解析器。

与 spectool_agent.model 中的解析器隔离，使 Vision-DeepResearch 原生
工具调用格式可以独立演化，不影响共享的（Qwen3.5）解析器。

- ``parse_answer`` / ``parse_fallback_answer``：格式无关，从共享模块复用
  （基于 <answer> 标签 / 裸 MCQ 字母）
- ``parse_tool_call``：VDR 专属，支持多种格式：
  1. XML 风格: <tool_call><sandbox><code>...</code></sandbox></tool_call>
  2. JSON 协议: {"name": "sandbox", "arguments": {"code": "..."}}
  3. 裸 <code> 标签
  4. Fenced 代码块 ```python ... ```

注意：VDR checkpoints 输出 XML 形式的工具调用。一旦确认精确的原生格式，
可以扩展下面的 ``parse_tool_call`` 来识别。
"""
import json
import re
from typing import Optional

from ..types import ToolCall
# Format-agnostic, shared with the Qwen3.5 path on purpose:
from ..model import parse_answer, parse_fallback_answer  # noqa: F401

_TOOL_CALL_RE = re.compile(r"<tool_call>(.*?)</tool_call>", re.DOTALL)
# VDR 原生 XML 工具标签。它们可能出现在 <tool_call>...</tool_call> 包裹内
# 或独立出现。code/query 标签是可选的（某些输出直接将载荷放在 <sandbox>/<web_search> 内部）。
_SANDBOX_RE = re.compile(
    r"<sandbox>\s*(?:<code>(?P<c>.*?)</code>|(?P<c2>.*?))\s*</sandbox>", re.DOTALL)
_WEB_SEARCH_RE = re.compile(
    r"<(?:web_search|search)>\s*(?:<query>(?P<q>.*?)</query>|(?P<q2>.*?))\s*</(?:web_search|search)>",
    re.DOTALL)
_BARE_CODE_RE = re.compile(r"<code>(.*?)</code>", re.DOTALL)
_FENCE_RE = re.compile(r"```(?:python|py)?\s*(.*?)```", re.DOTALL)
_PLACEHOLDER = {"", "..."}
# 残留的工具包裹标签。VDR 原生输出有时缺少 <code> 开标签却带 </code> 闭标签
# （或在 fenced 块里塞 stray </code>），导致提取出的载荷尾部混入闭标签，
# 送进沙箱即 SyntaxError。提取后统一剥除，等价于共享 JSON 路径的天然干净性。
_RESIDUAL_TAG_RE = re.compile(r"</?(?:code|sandbox|web_search|search|query)>")


def _strip_tags(payload: str) -> str:
    """剥除提取出的 code/query 载荷里残留的工具包裹标签并去空白。"""
    return _RESIDUAL_TAG_RE.sub("", payload).strip()


def _repair_json(text: str) -> str:
    """Add missing closing braces and brackets to balance a truncated JSON
    string. Accounts for string literals and escape sequences."""
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


def _from_json(json_str: str) -> Optional[ToolCall]:
    """Parse a JSON tool-call payload. Accepts both the main protocol
    (``name``/``arguments``) and the draft schema (``tool_name``/``params``)."""
    json_str = json_str.strip()
    if not json_str:
        return None
    try:
        data = json.loads(json_str, strict=False)
    except json.JSONDecodeError:
        repaired = _repair_json(json_str)
        if repaired == json_str:
            return None
        try:
            data = json.loads(repaired, strict=False)
        except json.JSONDecodeError:
            return None
    if not isinstance(data, dict):
        return None
    name = data.get("name") or data.get("tool_name")
    if not isinstance(name, str) or not name:
        return None
    args = data.get("arguments")
    if args is None:
        args = data.get("params", {})
    if not isinstance(args, dict):
        args = {}
    return ToolCall(tool_name=name, params=args)


def parse_tool_call(text: str) -> Optional[ToolCall]:
    """Parse a tool call from a model turn, tolerating multiple formats.

    VDR (Vision-DeepResearch) natively emits XML-ish tool calls, e.g.
    ``<tool_call><sandbox><code>...</code></sandbox></tool_call>`` and
    ``<tool_call><web_search><query>...</query></web_search></tool_call>``. We
    also keep the legacy JSON protocol working, and fall back to bare
    ``<code>`` / fenced code blocks, so small format drifts still parse.

    NOTE: calibrated to a single observed sample; validate against real VDR
    output (curl the endpoint) and tighten if the native tags differ.
    """
    if not text:
        return None

    # If a <tool_call> wrapper exists, search inside it first (most precise);
    # otherwise scan the whole turn for native tags.
    m = _TOOL_CALL_RE.search(text)
    inner = m.group(1) if m else text

    # 1) Legacy JSON protocol inside <tool_call>.
    if m:
        tc = _from_json(inner)
        if tc is not None:
            return tc

    # 2) Native sandbox: <sandbox>[<code>]CODE[</code>]</sandbox>
    sm = _SANDBOX_RE.search(inner)
    if sm:
        code = sm.group("c")
        if code is None:
            code = sm.group("c2") or ""
        code = _strip_tags(code)
        if code not in _PLACEHOLDER:
            return ToolCall(tool_name="sandbox", params={"code": code})

    # 3) Native web search: <web_search>[<query>]Q[</query>]</web_search>
    wm = _WEB_SEARCH_RE.search(inner)
    if wm:
        query = wm.group("q")
        if query is None:
            query = wm.group("q2") or ""
        query = _strip_tags(query)
        if query not in _PLACEHOLDER:
            return ToolCall(tool_name="web_search", params={"query": query})

    # 4) Bare <code>...</code> → sandbox
    cm = _BARE_CODE_RE.search(inner)
    if cm:
        code = _strip_tags(cm.group(1))
        if code not in _PLACEHOLDER:
            return ToolCall(tool_name="sandbox", params={"code": code})

    # 5) Fenced ```python ... ``` → sandbox (last-resort)
    fm = _FENCE_RE.search(inner)
    if fm:
        code = _strip_tags(fm.group(1))
        if code not in _PLACEHOLDER:
            return ToolCall(tool_name="sandbox", params={"code": code})

    return None
