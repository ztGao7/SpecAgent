"""
工具注册表和调度器。

Manage Agent tools. The default visual benchmark tool set is sandbox only.
sandbox 工具通过外部 Jupyter 沙箱服务（部署在 EvalKit/vLLM 旁）执行模型编写的
Python 代码，支持图像处理、数值计算、数据可视化等操作。

ToolRegistry 负责：
  - 工具注册和名称管理
  - 输入图片的持久化（保存到临时目录供 sandbox 使用）
  - 工具调度的分发和异常处理
  - 结果的格式化和类型转换
"""
import asyncio
import importlib.util
import json
import os
import shutil
import sys
import tempfile
import threading
import time
from typing import Callable

from PIL import Image

from .benchmarks import MMSEARCH_PROFILE, resolve_benchmark_profile
from .types import ToolCall
from . import sandbox as _sandbox


_INPUT_IMAGE_FILENAME = "input.png"
_MMSEARCH_MODULE = None
_MMSEARCH_MODULE_LOCK = threading.Lock()


def _boundary(message: str) -> None:
    if os.environ.get("SPECTOOL_BOUNDARY_DEBUG") == "1":
        print(f"[BOUNDARY] t={time.monotonic():.6f} pid={os.getpid()} tid={threading.get_ident()} {message}", file=sys.stderr, flush=True)


def _is_fatal_tool_error(exc: BaseException) -> bool:
    return any(
        cls.__name__ in {"FatalToolError", "InsufficientBalanceError"}
        for cls in type(exc).__mro__
    )


class ToolRegistry:
    """工具注册表，管理工具的注册、调度和执行。

    支持的工具：
      - sandbox: 在 Jupyter 沙箱中执行 Python 代码
    """

    def __init__(self):
        self._tools: dict[str, Callable] = {}  # 工具名称 -> 处理函数
        self._output_dir: str = ""             # 临时输出目录
        self._input_image_path: str = ""       # 输入图片路径
        self._input_image_paths: list[str] = []  # 全部输入图片路径
        self._previous_context: dict | None = None  # sandbox 上下文（跨步骤持久化）
        # Draft and Main can briefly overlap around a fallback.  Serialize a
        # sandbox request with state reset so a completed Draft request cannot
        # restore its session after Main has started cleanly.
        self._sandbox_context_lock = threading.RLock()
        self._mmsearch_module = None
        self._tool_audit_context = None

    def register(self, name: str, fn: Callable) -> None:
        """注册一个工具"""
        self._tools[name] = fn

    def names(self) -> list[str]:
        """获取所有已注册工具的名称列表"""
        return sorted(self._tools.keys())

    def validate_call(self, call: ToolCall) -> str | None:
        """Return a protocol error before an invalid call reaches a tool.

        Native tool-call parsers can emit ``sandbox({})`` when generation is
        truncated or its arguments cannot be parsed. Letting that reach
        ``_sandbox`` raises a TypeError and can trap an agent in retries. This
        deliberately validates only the registered name and sandbox's required
        code parameter, leaving valid calls unchanged.
        """
        if call.tool_name not in self._tools:
            return (
                f"Unknown tool '{call.tool_name}'. Available tools: "
                f"{', '.join(self.names())}."
            )
        if call.tool_name == "sandbox":
            code = call.params.get("code") if isinstance(call.params, dict) else None
            if not isinstance(code, str) or not code.strip():
                return (
                    "The sandbox call was rejected before execution because "
                    "it requires a non-empty string argument named 'code'."
                )
        return None

    def set_input_image(self, image: Image.Image) -> None:
        """单图兼容入口。"""
        self.set_input_images([image])

    def set_input_images(self, images: list[Image.Image]) -> None:
        """保存全部输入图片，供 sandbox 以 image_1、image_2 等读取。"""
        if not images:
            raise ValueError("at least one input image is required")
        self.cleanup()
        self._output_dir = tempfile.mkdtemp(prefix="spectool_sandbox_")
        self._input_image_paths = []
        for index, image in enumerate(images, start=1):
            if not isinstance(image, Image.Image):
                raise TypeError("all input images must be PIL Images")
            filename = _INPUT_IMAGE_FILENAME if index == 1 else f"input_{index}.png"
            path = os.path.join(self._output_dir, filename)
            prepared = image
            if prepared.mode not in ("RGB", "RGBA", "L"):
                prepared = prepared.convert("RGB")
            prepared.save(path)
            self._input_image_paths.append(path)
        self._input_image_path = self._input_image_paths[0]
        self._previous_context = None  # 新样本重置上下文

    def cleanup(self) -> None:
        """Clear sandbox state and remove all temporary images for one sample."""
        with self._sandbox_context_lock:
            context = self._previous_context
            output_dir = self._output_dir
            self._previous_context = None
            self._input_image_path = ""
            self._input_image_paths = []
            self._output_dir = ""

            _sandbox.clear_session(context)
        if output_dir:
            shutil.rmtree(output_dir, ignore_errors=True)

    def reset_sandbox_context(self) -> None:
        """Reset Python state only; use fork_clean_sandbox for a clean branch."""
        with self._sandbox_context_lock:
            context = self._previous_context
            self._previous_context = None
            _sandbox.clear_session(context)

    def is_stateful_tool(self, name: str) -> bool:
        """True when the tool runs in this registry's persistent Python session."""
        handler = self._tools.get(name)
        return getattr(handler, "__func__", None) is ToolRegistry._sandbox

    def fork_clean_sandbox(self, images: list[Image.Image]) -> "ToolRegistry":
        """Give Main its own files/session without disturbing in-flight Draft IO."""
        child = ToolRegistry()
        child._mmsearch_module = self._mmsearch_module
        # External readers and the audit belong to the same benchmark sample.
        # Rebind registry methods so sandbox/PythonInterpreter use child state.
        child._tool_audit_context = self._tool_audit_context
        for name, handler in self._tools.items():
            if getattr(handler, "__self__", None) is self:
                handler = handler.__func__.__get__(child, type(child))
            child.register(name, handler)
        try:
            # Recreate lossless originals from the supplied images, not files
            # which Draft code might have overwritten. No JPEG/resize change.
            child.set_input_images(images)
        except BaseException:
            child.cleanup()
            raise
        return child

    def dispatch(self, call: ToolCall, image: Image.Image) -> tuple[str, list]:
        """Run dispatch with the sample audit context bound in worker threads."""
        t0 = time.perf_counter()
        _boundary(f"dispatch_start tool={call.tool_name}")
        token = None
        if self._mmsearch_module is not None and self._tool_audit_context is not None:
            token = self._mmsearch_module.bind_tool_audit_context(
                self._tool_audit_context
            )
        try:
            return self._dispatch_impl(call, image)
        finally:
            if token is not None:
                self._mmsearch_module.unbind_tool_audit_context(token)
            _boundary(f"dispatch_done tool={call.tool_name} elapsed={time.perf_counter() - t0:.3f}")

    def _dispatch_impl(self, call: ToolCall, image: Image.Image) -> tuple[str, list]:
        """分发并执行一次工具调用。

        参数:
            call: 包含工具名称和参数的工具调用对象
            image: 原始输入图片（大部分工具操作基于此）
        返回:
            (观察文本, 结果图片列表)
        """
        fn = self._tools.get(call.tool_name)
        if fn is None:
            return (f"未知工具: {call.tool_name}. "
                    f"可用工具: {', '.join(self.names())}.", [])
        try:
            result = fn(image, **call.params)
        except TypeError as e:
            return (f"工具 {call.tool_name} 参数错误: {e}", [])
        except Exception as e:
            if _is_fatal_tool_error(e):
                raise
            return (f"工具 {call.tool_name} 执行错误: "
                    f"{type(e).__name__}: {e}", [])
        if isinstance(result, Image.Image):
            return (f"工具 {call.tool_name} 执行成功。新图片尺寸: {result.size}.", [result])
        if isinstance(result, tuple) and len(result) == 2:
            observation, imgs = result
            if (isinstance(observation, str) and isinstance(imgs, list)
                    and all(isinstance(i, Image.Image) for i in imgs)):
                return (observation, imgs)
        return (f"工具 {call.tool_name} 返回非图片结果: {result!r}", [])

    def start_tool_audit(self, sample_id: str):
        if self._mmsearch_module is None:
            return None
        if not hasattr(self._mmsearch_module, "set_tool_audit_sample"):
            return None
        token = self._mmsearch_module.set_tool_audit_sample(sample_id)
        self._tool_audit_context = self._mmsearch_module.get_tool_audit_context()
        return token

    def finish_tool_audit(self, token):
        if self._mmsearch_module is None or token is None:
            return {}
        try:
            return self._mmsearch_module.reset_tool_audit_sample(token)
        finally:
            self._tool_audit_context = None

    @staticmethod
    def is_fatal_tool_error(exc: BaseException) -> bool:
        return _is_fatal_tool_error(exc)

    def _sandbox(self, image: Image.Image, code: str) -> tuple[str, list[Image.Image]]:
        """在 Jupyter 沙箱中执行 Python 代码，返回输出文本和生成的图片。

        sandbox 工具是 Agent 进行视觉分析的核心工具，支持：
          - 区域裁剪和显示
          - 数值分析（计数、测量、统计）
          - 图像处理（边缘检测、滤波、颜色分析）
          - 数据可视化

        参数:
            image: PIL 图片（原始输入）
            code: 要执行的 Python 代码字符串
        返回:
            (标准输出文本, PIL 图片列表)
        """
        with self._sandbox_context_lock:
            output_dir = self._output_dir
            input_path = self._input_image_path

            if not os.path.isdir(output_dir):
                os.makedirs(output_dir, exist_ok=True)

            if not os.path.isfile(input_path):
                image.save(input_path)

            _boundary(f"sandbox_tool_execute_start code_chars={len(code)} input={input_path}")
            tool_t0 = time.perf_counter()
            result = _sandbox.execute(
                code=code,
                input_image_path=input_path,
                input_image_paths=self._input_image_paths,
                output_dir=output_dir,
                previous_context=self._previous_context,
            )

            _boundary(f"sandbox_tool_execute_done elapsed={time.perf_counter() - tool_t0:.3f} success={result.success} result_images={len(result.images)} output_chars={len(result.output or '')}")
            self._previous_context = result.execution_context

        if not result.success:
            error_msg = result.error or "未知沙箱错误"
            return (f"沙箱错误: {error_msg}\n\n标准输出: {result.output}", [])

        images = []
        for img_path in result.images:
            try:
                _boundary(f"sandbox_image_load_start path={img_path}")
                image_t0 = time.perf_counter()
                pil_img = Image.open(img_path)
                pil_img.load()
                images.append(pil_img)
                _boundary(f"sandbox_image_load_done path={img_path} size={pil_img.size} elapsed={time.perf_counter() - image_t0:.3f}")
            except Exception as e:
                return (f"沙箱生成了不可读的图片 {img_path}: {e}", [])

        obs = result.output or "(沙箱执行成功，无输出)"
        _boundary(f"sandbox_tool_return images={len(images)} obs_chars={len(obs)}")
        return (obs, images)


SANDBOX_TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "sandbox",
        "description": (
            "Execute Python on the input images to gather visual evidence. Original "
            "images are available in input order as `image_1`, `image_2`, and so on "
            "(each a PIL.Image). Use them to crop/zoom into "
            "regions to inspect small details, count or measure objects, do image "
            "processing, or compute on pixels. Available modules: numpy/np, cv2, "
            "matplotlib.pyplot/plt, PIL.Image, math, json. Call plt.show() to display "
            "any image (e.g. a crop) back to yourself."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "code": {
                    "type": "string",
                    "description": (
                        "Python code to run. Original inputs are image_1, image_2, "
                        "and so on in prompt order."
                    ),
                },
            },
            "required": ["code"],
        },
    },
}

DEFAULT_TOOL_SCHEMAS = [SANDBOX_TOOL_SCHEMA]


MMSEARCH_TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "search",
            "description": (
                "Perform batched web searches and return the top 10 results "
                "for each query. Accepts multiple complementary queries."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "array",
                        "items": {"type": "string"},
                        "minItems": 1,
                        "description": "One or more web search queries.",
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "visit",
            "description": "Visit webpages and return evidence relevant to a goal.",
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {
                        "type": ["string", "array"],
                        "items": {"type": "string"},
                        "description": "One URL or a list of URLs to visit.",
                    },
                    "goal": {
                        "type": "string",
                        "description": "What information to extract from the pages.",
                    },
                },
                "required": ["url", "goal"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "PythonInterpreter",
            "description": (
                "Execute Python in a sandbox. Use print() for any evidence that "
                "should be returned. The input image is available as image_1."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {"type": "string", "description": "Python code to execute."},
                },
                "required": ["code"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "crop_and_search",
            "description": (
                "Crop important regions from the input image and perform visual "
                "search for the specified goal. One or more bounding boxes may "
                "be supplied; coordinates use the 0-1000 scale."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "image_id": {"type": "string", "description": "Input image path."},
                    "bbox": {
                        "type": "array",
                        "items": {
                            "anyOf": [
                                {
                                    "type": "array",
                                    "items": {"type": "number"},
                                    "minItems": 4,
                                    "maxItems": 4,
                                },
                                {"type": "number"},
                            ]
                        },
                        "minItems": 1,
                        "description": "[x1,y1,x2,y2] or a list of such boxes in 0-1000 coordinates.",
                    },
                    "goal": {"type": "string", "description": "The visual search goal."},
                },
                "required": ["image_id", "bbox", "goal"],
            },
        },
    },
]


def _load_mmsearch_vdr_module():
    """Load the already validated VDR/MMSearch tool implementations lazily."""
    global _MMSEARCH_MODULE
    if _MMSEARCH_MODULE is not None:
        return _MMSEARCH_MODULE
    path = os.environ.get("MMSEARCH_TOOLS_PATH")
    if not path:
        raise RuntimeError("Set MMSEARCH_TOOLS_PATH to your VDR rllm/eval/tools.py for MMSearch.")
    with _MMSEARCH_MODULE_LOCK:
        if _MMSEARCH_MODULE is None:
            spec = importlib.util.spec_from_file_location(
                "spectool_mmsearch_vdr_tools", path
            )
            if spec is None or spec.loader is None:
                raise ImportError(f"Cannot load MMSearch tools from {path}")
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            _MMSEARCH_MODULE = module
    return _MMSEARCH_MODULE


def _load_mmsearch_vdr_tools() -> dict[str, object]:
    return dict(_load_mmsearch_vdr_module().DEEPRESEARCH_TOOLS)


def _call_async_tool(tool: object, params: dict) -> str:
    """Run a VDR async tool from Spectool's synchronous dispatch path."""
    async def invoke():
        return await tool.call(**params)

    result = asyncio.run(invoke())
    return str(result)


def _register_mmsearch_tools(registry: ToolRegistry) -> None:
    """Expose the four active MMSearch/VDR tool names to the AgentLoop."""
    tools = _load_mmsearch_vdr_tools()
    registry._mmsearch_module = _load_mmsearch_vdr_module()
    for name in ("search", "visit", "crop_and_search"):
        tool = tools.get(name)
        if tool is None:
            continue

        def call_vdr_tool(image, _tool=tool, _name=name, **params):
            if _name == "crop_and_search":
                params = dict(params)
                params["image_id"] = registry._input_image_path
            try:
                result = _call_async_tool(_tool, params)
                if _name != "crop_and_search":
                    return (result, [])
                context = registry._mmsearch_module.get_tool_audit_context() or {}
                counts = context.get("counts", {})
                reader_failed = result.count("[Error] Failed to fetch content") + result.count("[Error visiting webpage")
                reader_success = result.count("The useful information in [")
                stats = {
                    "bbox_requested": len(params.get("bbox", [])) if isinstance(params.get("bbox"), list) and params.get("bbox") and isinstance(params["bbox"][0], list) else 1,
                    "reader_success": reader_success,
                    "reader_failed": reader_failed,
                    "image_search_cache_hits": int(counts.get("image_search_cache_hits", 0)),
                    "image_search_cache_misses": int(counts.get("image_search_cache_misses", 0)),
                    "image_visit_cache_hits": int(counts.get("image_visit_cache_hits", 0)),
                    "image_visit_cache_misses": int(counts.get("image_visit_cache_misses", 0)),
                    "external_calls_so_far": int(context.get("external_calls", 0)),
                }
                if reader_failed and not reader_success:
                    stats["next_action"] = "No usable image-search webpage was read. Do not retry crop_and_search with more boxes; switch to text search or existing evidence."
                else:
                    stats["next_action"] = "Use retrieved evidence before another visual search."
                return (result + "\n\n[CropAndSearchStats]\n" + json.dumps(stats, ensure_ascii=False), [])
            except Exception as exc:
                if _is_fatal_tool_error(exc):
                    raise
                return (f"MMSearch tool {_name} failed: {type(exc).__name__}: {exc}", [])

        registry.register(name, call_vdr_tool)

    # Keep the exact MMSearch name while preserving Spectool's image-aware sandbox.
    registry.register("PythonInterpreter", registry._sandbox)


def get_tool_schemas(benchmark_profile=None, dataset=None) -> list[dict]:
    """Return per-sample schemas without mutating ``DEFAULT_TOOL_SCHEMAS``."""
    profile = resolve_benchmark_profile(benchmark_profile, dataset)
    if profile == MMSEARCH_PROFILE:
        return MMSEARCH_TOOL_SCHEMAS
    return DEFAULT_TOOL_SCHEMAS


def register_default_tools(
    registry: ToolRegistry,
    benchmark_profile=None,
    dataset=None,
) -> list[dict]:
    """Register the tool set selected for one benchmark sample.

    Default visual benchmarks expose only ``sandbox``; web_search is disabled.
    MMSearch's VDR tools are loaded only
    for an explicit MMSearch profile or an MMSearch dataset name.
    """
    schemas = get_tool_schemas(benchmark_profile, dataset)
    if schemas is MMSEARCH_TOOL_SCHEMAS:
        _register_mmsearch_tools(registry)
        return schemas
    registry.register("sandbox", registry._sandbox)
    return schemas
