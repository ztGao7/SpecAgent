"""
Jupyter 沙箱 HTTP 客户端。

通过 REST API 与容器化部署的 Jupyter IPython 内核通信，
执行 Agent 生成的 Python 代码并获取结果（文本输出 + 图片）。

支持：
  - 多沙箱 URL 轮询（负载均衡）
  - 会话管理（session_id，变量状态跨调用持久化）
  - 自动初始化（注入输入图片和必要的 import）
  - 结果收集（自动捕获 print 输出和 matplotlib 图片）
  - 请求追踪；执行结果未知时不自动重放有状态代码
"""

import base64
import json
import os
import re
import sys
import time
import uuid
import zlib
from dataclasses import dataclass, field
from io import BytesIO
from typing import Optional

import requests
from PIL import Image


DEFAULT_SANDBOX_URL = "http://127.0.0.1:18901/run_jupyter"
EXECUTION_TIMEOUT = int(os.environ.get("SPECTOOL_SANDBOX_CODE_TIMEOUT", "200"))      # 代码执行超时
REQUEST_TIMEOUT = int(os.environ.get("SPECTOOL_SANDBOX_REQUEST_TIMEOUT", "240"))     # HTTP 请求超时
_IMAGE_PATH_MARKER = "__SPECTOOL_IMAGE_PATH__="


def _boundary(message: str) -> None:
    if os.environ.get("SPECTOOL_BOUNDARY_DEBUG") == "1":
        print(f"[BOUNDARY] t={time.monotonic():.6f} pid={os.getpid()} {message}", file=sys.stderr, flush=True)


@dataclass
class SandboxResult:
    """沙箱执行结果"""
    success: bool
    output: str = ""
    images: list = field(default_factory=list)
    error: Optional[str] = None
    execution_context: Optional[dict] = None        # 执行上下文（包含 session_id）


_sandbox_urls: list[str] = []       # 沙箱 URL 列表（轮询使用）
_url_index: int = 0                # 当前轮询索引


def _get_sandbox_urls() -> list[str]:
    """获取沙箱 URL 列表。
    优先从环境变量 SPECTOOL_SANDBOX_URLS 读取（逗号分隔）；
    其次 SPECTOOL_SANDBOX_URL 或 CODE_SERVER_BASE；
    最后使用默认 URL。"""
    global _sandbox_urls
    if _sandbox_urls:
        return _sandbox_urls
    urls_env = os.environ.get("SPECTOOL_SANDBOX_URLS", "")
    if urls_env:
        _sandbox_urls = [u.strip() for u in urls_env.split(",") if u.strip()]
    if not _sandbox_urls:
        single = (
            os.environ.get("SPECTOOL_SANDBOX_URL")
            or os.environ.get("CODE_SERVER_BASE")
            or DEFAULT_SANDBOX_URL
        )
        _sandbox_urls = [single]
    return _sandbox_urls


def _sandbox_url(session_id: Optional[str] = None) -> str:
    """选择一个沙箱 URL。

    传入 session_id 时使用会话亲和路由（按 session_id 哈希固定到同一个沙箱），
    确保同一会话的初始化（注入图片）与后续代码执行落在同一个沙箱实例上——否则
    多沙箱下同一会话的调用会在实例间漂移，目标沙箱没有该会话的 kernel/图片，
    导致 FileNotFoundError / 状态丢失。不同会话仍分散到不同沙箱以均衡负载。
    无 session_id 或单沙箱时退化为简单轮询。"""
    global _url_index
    urls = _get_sandbox_urls()
    if len(urls) == 1:
        return urls[0]
    if session_id:
        return urls[zlib.crc32(session_id.encode("utf-8")) % len(urls)]
    url = urls[_url_index % len(urls)]
    _url_index += 1
    return url


def _new_session_id() -> str:
    """生成新的沙箱会话 ID"""
    return f"spectool-{int(time.time() * 1000)}-{uuid.uuid4().hex}"


def _encode_image(path: str) -> str:
    """将图片文件编码为 base64 字符串"""
    with Image.open(path) as img:
        if img.mode not in ("RGB", "RGBA", "L"):
            img = img.convert("RGB")
        buf = BytesIO()
        img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def _shared_fs_enabled() -> bool:
    """Return whether the sandbox should exchange images through shared paths."""
    value = os.environ.get("SPECTOOL_SANDBOX_SHARED_FS", "0")
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _shared_init_code(
    input_image_paths: list[str],
    output_dir: str,
    seen_image_paths: list[str],
) -> str:
    """Bootstrap one shared-filesystem tool step without embedding image bytes."""
    return f'''
from PIL import Image
import os
import time as _spectool_time

try:
    import cv2
except ImportError:
    cv2 = None

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except Exception:
    plt = None

image_paths = {json.dumps([os.path.realpath(path) for path in input_image_paths])}
image_path = image_paths[0]
temp_output_dir = {json.dumps(os.path.realpath(output_dir))}
os.makedirs(temp_output_dir, exist_ok=True)
_spectool_images = []
for _spectool_index, _spectool_path in enumerate(image_paths, start=1):
    _spectool_image = Image.open(_spectool_path)
    _spectool_image.load()
    if _spectool_image.mode not in ("RGB", "RGBA", "L"):
        _spectool_image = _spectool_image.convert("RGB")
    globals()[f"image_{{_spectool_index}}"] = _spectool_image
    _spectool_images.append(_spectool_image)

_SPECTOOL_SEEN_IMAGE_PATHS = set({json.dumps(seen_image_paths)})
_SPECTOOL_TRANSIENT_NAMES = {{
    "Image", "cv2", "plt", "image_path", "image_paths", "temp_output_dir",
    "_spectool_images", "_spectool_image", "_spectool_index", "_spectool_path",
    "processed_path", "processed_paths",
}}
_SPECTOOL_TRANSIENT_NAMES.update(
    f"image_{{index}}" for index in range(1, len(_spectool_images) + 1)
)
_SPECTOOL_TRANSIENT_OBJECT_IDS = {{id(image) for image in _spectool_images}}

def _spectool_shared_show(*args, **kwargs):
    if plt is None:
        return []
    saved_paths = []
    for figure_number in list(plt.get_fignums()):
        figure = plt.figure(figure_number)
        output_path = os.path.join(
            temp_output_dir,
            f"figure_{{_spectool_time.time_ns()}}_{{figure_number}}.png",
        )
        figure.savefig(output_path, format="png", bbox_inches="tight", dpi=150)
        saved_paths.append(output_path)
    plt.close("all")
    return saved_paths

if plt is not None:
    plt.show = _spectool_shared_show
'''.strip()


def _init_code(input_image_paths: list[str]) -> str:
    """生成沙箱初始化代码。

    在沙箱中执行以下操作：
      1. 导入常用库（PIL, cv2, matplotlib, numpy 等）
      2. 将输入图片嵌入为 image_1、image_2 等变量（PIL Image 格式）
      3. 创建临时输出目录
      4. 初始化已见图片路径集合（用于去重）
    """
    image_b64_values = [_encode_image(path) for path in input_image_paths]
    return f'''
from PIL import Image
import base64
import os
import tempfile
from io import BytesIO

try:
    import cv2
except ImportError:
    import subprocess, sys
    subprocess.check_call([sys.executable, "-m", "pip", "install", "opencv-python-headless", "-q"])
    import cv2

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except Exception:
    plt = None

_img_base64_values = {json.dumps(image_b64_values)}
temp_output_dir = tempfile.mkdtemp(prefix="spectool_sandbox_")
image_paths = []
for _spectool_index, _img_base64 in enumerate(_img_base64_values, start=1):
    _spectool_image = Image.open(BytesIO(base64.b64decode(_img_base64)))
    if _spectool_image.mode not in ("RGB", "RGBA", "L"):
        _spectool_image = _spectool_image.convert("RGB")
    globals()[f"image_{{_spectool_index}}"] = _spectool_image
    _spectool_path = os.path.join(temp_output_dir, f"input_{{_spectool_index}}.png")
    _spectool_image.save(_spectool_path)
    image_paths.append(_spectool_path)
image_path = image_paths[0]
_SPECTOOL_SEEN_IMAGE_PATHS = set(image_paths)
'''.strip()


_OUTPUT_COLLECTOR = r'''
# 输出收集器：自动收集沙箱执行过程中生成的所有图片文件和标准输出。
#
# 收集策略：
#   1. 检查 processed_path 和 processed_paths 变量（用户代码设置的显式路径）
#   2. 扫描 temp_output_dir 目录下的所有图片文件（*.png, *.jpg 等）
#   3. 去重（只返回新增的图片，跳过之前步骤已见过的）
#   4. 如果 IPython.display 可用则显示图片，否则打印路径
try:
    import glob as _spectool_glob
    import os as _spectool_os
    from PIL import Image as _SpectoolImage
    try:
        from IPython.display import display as _spectool_display
    except Exception:
        _spectool_display = None

    _spectool_paths = []
    _spectool_processed_path = globals().get("processed_path")
    if isinstance(_spectool_processed_path, str):
        _spectool_paths.append(_spectool_processed_path)
    elif isinstance(_spectool_processed_path, (list, tuple)):
        _spectool_paths.extend([p for p in _spectool_processed_path if isinstance(p, str)])

    _spectool_processed_paths = globals().get("processed_paths")
    if isinstance(_spectool_processed_paths, (list, tuple)):
        _spectool_paths.extend([p for p in _spectool_processed_paths if isinstance(p, str)])

    _spectool_output_dir = globals().get("temp_output_dir")
    if isinstance(_spectool_output_dir, str) and _spectool_os.path.isdir(_spectool_output_dir):
        for _spectool_ext in ("*.png", "*.jpg", "*.jpeg", "*.webp", "*.bmp", "*.gif", "*.tiff"):
            _spectool_paths.extend(_spectool_glob.glob(_spectool_os.path.join(_spectool_output_dir, _spectool_ext)))

    _spectool_seen = globals().setdefault("_SPECTOOL_SEEN_IMAGE_PATHS", set())
    for _spectool_path in _spectool_paths:
        if not isinstance(_spectool_path, str):
            continue
        if not _spectool_os.path.isfile(_spectool_path):
            continue
        if _spectool_path in _spectool_seen:
            continue
        _spectool_seen.add(_spectool_path)
        if _spectool_display is not None:
            _spectool_display(_SpectoolImage.open(_spectool_path))
        else:
            print(_spectool_path)
except Exception as _spectool_collect_error:
    print(f"[spectool] output collection failed: {_spectool_collect_error}")
'''.strip()


_SHARED_OUTPUT_COLLECTOR = rf'''
# Shared-FS output collector: return paths instead of rendering files as base64.
try:
    import glob as _spectool_glob
    import os as _spectool_os

    _spectool_paths = []
    _spectool_processed_path = globals().get("processed_path")
    if isinstance(_spectool_processed_path, str):
        _spectool_paths.append(_spectool_processed_path)
    elif isinstance(_spectool_processed_path, (list, tuple)):
        _spectool_paths.extend(
            p for p in _spectool_processed_path if isinstance(p, str)
        )

    _spectool_processed_paths = globals().get("processed_paths")
    if isinstance(_spectool_processed_paths, (list, tuple)):
        _spectool_paths.extend(
            p for p in _spectool_processed_paths if isinstance(p, str)
        )

    _spectool_output_dir = globals().get("temp_output_dir")
    if isinstance(_spectool_output_dir, str) and _spectool_os.path.isdir(
        _spectool_output_dir
    ):
        for _spectool_ext in (
            "*.png", "*.jpg", "*.jpeg", "*.webp", "*.bmp", "*.gif", "*.tiff"
        ):
            _spectool_paths.extend(
                _spectool_glob.glob(
                    _spectool_os.path.join(_spectool_output_dir, _spectool_ext)
                )
            )

    _spectool_seen = globals().setdefault("_SPECTOOL_SEEN_IMAGE_PATHS", set())
    for _spectool_path in _spectool_paths:
        if not isinstance(_spectool_path, str):
            continue
        _spectool_path = _spectool_os.path.realpath(_spectool_path)
        if not _spectool_os.path.isfile(_spectool_path):
            continue
        if _spectool_path in _spectool_seen:
            continue
        _spectool_seen.add(_spectool_path)
        print("{_IMAGE_PATH_MARKER}" + _spectool_path)

    _spectool_image = globals().get("image_1")
    if _spectool_image is not None:
        _SPECTOOL_TRANSIENT_OBJECT_IDS = {{id(_spectool_image)}}
except Exception as _spectool_collect_error:
    print(f"[spectool] output collection failed: {{_spectool_collect_error}}")
'''.strip()


def _with_output_collector(code: str, shared_fs: bool = False) -> str:
    """将输出收集器代码附加到用户代码末尾"""
    collector = _SHARED_OUTPUT_COLLECTOR if shared_fs else _OUTPUT_COLLECTOR
    return f"{code.rstrip()}\n\n{collector}"


def _post_code(session_id: str, code: str, code_timeout: int, max_retries: int = 1) -> tuple[Optional[dict], Optional[str]]:
    """Submit once. The legacy max_retries argument never permits code replay."""
    request_id = uuid.uuid4().hex
    _boundary(f"sandbox_post_start session={session_id} code_chars={len(code)} timeout={code_timeout}")
    try:
        _boundary(f"sandbox_post_attempt session={session_id} request_id={request_id} attempt=1")
        response = requests.post(
            _sandbox_url(session_id),
            json={"session_id": session_id, "code": code, "timeout": code_timeout,
                  "request_id": request_id},
            # Allow bounded kernel cleanup after the server-side deadline.
            timeout=max(REQUEST_TIMEOUT, code_timeout + 30),
        )
        if hasattr(response, "raise_for_status"):
            response.raise_for_status()
        payload = response.json()
        if not isinstance(payload, dict):
            raise ValueError("Sandbox returned a non-object response")
        if payload.get('status') not in ('success', 'error'):
            raise ValueError("Sandbox response has no recognized status")
        if payload.get('request_id') not in (None, request_id):
            raise ValueError("Sandbox response request_id mismatch")
        _boundary(f"sandbox_post_done session={session_id} request_id={request_id} http={response.status_code} status={payload.get('status')!r}")
        return payload, None
    except requests.HTTPError:
        body = response.text[:500] if response.text else "(empty body)"
        last_error = f"沙箱 HTTP {response.status_code}: {body}"
    except Exception as e:
        last_error = f"请求 Jupyter 沙箱失败: {type(e).__name__}: {e}"
    last_error += f" [request_id={request_id}; execution outcome unknown; not replayed]"
    _boundary(f"sandbox_post_failed session={session_id} error={last_error}")
    return None, last_error


def clear_session(context: Optional[dict]) -> None:
    """Best-effort removal of one sandbox session's persisted Redis state."""
    if not context:
        return
    session_id = context.get("session_id")
    if not session_id:
        return
    run_url = _sandbox_url(session_id)
    clear_url = run_url.rsplit("/", 1)[0] + "/clear_session"
    try:
        response = requests.post(
            clear_url,
            json={"session_id": session_id},
            timeout=min(REQUEST_TIMEOUT, 10),
        )
        response.raise_for_status()
    except Exception:
        # Cleanup must never turn a completed evaluation into a failed sample.
        pass


def _decode_images(image_payloads: list, output_dir: str) -> list[str]:
    """将沙箱返回的 base64 图片 payloads 解码并保存到本地文件"""
    paths = []
    os.makedirs(output_dir, exist_ok=True)
    for idx, payload in enumerate(image_payloads):
        if not isinstance(payload, str):
            continue
        raw_payload = payload.split(",", 1)[-1]
        try:
            img = Image.open(BytesIO(base64.b64decode(raw_payload)))
            img.load()
        except Exception:
            continue
        path = os.path.join(output_dir, f"sandbox_output_{int(time.time() * 1000)}_{idx}.png")
        img.save(path)
        paths.append(path)
    return paths


def _extract_shared_image_paths(
    output: dict,
    output_dir: str,
    context: dict,
) -> list[str]:
    """Extract path markers from stdout and remove them from model-visible text."""
    stdout = str(output.get("stdout", ""))
    stdout_raw = str(output.get("stdout_raw", stdout))
    root = os.path.realpath(output_dir)
    seen = set(context.get("seen_image_paths", []))
    paths = []

    def parse(value: str) -> str:
        kept_lines = []
        for line in value.splitlines(keepends=True):
            stripped = line.strip()
            if not stripped.startswith(_IMAGE_PATH_MARKER):
                kept_lines.append(line)
                continue
            path = os.path.realpath(stripped[len(_IMAGE_PATH_MARKER):])
            try:
                inside_root = os.path.commonpath([root, path]) == root
            except ValueError:
                inside_root = False
            if inside_root and os.path.isfile(path) and path not in seen:
                seen.add(path)
                paths.append(path)
        return "".join(kept_lines)

    output["stdout"] = parse(stdout)
    output["stdout_raw"] = parse(stdout_raw)
    context["seen_image_paths"] = sorted(seen)
    return paths


def _format_output(output: dict) -> str:
    """格式化沙箱输出字典为可读字符串"""
    stdout = output.get("stdout", "")
    stderr = output.get("stderr", "")
    result = output.get("result", "")
    parts = []
    if stdout.strip():
        parts.append(str(stdout))
    if result.strip():
        parts.append(str(result))
    if stderr.strip():
        parts.append("stderr:\n" + str(stderr))
    return "\n".join(p for p in parts)


def _python_execution_error(stderr: str) -> Optional[str]:
    """Recognize Python/Jupyter failures, not arbitrary stderr or warnings."""
    text = re.sub(r"\x1b\[[0-?]*[ -/]*[@-~]", "", stderr or "")
    summaries = re.findall(
        r"(?m)^[ \t]*(?:[A-Za-z_]\w*\.)*"
        r"(?:\w*(?:Error|Exception)|KeyboardInterrupt|SystemExit|GeneratorExit)"
        r"(?::[^\n]*)?[ \t]*$", text)
    if summaries:
        return summaries[-1].strip()
    traceback_header = re.search(
        r"(?m)^[ \t]*(?:[A-Za-z_][\w.]*[ \t]+)?"
        r"Traceback[ \t]+\(most recent call last\)", text)
    if traceback_header:
        return text.strip().splitlines()[-1].strip()
    timeout = re.search(
        r"(?m)^Timed out waiting for Jupyter kernel "
        r"(?:output|execution reply)\.", text)
    return timeout.group(0) if timeout else None


def _response_to_result(
    resjson: dict,
    output_dir: str,
    context: dict,
    shared_fs: bool = False,
) -> SandboxResult:
    """将沙箱 JSON 响应解析为 SandboxResult 对象"""
    _boundary(f"sandbox_response_parse_start status={resjson.get('status')!r} keys={sorted(resjson.keys())}")
    status = resjson.get("status")
    error = resjson.get("error")
    output = resjson.get("output")
    if resjson.get("session_usable") is False:
        context["unusable_reason"] = error or "Sandbox could not confirm safe session state"
    if not isinstance(output, dict):
        return SandboxResult(
            success=False,
            output="",
            error=error or f"沙箱返回 status={status}",
            execution_context=context,
        )
    image_paths = []
    if shared_fs:
        image_paths.extend(_extract_shared_image_paths(output, output_dir, context))
    image_paths.extend(_decode_images(output.get("images", []), output_dir))
    # Some servers report request success even when the kernel raised an
    # exception. Keep that execution failure out of the QJ evidence path.
    execution_error = _python_execution_error(output.get("stderr") or "")
    success = status == "success" and not error and execution_error is None
    text = _format_output(output)
    if not success and error is None:
        error = execution_error or str(
            output.get("stderr") or output.get("result") or status)
    result = SandboxResult(
        success=success,
        output=text,
        images=image_paths,
        error=error,
        execution_context=context,
    )
    _boundary(f"sandbox_response_parse_done status={status!r} success={success} images={len(image_paths)} output_chars={len(text)}")
    return result


def execute(
    code: str,
    input_image_path: str,
    output_dir: str,
    timeout: int = EXECUTION_TIMEOUT,
    previous_context: Optional[dict] = None,
    input_image_paths: Optional[list[str]] = None,
) -> SandboxResult:
    """在 Jupyter 沙箱中执行 Python 代码。

    这是沙箱客户端的主入口函数。流程：
      1. 创建/恢复会话（session_id 在上下文中持久化）
      2. 首次执行时运行初始化代码（导入库、加载图片）
      3. 将用户代码与输出收集器拼接后发送执行
      4. 解析响应并返回文本输出和图片

    参数:
        code:        要执行的 Python 代码
        input_image_path: 输入图片文件路径
        input_image_paths: 全部输入图片路径；省略时仅使用 input_image_path
        output_dir:  输出目录（存放结果图片）
        timeout:     代码执行超时（秒）
        previous_context: 之前步骤的执行上下文（用于变量持久化）
    返回:
        SandboxResult 对象
    """
    _boundary(f"sandbox_execute_start shared_fs={_shared_fs_enabled()} code_chars={len(code)} output_dir={output_dir}")
    os.makedirs(output_dir, exist_ok=True)
    source_paths = list(input_image_paths or [input_image_path])
    if not source_paths:
        source_paths = [input_image_path]
    context = dict(previous_context or {})
    if context.get("unusable_reason"):
        return SandboxResult(success=False,
            error="Sandbox session is quarantined after an uncertain result; "
                  "start a fresh sandbox branch. " + context["unusable_reason"],
            execution_context=context)
    session_id = context.get("session_id") or _new_session_id()
    context["session_id"] = session_id

    if _shared_fs_enabled():
        source_paths = [os.path.realpath(path) for path in source_paths]
        input_image_path = source_paths[0]
        output_dir = os.path.realpath(output_dir)
        seen_paths = set(context.get("seen_image_paths", []))
        seen_paths.update(source_paths)
        context["seen_image_paths"] = sorted(seen_paths)
        bootstrap = _shared_init_code(
            source_paths,
            output_dir,
            context["seen_image_paths"],
        )
        request_code = "\n\n".join(
            (bootstrap, _with_output_collector(code, shared_fs=True))
        )
        resjson, error = _post_code(session_id, request_code, timeout)
        if error is not None:
            context["unusable_reason"] = error
            return SandboxResult(
                success=False,
                error=error,
                execution_context=context,
            )
        context["initialized"] = True
        return _response_to_result(
            resjson or {},
            output_dir,
            context,
            shared_fs=True,
        )

    if not context.get("initialized"):
        # 首次执行：运行初始化代码
        init_resjson, init_error = _post_code(
            session_id, _init_code(source_paths), timeout)
        if init_error is not None:
            context["unusable_reason"] = init_error
            return SandboxResult(success=False, error=init_error, execution_context=context)
        init_result = _response_to_result(init_resjson or {}, output_dir, context)
        if not init_result.success:
            init_result.error = f"沙箱初始化失败: {init_result.error}"
            return init_result
        context["initialized"] = True

    # 执行用户代码（附加输出收集器）
    resjson, error = _post_code(session_id, _with_output_collector(code), timeout)
    if error is not None:
        context["unusable_reason"] = error
        return SandboxResult(success=False, error=error, execution_context=context)
    return _response_to_result(resjson or {}, output_dir, context)
