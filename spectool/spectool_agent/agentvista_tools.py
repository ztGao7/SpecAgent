"""AgentVista tool environment for every EvalKit method (Agent, Spectool, SpecEyes, SpecReason, DREAM-R).

The four tool names, descriptions and parameters follow the official AgentVista
harness (hkust-nlp/AgentVista @ 3bc3cad): ``web_search``, ``image_search``,
``visit`` and ``code_interpreter``.

* ``web_search`` / ``image_search``: the official tool classes are loaded from
  the official repository (``AGENTVISTA_REPO``) and call Serper unchanged
  (``SERPAPI_KEY``; Google Search, Google Images and Lens reverse search).
* ``visit``: pages are read with the Serper scrape API instead of Jina, then
  summarized with the official prompt (``Visit._summarize_with_api``) by one
  fixed summary model (default: the Main model, thinking off), so the tool
  environment is identical for every method.
* ``code_interpreter``: Spectool's image-aware Jupyter sandbox.  The original
  images are aliased to the official names (``original_image``,
  ``original_image_1`` ...); images it displays become ``tool_image_N`` for a
  later reverse ``image_search``.

The tool handlers are bound methods of the registry, so ``fork_clean_sandbox``
rebinds them to the child registry used by a Main fallback.
"""
import importlib
import json
import os
import re
import sys
import threading
import time
import types

import requests

_OFFICIAL = None
_OFFICIAL_LOCK = threading.Lock()
_SERPER_SCRAPE_URL = "https://scrape.serper.dev"

AGENTVISTA_SYSTEM_PROMPT = """You are a visual reasoning agent. Your goal is to answer questions about images.

# AVAILABLE TOOLS:

You have access to the following tools:
1. **web_search**: Search the web for information, facts, or current events
2. **image_search**: Search for related images using text query or reverse image search
3. **visit**: Visit a webpage and extract its main content
4. **code_interpreter**: Execute Python code for image processing, analysis, and calculations

# INSTRUCTIONS:

1. Analyze: Carefully observe the image and the user's question.
2. Think: Explain your step-by-step reasoning process.
3. Use Tools: Call appropriate tool to gather information and help answer the question.
4. Iterate as needed: Continue reasoning and using tools in next turns until you are confident in your findings.
5. Answer: Once confident, provide the final answer inside <answer>...</answer> tags!

# IMPORTANT:

- Always explain your detailed reasoning process before using any tool.
- You can ONLY call one tool at one turn! Do not call multiple tools in one turn!
- You MUST provide your final answer using complete <answer>...</answer> tags!
"""

_CODE_DESCRIPTION = """
Executes Python code in a stateful Jupyter kernel and return results.

Capabilities:
- Use PIL, NumPy, or OpenCV to process or analyze images to improve understanding.
- Perform complex mathematical calculations or data manipulation.
- Code execution is persistent: variables and functions are stored and reusable in subsequent calls.

Image Variables (Pre-loaded as PIL Image objects):
- original_image / original_image_N (N=1,2,3,...): The original input images are already loaded as PIL Image objects (original_image is the first input image, original_image_1 the second, and so on; the same images are also available as image_1, image_2, ...). You can directly use these variables without loading them again.

Visualization & Output:
- To view generated images, use plt.show().
- DO NOT use image.show().
- Displayed images will be available as "tool_image_N" in the next turn (e.g. for a reverse image_search).

Pre-installed packages:
- PIL
- NumPy
- OpenCV
- Matplotlib
"""

AGENTVISTA_TOOL_SCHEMAS = [
    {"type": "function", "function": {
        "name": "web_search",
        "description": "Search the web for information online. Use when you need to find information, facts, or current events. Returns web search results with titles, URLs, and text snippets. You only have limited search times, so please use it wisely.",
        "parameters": {"type": "object", "properties": {
            "query": {"type": "string", "description": "Search query string"},
            "max_results": {"type": "integer", "description": "Maximum number of results to return (default: 10)", "default": 10}},
            "required": ["query"]}}},
    {"type": "function", "function": {
        "name": "image_search",
        "description": """
Search for related images using text query or reverse image search.
- For text-to-image search: specify search_type="text" and provide a query.
- For reverse image search: specify search_type="reverse" and provide an image_url.
Returns images with URLs and descriptions.

Notice:
To perform reverse image search, specify the image name directly.
- "original_image" - use the original input image ("original_image_1", "original_image_2", ... for further input images)
- "tool_image_N" - use tool-generated image N (from code_interpreter outputs, e.g., tool_image_1, tool_image_2)
- Image URLs: Direct URL string (http:// or https://) from previous search results or other sources

You only have limited search times, so please use it wisely.
""",
        "parameters": {"type": "object", "properties": {
            "search_type": {"type": "string", "enum": ["text", "reverse"], "description": "Type of search: 'text' for text-to-image search, 'reverse' for reverse image search", "default": "text"},
            "query": {"type": "string", "description": "Search query string (required for text search)"},
            "image_url": {"type": "string", "description": "Image name or URL (required for reverse search). Use 'original_image', 'original_image_N', 'tool_image_N', or a direct image URL (http://... or https://...)"},
            "max_results": {"type": "integer", "description": "Maximum number of image results to return (default: 10)", "default": 10}},
            "required": []}}},
    {"type": "function", "function": {
        "name": "visit",
        "description": "Visit a webpage and extract its main content. Use when you have a specific URL to visit (often after getting a URL from web search or image search). Extracts and returns the main textual content of the webpage.",
        "parameters": {"type": "object", "properties": {
            "url": {"type": "string", "description": "Full URL of the webpage to visit (must start with http:// or https://)"},
            "goal": {"type": "string", "description": "What information you want to find on this page (helps focus the extraction)"}},
            "required": ["url", "goal"]}}},
    {"type": "function", "function": {
        "name": "code_interpreter",
        "description": _CODE_DESCRIPTION,
        "parameters": {"type": "object", "properties": {
            "code": {"type": "string", "description": "Python code to execute in the Jupyter kernel"}},
            "required": ["code"]}}},
]


def _serper_key() -> str:
    key = os.environ.get("SERPAPI_KEY") or os.environ.get("SERPER_API_KEY") or ""
    if not key:
        raise RuntimeError("AgentVista tools need SERPAPI_KEY (a Serper.dev key) in the environment")
    return key


def _load_official():
    """Import the official tool classes without leaving a top-level ``tools`` module behind."""
    global _OFFICIAL
    if _OFFICIAL is not None:
        return _OFFICIAL
    with _OFFICIAL_LOCK:
        if _OFFICIAL is not None:
            return _OFFICIAL
        root = os.environ.get("AGENTVISTA_REPO", "")
        if not root or not os.path.isfile(os.path.join(root, "tools", "web_search.py")):
            raise RuntimeError("Set AGENTVISTA_REPO to the official AgentVista checkout (3bc3cad)")
        clash = lambda name: name == "tools" or name.startswith("tools.")
        saved = {k: v for k, v in sys.modules.items() if clash(k)}
        for k in saved:
            del sys.modules[k]
        sys.path.insert(0, root)
        try:
            web = importlib.import_module("tools.web_search")
            img = importlib.import_module("tools.image_search")
            vis = importlib.import_module("tools.visit")
        finally:
            sys.path.remove(root)
            for k in [k for k in sys.modules if clash(k)]:
                del sys.modules[k]
            sys.modules.update(saved)
        _OFFICIAL = types.SimpleNamespace(WebSearch=web.WebSearch, ImageSearch=img.ImageSearch, Visit=vis.Visit)
    return _OFFICIAL


def _summary_endpoint() -> tuple[str, str, str]:
    base = (os.environ.get("AGENTVISTA_SUMMARY_BASE") or os.environ.get("SPECTOOL_VLLM_BASE") or "").rstrip("/")
    model = os.environ.get("AGENTVISTA_SUMMARY_MODEL") or os.environ.get("SPECTOOL_VLLM_MODEL") or ""
    key = os.environ.get("AGENTVISTA_SUMMARY_KEY") or os.environ.get("SPECTOOL_VLLM_KEY") or "EMPTY"
    return (base + "/chat/completions" if base else ""), model, key


_TLS = threading.local()
_ORIGINAL_POST = requests.post
_POST_HOOK_LOCK = threading.Lock()


def _post_without_thinking(url, *args, **kwargs):
    """requests.post pass-through; only inside a summary call it disables Qwen thinking."""
    if getattr(_TLS, "summary_endpoint", None) == url and isinstance(kwargs.get("json"), dict):
        payload = dict(kwargs["json"])
        payload.setdefault("chat_template_kwargs", {"enable_thinking": False})
        kwargs["json"] = payload
    return _ORIGINAL_POST(url, *args, **kwargs)


def _install_post_hook():
    with _POST_HOOK_LOCK:
        if requests.post is not _post_without_thinking:
            requests.post = _post_without_thinking


class _SummaryVisit:
    """Official ``Visit._summarize_with_api`` (same prompt, temperature 0.3, 8192 tokens),
    sent to one fixed summary model with thinking off, as on the official 27B server."""

    def __init__(self, official_visit_cls):
        self._cls = official_visit_cls
        _install_post_hook()

    def summarize(self, content, goal, url):
        endpoint, model, key = _summary_endpoint()
        if not endpoint or not model:
            raise RuntimeError("no summary model configured (SPECTOOL_VLLM_BASE / SPECTOOL_VLLM_MODEL)")
        obj = object.__new__(self._cls)
        obj.api_key, obj.api_endpoint, obj.model_name = key, endpoint, model
        _TLS.summary_endpoint = endpoint
        try:
            return obj._summarize_with_api(content, goal, url)
        finally:
            _TLS.summary_endpoint = None


def serper_scrape(url: str, timeout: float = 120.0) -> str:
    """Read one page with the Serper scrape API; markdown when available, else text."""
    last = ""
    for attempt in range(3):
        try:
            r = requests.post(_SERPER_SCRAPE_URL, headers={"X-API-KEY": _serper_key(), "Content-Type": "application/json"},
                              json={"url": url, "includeMarkdown": True}, timeout=timeout)
            if r.status_code == 200:
                data = r.json()
                text = data.get("markdown") or data.get("text") or ""
                if text.strip():
                    return text
                last = "empty page"
            else:
                last = f"status {r.status_code}: {r.text[:200]}"
        except Exception as exc:  # network errors are returned to the agent as tool errors
            last = f"{type(exc).__name__}: {exc}"
        time.sleep(1 + attempt)
    raise RuntimeError(f"Serper scrape failed for {url}: {last}")


def _tool_image_dir(registry) -> str:
    path = os.path.join(registry._output_dir or "/tmp", "agentvista_tool_images")
    os.makedirs(path, exist_ok=True)
    return path


def _av_code_interpreter(self, image, code):
    """Run code in the shared sandbox with the official original-image variable names."""
    n = max(1, len(self._input_image_paths))
    alias = ["original_image = image_1"] + [f"original_image_{k} = image_{k + 1}" for k in range(1, n)]
    observation, images = self._sandbox(image, code="\n".join(alias) + "\n" + code)
    if images:
        saved = getattr(self, "_av_tool_images", None)
        if saved is None:
            saved = self._av_tool_images = []
        names = []
        for pil in images:
            name = f"tool_image_{len(saved) + 1}"
            path = os.path.join(_tool_image_dir(self), name + ".png")
            pil.save(path)
            saved.append(path)
            names.append(name)
        observation = f"{observation}\n\n[Displayed images are available as: {', '.join(names)}]"
    return observation, images


def _resolve_image_ref(self, ref: str) -> str:
    ref = (ref or "").strip()
    if not ref or ref.startswith(("http://", "https://", "data:")):
        return ref
    key = os.path.splitext(os.path.basename(ref))[0]
    if key == "original_image" and self._input_image_paths:
        return self._input_image_paths[0]
    m = re.fullmatch(r"original_image_(\d+)", key)
    if m and int(m.group(1)) < len(self._input_image_paths):
        return self._input_image_paths[int(m.group(1))]
    m = re.fullmatch(r"(?:tool_image|observation)_(\d+)", key)
    saved = getattr(self, "_av_tool_images", None) or []
    if m and 1 <= int(m.group(1)) <= len(saved):
        return saved[int(m.group(1)) - 1]
    raise ValueError(f"Image reference '{ref}' not found. Available: original_image"
                     + "".join(f", original_image_{k}" for k in range(1, len(self._input_image_paths)))
                     + "".join(f", tool_image_{k}" for k in range(1, len(saved) + 1))
                     + ", or a direct image URL.")


def _av_image_search(self, image, **params):
    params = dict(params)
    if str(params.get("search_type", "text")) in ("reverse", "image"):
        try:
            params["image_url"] = _resolve_image_ref(self, params.get("image_url", ""))
        except ValueError as exc:
            return (f"Error: {exc}", [])
    tool = self._av_tools["image_search"]
    return (str(tool.call(params)), [])


def _av_web_search(self, image, **params):
    return (str(self._av_tools["web_search"].call(dict(params))), [])


def _av_visit(self, image, url="", goal="", **_):
    if not url:
        return ("Error: No URL provided", [])
    try:
        content = serper_scrape(url, timeout=float(os.environ.get("AGENTVISTA_VISIT_TIMEOUT", "120")))
    except Exception as exc:
        return (f"Error visiting {url}: {exc}", [])
    limit = int(os.environ.get("AGENTVISTA_VISIT_MAX_CHARS", "150000"))
    if len(content) > limit:
        content = content[:limit] + "\n\n[Content truncated due to length...]"
    if goal and os.environ.get("AGENTVISTA_VISIT_SUMMARY", "1") == "1":
        try:
            return (self._av_summarizer.summarize(content, goal, url), [])
        except Exception as exc:
            print(f"[AgentVista visit] summary failed: {type(exc).__name__}: {exc}; returning raw content", file=sys.stderr)
    return (f"Content from {url}:\n\nGoal: {goal}\n\n{content}", [])


def register_agentvista_tools(registry) -> list[dict]:
    """Register the four AgentVista tools on one per-sample registry."""
    official = _load_official()
    _serper_key()
    registry._av_tools = {
        "web_search": official.WebSearch({"max_results": 10, "timeout": 10}),
        "image_search": official.ImageSearch({"max_results": 5, "search_type": "text",
                                              "search_image_max_pixels": 1000000, "search_image_quality": 75}),
    }
    registry._av_summarizer = _SummaryVisit(official.Visit)
    registry._av_tool_images = []
    for name, fn in (("web_search", _av_web_search), ("image_search", _av_image_search),
                     ("visit", _av_visit), ("code_interpreter", _av_code_interpreter)):
        registry.register(name, types.MethodType(fn, registry))
    return AGENTVISTA_TOOL_SCHEMAS
