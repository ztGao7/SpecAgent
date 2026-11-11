"""
网络搜索工具。

使用 DDGS（DuckDuckGo/Bing 文本搜索）从互联网获取外部知识。
当 Agent 需要图片之外的额外信息时使用（如识别地标、名人、符号含义等）。
"""
try:
    from ddgs import DDGS
except ImportError:
    from duckduckgo_search import DDGS  # 兼容旧安装

MAX_RESULTS = 5   # 默认最大搜索结果数
TIMEOUT = 30      # 搜索超时（秒）


def search(query: str, max_results: int = MAX_RESULTS, timeout: int = TIMEOUT) -> str:
    """执行网络搜索并返回格式化的结果文本。

    参数:
        query: 搜索查询字符串
        max_results: 返回的最大结果条数
        timeout: 搜索超时时间（秒）
    返回:
        格式化后的搜索结果文本（标题+摘要+URL）
    """
    try:
        with DDGS(timeout=timeout) as ddgs:
            results = list(ddgs.text(query, max_results=max_results))
    except Exception as e:
        return f"网络搜索失败: {type(e).__name__}: {e}"

    if not results:
        return f"未找到结果: {query}"

    lines = [f"网络搜索结果: {query}\n"]
    for i, r in enumerate(results, 1):
        title = r.get("title", "N/A")
        body = r.get("body", "N/A")
        href = r.get("href", "N/A")
        lines.append(f"{i}. {title}\n   {body}\n   URL: {href}\n")
    return "\n".join(lines)
