raise ImportError(
    "pyarrow is intentionally disabled in the vLLM server process to avoid "
    "libarrow jemalloc state being inherited by CUDA worker processes"
)
