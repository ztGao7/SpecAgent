#!/usr/bin/env python3
"""Check the selected environment without starting services or loading weights."""
import argparse
import importlib
import json
import os
from pathlib import Path
import shutil
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--server", action="store_true")
    parser.add_argument("--core-only", action="store_true")
    parser.add_argument("--data", type=Path)
    args = parser.parse_args()
    if sys.version_info < (3, 10):
        raise SystemExit("Python >= 3.10 is required")
    if args.server:
        import torch
        import vllm
        print(json.dumps({"python": sys.version.split()[0], "vllm": vllm.__version__,
                          "torch": torch.__version__, "cuda": torch.version.cuda}))
        if vllm.__version__ != "0.25.1+cu129" or torch.__version__ != "2.11.0+cu129":
            raise SystemExit("Use the pinned vllm0251 environment")
        if not torch.cuda.is_available():
            raise SystemExit("CUDA unavailable; check NVIDIA driver / device visibility")
        value = torch.ones((32, 32), device="cuda")
        if (value @ value).sum().item() != 32768:
            raise SystemExit("CUDA tensor check failed")
        print("OK CUDA tensor operation:", torch.cuda.get_device_name(0))
        return
    root = Path(__file__).resolve().parents[1]
    sys.path[:0] = [str(root / "spectool"), str(root / "spectool/VLMEvalKit")]
    os.environ.setdefault("SPECTOOL_AGENT_ROOT", str(root / "spectool"))
    os.environ.setdefault("LITELLM_LOCAL_MODEL_COST_MAP", "True")
    modules = ["openai", "httpx", "PIL", "numpy", "requests", "redis", "ipykernel"]
    if not args.core_only:
        modules += ["torch", "torchvision", "transformers", "datasets", "pyarrow", "vlmeval"]
    for name in modules:
        module = importlib.import_module(name)
        print("OK", name, getattr(module, "__version__", ""))
    if not args.core_only:
        from vlmeval.config import supported_VLM
        for name in ["Agent-Qwen3.5-27B", "Spectool-Qwen3.5", "SpecEyes-Qwen3.5"]:
            if name not in supported_VLM:
                raise SystemExit(f"Adapter not registered: {name}")
        for command in ["redis-server", "setsid"]:
            if not shutil.which(command):
                raise SystemExit(f"Missing system command: {command}; see docs/QUICKSTART.md")
    if args.data:
        from download_data import validate_table
        print("OK dataset rows:", validate_table(args.data))


if __name__ == "__main__":
    main()
