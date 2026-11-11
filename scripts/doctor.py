#!/usr/bin/env python3
"""Check the selected environment without starting services or loading weights."""
import argparse
import importlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--server", action="store_true")
    parser.add_argument("--core-only", action="store_true")
    parser.add_argument("--data", type=Path)
    parser.add_argument("--paths", action="store_true", help="Check configured interpreters, models and TSV without importing GPU libraries")
    parser.add_argument("--bench", choices=["vstar", "mmstar", "hrbench4k", "hrbench8k", "pope", "mathvista"], default="vstar")
    args = parser.parse_args()
    if sys.version_info < (3, 10):
        raise SystemExit("Python >= 3.10 is required")
    if args.paths:
        from download_data import DATASETS, validate_table
        for role in ["CLIENT", "SERVER"]:
            python = os.environ[f"{role}_PYTHON"]
            prefix = Path(os.environ[f"{role}_ENV_PREFIX"]).resolve()
            command = "import sys,json; print(json.dumps({'python':sys.executable,'prefix':sys.prefix}))"
            actual = json.loads(subprocess.check_output([python, "-c", command], text=True))
            if Path(actual["prefix"]).resolve() != prefix:
                raise SystemExit(f"{role}: expected environment {prefix}, got {actual}")
            print(f"OK {role}: {actual['python']} (prefix={actual['prefix']})")
        for role in ["MAIN", "DRAFT"]:
            model = Path(os.environ[f"{role}_MODEL_PATH"])
            with (model / "config.json").open() as stream:
                config = json.load(stream)
            print(f"OK {role}: {model} (architectures={config.get('architectures')})")
        table = args.data or Path(os.environ["LMUData"]) / f"{DATASETS[args.bench][0]}.tsv"
        print(f"OK dataset: {table} (rows={validate_table(table)})")
        print("Paths verified; no model service was started. Use check_models.py to check all weight shards.")
        return
    if args.server:
        import torch
        import vllm
        distribution = version("vllm")
        codec = version("torchcodec")
        print(json.dumps({"python": sys.version.split()[0], "vllm": vllm.__version__,
                          "vllm_distribution": distribution,
                          "torch": torch.__version__, "cuda": torch.version.cuda, "torchcodec": codec}))
        # The cu129 wheel can expose __version__ == "0.25.1" while its
        # distribution metadata carries +cu129. Validate that metadata and the
        # actual Torch/CUDA runtime instead of rejecting that official build.
        if (distribution != "0.25.1+cu129"
                or torch.__version__ != "2.11.0+cu129" or torch.version.cuda != "12.9"
                or codec != "0.15.0+cu129"):
            raise SystemExit("Use the pinned vllm0251 environment")
        if not torch.cuda.is_available():
            raise SystemExit("CUDA unavailable; check NVIDIA driver / device visibility")
        value = torch.ones((32, 32), device="cuda")
        if (value @ value).sum().item() != 32768:
            raise SystemExit("CUDA tensor check failed")
        print("OK CUDA tensor operation:", torch.cuda.get_device_name(0))
        # Exercise the actual CLI import chain, including media/codec imports,
        # without loading any model weights or opening a listening socket.
        subprocess.run([str(Path(sys.executable).with_name("vllm")), "serve", "--help"],
                       check=True, stdout=subprocess.DEVNULL)
        print("OK vLLM serving CLI imports")
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
