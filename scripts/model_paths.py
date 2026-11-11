"""Shared destinations for model download and offline checks."""
import os
from pathlib import Path


def selected_models(resource_dir, selection):
    for role, size in [("MAIN", "27B"), ("DRAFT", "9B")]:
        if selection not in ("both", size):
            continue
        repo = os.environ.get(f"{role}_MODEL_ID", f"Qwen/Qwen3.5-{size}")
        dest = Path(os.environ.get(f"{role}_MODEL_PATH", str(resource_dir / repo))).expanduser().resolve()
        yield repo, dest
