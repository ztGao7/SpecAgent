#!/usr/bin/env python3
"""Copy common CPU-side code once; never edit the production installation."""
import argparse
import hashlib
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True, help="Path to an existing, separately supplied common runtime")
    args = parser.parse_args()
    target = ROOT / "runtime"
    if target.exists():
        raise SystemExit("runtime already exists; use a new adaptation directory to freeze a new version")
    staging = ROOT / "runtime.preparing"
    if staging.exists():
        raise SystemExit("runtime.preparing exists; inspect the previous preparation first")
    staging.mkdir()
    for name in ("spectool_agent", "VLMEvalKit"):
        shutil.copytree(args.source / name, staging / name, ignore=shutil.ignore_patterns(
            ".git", ".env*", "__pycache__", "*.pyc", "._*", ".DS_Store", "*.bak*", "*.orig",
            "*.db", "*.db-shm", "*.db-wal", "outputs", "result", ".tmp", ".cache"))
    timing = staging / "VLMEvalKit/vlmeval/sample_timing.py"
    source = timing.read_text()
    old = "result = generate(message=message, dataset=dataset)"
    new = ("result = generate(message=message, dataset=dataset, **(\n"
        "            {'sample_id': sample_index} if getattr(\n"
        "                getattr(generate, '__self__', None), 'requires_sample_id', False) else {}))")
    if source.count(old) != 1:
        raise SystemExit("sample timing entry changed; inspect before patching the frozen copy")
    timing.write_text(source.replace(old, new))
    (staging / "VLMEvalKit/.env").write_text(
        "# Isolated adapter: configuration is supplied explicitly by the runner.\n")
    files = {str(p.relative_to(staging)): digest(p) for p in sorted(staging.rglob("*")) if p.is_file()}
    manifest = {"source": str(args.source.resolve()), "date": "2026-11-07",
        "environment_file": "Generated empty .env; no production environment file copied",
        "files": files, "patches": [{"file": "VLMEvalKit/vlmeval/sample_timing.py",
            "reason": "Pass the existing sample index only to wrappers that opt in via requires_sample_id; timing scope unchanged."}]}
    (staging / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    staging.rename(target)
    print(f"Frozen {len(files)} common runtime files into {target}")


if __name__ == "__main__":
    main()
