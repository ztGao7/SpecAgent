"""Register the adapter in an isolated, frozen VLMEvalKit process."""
import argparse
import hashlib
import importlib.metadata
import json
import os
import runpy
import sys
from dataclasses import asdict
from functools import partial
from pathlib import Path

from .core import Config

ROOT = Path(__file__).resolve().parents[1]
DATASETS = {"vstar": "VStarBench", "mmstar": "MMStar", "mathvista": "MathVista_MINI",
    "hrbench4k": "HRBench4K", "pope": "POPE"}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser()
    action = p.add_mutually_exclusive_group(required=True)
    action.add_argument("--check-only", action="store_true", help="Offline import/config check; no model requests")
    action.add_argument("--execute", action="store_true")
    p.add_argument("--data", choices=list(DATASETS) + list(DATASETS.values()), required=True)
    p.add_argument("--work-dir", type=Path, required=True)
    p.add_argument("--index-file", type=Path)
    p.add_argument("--api-nproc", type=int, default=1)
    p.add_argument("--threshold", type=int, default=7)
    p.add_argument("--max-turns", type=int, default=8)
    p.add_argument("--max-tokens", type=int, default=4096)
    p.add_argument("--max-tool-calls", type=int, default=8)
    p.add_argument("--max-total-output-tokens", type=int, default=0)
    p.add_argument("--wall-seconds", type=float, default=0)
    p.add_argument("--method", choices=["specreason", "main-only", "draft-only"], default="specreason")
    p.add_argument("--main-url", default="http://127.0.0.1:55980/v1")
    p.add_argument("--draft-url", default="http://127.0.0.1:55981/v1")
    p.add_argument("--main-model", required=True)
    p.add_argument("--draft-model", required=True)
    p.add_argument("--main-thinking", choices=["on", "off"], default="off")
    p.add_argument("--draft-thinking", choices=["on", "off"], default="on")
    p.add_argument("--main-timeout", type=int, default=180)
    p.add_argument("--draft-timeout", type=int, default=200)
    args = p.parse_args()
    args.work_dir = args.work_dir.expanduser().resolve()
    if args.index_file:
        args.index_file = args.index_file.expanduser().resolve()
    if args.api_nproc < 1:
        p.error("api-nproc must be positive")
    runtime = ROOT / "runtime"
    manifest_path = runtime / "manifest.json"
    if not manifest_path.exists():
        p.error("Run freeze_runtime.py first")
    frozen = json.loads(manifest_path.read_text())
    for relative, expected in frozen["files"].items():
        if sha(runtime / relative) != expected:
            raise SystemExit(f"Frozen runtime changed: {relative}")
    sys.path[:0] = [str(runtime), str(runtime / "VLMEvalKit")]
    dataset = DATASETS.get(args.data, args.data)
    lmu = Path(os.environ.get("LMUData", str(Path.home() / "LMUData")))
    table = lmu / f"{dataset}.tsv"
    if not table.is_file():
        raise SystemExit(f"Local dataset missing; no auto-download: {table}")
    if args.index_file and not args.index_file.is_file():
        p.error("index-file does not exist")
    # Explicitly pin common image settings; fail rather than inherit a SpecEyes
    # factor28/PNG environment or an unrelated experiment's override.
    settings = {
        "LMUData": str(lmu), "SPECTOOL_AGENT_ROOT": str(runtime),
        "SPECTOOL_IMAGE_MAX_SIZE": "0", "SPECTOOL_IMAGE_MAX_PIXELS": "12845056",
        "SPECTOOL_IMAGE_RESIZE_FACTOR": "32", "SPECTOOL_IMAGE_MIN_PIXELS": "0",
        "SPECTOOL_IMAGE_RESIZE_POLICY": "downscale", "SPECTOOL_IMAGE_FORMAT": "JPEG",
        "SPECTOOL_IMAGE_QUALITY": "90", "SPECTOOL_IMAGE_TRANSPORT": os.environ.get("SPECTOOL_IMAGE_TRANSPORT", "data"),
        "SPECTOOL_LOCAL_MEDIA_ROOT": os.environ.get("SPECTOOL_LOCAL_MEDIA_ROOT",
            str(args.work_dir / "media")),
        "SPECTOOL_TOOL_IMAGE_MAX_SIZE": "1536", "SPECTOOL_MAX_IMAGE_HISTORY": "5",
        "SPECTOOL_SANDBOX_SHARED_FS": "1",
        "SPECTOOL_SANDBOX_URL": os.environ.get("SPECTOOL_SANDBOX_URL", "http://127.0.0.1:55982/run_jupyter"),
        "SPECTOOL_SANDBOX_URLS": "",
        "SPECTOOL_SANDBOX_CODE_TIMEOUT": "200", "SPECTOOL_SANDBOX_REQUEST_TIMEOUT": "240",
        "SPECTOOL_OPENAI_MAX_RETRIES": "0", "SPECTOOL_MAX_TOKENS_CAP": "0",
        "LITELLM_LOCAL_MODEL_COST_MAP": "True",
    }
    for key, value in settings.items():
        os.environ[key] = value
    if args.index_file:
        os.environ["VLMEVAL_INDEX_FILE"] = str(args.index_file.resolve())
    else:
        os.environ.pop("VLMEVAL_INDEX_FILE", None)
    cfg = Config(threshold=args.threshold, max_turns=args.max_turns, max_tokens=args.max_tokens,
        max_tool_calls=args.max_tool_calls, max_total_output_tokens=args.max_total_output_tokens,
        wall_seconds=args.wall_seconds, mode=args.method)
    from .prompts import prompt_manifest
    from .vlmeval_api import SpecReasonAPI, atomic_json
    from vlmeval.config import supported_VLM
    name = {"specreason": "SpecReason-Agent-Qwen3.5", "main-only": "Neutral-Main-Qwen3.5",
        "draft-only": "Neutral-Draft-Qwen3.5"}[args.method]
    supported_VLM[name] = partial(SpecReasonAPI, main_model=args.main_model, main_url=args.main_url,
        draft_model=args.draft_model, draft_url=args.draft_url, config=cfg,
        raw_dir=str(args.work_dir.resolve() / "raw_trajectories"),
        main_thinking=args.main_thinking == "on", draft_thinking=args.draft_thinking == "on",
        main_timeout=args.main_timeout, draft_timeout=args.draft_timeout)
    manifest = {"method": name, "config": asdict(cfg), "prompts": prompt_manifest(),
        "main": {"model": args.main_model, "url": args.main_url, "thinking": args.main_thinking,
            "timeout": args.main_timeout},
        "draft": {"model": args.draft_model, "url": args.draft_url, "thinking": args.draft_thinking,
            "timeout": args.draft_timeout},
        "settings": settings, "dataset": dataset, "dataset_file": str(table),
        "dataset_sha256": sha(table), "index_file": str(args.index_file) if args.index_file else None,
        "index_sha256": sha(args.index_file) if args.index_file else None,
        "runtime_manifest_sha256": sha(manifest_path), "api_nproc": args.api_nproc,
        "adapter_files": {str(f.relative_to(ROOT)): sha(f) for f in sorted(
            (ROOT / "specreason_adapter").glob("*.py"))},
        "package_versions": {name: importlib.metadata.version(name)
            for name in ("openai", "transformers", "pillow")},
        "execute": args.execute, "models_loaded_by_adapter": False,
        "service_configuration_verified": False,
        "warning": "Deployment/cache isolation must be verified separately before timing. Total-token caps are comparable only if all methods enforce them."}
    if args.check_only:
        print(json.dumps(manifest, ensure_ascii=False, indent=2))
        return
    args.work_dir.mkdir(parents=True, exist_ok=False)
    atomic_json(args.work_dir / "experiment_manifest.json", manifest)
    os.chdir(runtime / "VLMEvalKit")
    sys.argv = ["run.py", "--data", dataset, "--model", name, "--mode", "infer",
        "--api-nproc", str(args.api_nproc), "--retry", "1", "--timeout", str(args.main_timeout),
        "--work-dir", str(args.work_dir.resolve()), "--keep-failed"]
    runpy.run_path(str(runtime / "VLMEvalKit/run.py"), run_name="__main__")


if __name__ == "__main__":
    main()
