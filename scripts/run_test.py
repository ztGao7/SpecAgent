#!/usr/bin/env python3
"""CPU smoke by default; explicit method selects a small real benchmark run."""
import argparse
import json
import os
from pathlib import Path
import shlex
import subprocess
import time

from download_data import DATASETS

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=["offline", "spectool", "speceyes", "specreason", "dreamr", "baseline", "direct", "draft-only"], default="offline")
    parser.add_argument("--bench", choices=DATASETS, default="vstar")
    parser.add_argument("--samples", type=int, default=2)
    parser.add_argument("--full", action="store_true", help="Use the full selected dataset")
    parser.add_argument("--api-nproc", type=int, help="Samples evaluated concurrently (default: $API_NPROC or 4)")
    parser.add_argument("--reuse-services", action="store_true", help="Use, never stop, existing endpoints on configured ports")
    parser.add_argument("--dry-run", action="store_true", help="Print the plan without installing, downloading or starting anything")
    parser.add_argument("--main-gpus")
    parser.add_argument("--main-tp", type=int)
    parser.add_argument("--draft-gpus")
    parser.add_argument("--draft-tp", type=int)
    parser.add_argument("--main-port", type=int)
    parser.add_argument("--draft-port", type=int)
    parser.add_argument("--sandbox-port", type=int)
    parser.add_argument("--redis-port", type=int)
    parser.add_argument("--output-dir", type=Path, help="One directory for this run's predictions, logs and token totals")
    args = parser.parse_args()
    if args.samples < 1 or (args.main_tp is not None and args.main_tp < 1) or (args.draft_tp is not None and args.draft_tp < 1):
        parser.error("sample count and tensor parallel sizes must be positive")
    if args.api_nproc is not None and args.api_nproc < 1:
        parser.error("api-nproc must be positive")
    env = dict(os.environ)
    work = Path(env.get("SPECAGENT_WORK_ROOT", ROOT / ".local")).expanduser().resolve()
    conda = Path(env.get("CONDA_BASE", work / "miniconda3"))
    python = env.get("CLIENT_PYTHON", str(conda / "envs/spectool/bin/python"))
    server = env.get("SERVER_PYTHON", str(conda / "envs/vllm0251/bin/python"))
    env["PATH"] = str(Path(python).parent) + os.pathsep + str(conda / "bin") + os.pathsep + env.get("PATH", "")
    if args.method == "offline":
        command = ["bash", str(ROOT / "scripts/test.sh")]
        print(shlex.join(command), flush=True)
        if not args.dry_run:
            subprocess.run(command, env=env, check=True)
        return
    for key, value in {
        "MAIN_GPUS": args.main_gpus or env.get("MAIN_GPUS", ""),
        "MAIN_TP": args.main_tp or env.get("MAIN_TP", "1"),
        "DRAFT_GPUS": args.draft_gpus or env.get("DRAFT_GPUS", ""),
        "DRAFT_TP": args.draft_tp or env.get("DRAFT_TP", "1"),
        "MODEL_MAX_LEN": env.get("MODEL_MAX_LEN", "80000"),
        "MAIN_GPU_MEMORY_UTILIZATION": env.get("MAIN_GPU_MEMORY_UTILIZATION", "0.9"),
        "DRAFT_GPU_MEMORY_UTILIZATION": env.get("DRAFT_GPU_MEMORY_UTILIZATION", "0.9"),
        "MAIN_EXTRA_ARGS": env.get("MAIN_EXTRA_ARGS", "--dtype float16 --max-num-seqs 8 --enable-prefix-caching"),
        "DRAFT_EXTRA_ARGS": env.get("DRAFT_EXTRA_ARGS", "--dtype float16 --max-num-seqs 8 --enable-prefix-caching"),
        "MAIN_MAX_NUM_BATCHED_TOKENS": env.get("MAIN_MAX_NUM_BATCHED_TOKENS", "32768"),
        "DRAFT_MAX_NUM_BATCHED_TOKENS": env.get("DRAFT_MAX_NUM_BATCHED_TOKENS", "32768"),
    }.items():
        env[key] = str(value)
    resource = Path(env.get("RESOURCE_DIR", work / "resources"))
    data = Path(env.get("LMUData", resource / "LMUData"))
    dataset = DATASETS[args.bench][0]
    table = data / f"{dataset}.tsv"
    run_id = f"quick_{args.method}_{args.bench}_{time.strftime('%Y%m%d_%H%M%S')}_{time.time_ns()}"
    run = args.output_dir.expanduser().resolve() if args.output_dir else work / "runs" / run_id
    indices = run / "indices.json"
    output = run / "predictions"
    selected_method = "spectool" if args.method == "specreason" else args.method
    env.update(RUN_ID=run_id, RUN_DIR=str(run),
               IMAGE_TRANSPORT="data", SPECTOOL_IMAGE_TRANSPORT="data", LMUData=str(data), RESOURCE_DIR=str(resource))
    # --full must not inherit a subset from an earlier shell / .env configuration.
    env["INDEX_FILE"] = "" if args.full else str(indices)
    # Data URLs work with both freshly started and externally supplied services;
    # they do not depend on those services' allowed-local-media-path setting.
    env["SPECAGENT_EVAL_DRIVER"] = args.method if args.method in {"specreason", "dreamr"} else "vlmeval"
    api_nproc = args.api_nproc or int(env.get("API_NPROC", "4"))
    command = ["bash", str(ROOT / "start_test.sh"), f"--{selected_method}",
               "--bench", args.bench, "--mode", "infer", "--api-nproc", str(api_nproc),
               "--output-dir", str(run)]
    # Forward runtime allocation explicitly to the actual service launcher.
    for name in ("main_gpus", "main_tp", "draft_gpus", "draft_tp",
                 "main_port", "draft_port", "sandbox_port", "redis_port"):
        value = getattr(args, name)
        if value is not None:
            command += ["--" + name.replace("_", "-"), str(value)]
    if not args.full:
        command += ["--index-file", str(indices)]
    if args.reuse_services:
        command += ["--reuse-services"]
    print(json.dumps({"method": args.method, "dataset": dataset,
                      "eval_driver": env["SPECAGENT_EVAL_DRIVER"],
                      "samples": "all" if args.full else args.samples,
                      "output": str(run), "predictions": str(output), "main_gpus": env["MAIN_GPUS"],
                      "main_tp": env["MAIN_TP"], "draft_gpus": env["DRAFT_GPUS"],
                      "draft_tp": env["DRAFT_TP"], "api_nproc": api_nproc,
                      "reusing_services": args.reuse_services,
                      "vllm": {"max_model_len": env["MODEL_MAX_LEN"],
                               "max_num_batched_tokens": [env["MAIN_MAX_NUM_BATCHED_TOKENS"], env["DRAFT_MAX_NUM_BATCHED_TOKENS"]],
                               "gpu_memory_utilization": [env["MAIN_GPU_MEMORY_UTILIZATION"], env["DRAFT_GPU_MEMORY_UTILIZATION"]],
                               "main_extra_args": env["MAIN_EXTRA_ARGS"], "draft_extra_args": env["DRAFT_EXTRA_ARGS"]}}, indent=2), flush=True)
    print(json.dumps({"client_python": python, "server_python": server,
                      "main_model": env.get("MAIN_MODEL_PATH"), "draft_model": env.get("DRAFT_MODEL_PATH"),
                      "data_table": str(table)}, indent=2), flush=True)
    print(shlex.join(command), flush=True)
    if args.dry_run:
        return
    if (run / "run_info.json").exists() or (run / ".run-owner").exists():
        raise SystemExit(f"Run directory already contains results; choose a new --output-dir: {run}")
    if not args.reuse_services:
        if args.method != "draft-only" and not env["MAIN_GPUS"]:
            parser.error("Specify --main-gpus LIST when starting Main")
        if args.method not in {"baseline", "direct"} and not env["DRAFT_GPUS"]:
            parser.error("Specify --draft-gpus LIST when starting Draft")
    if not Path(python).is_file():
        raise SystemExit("Client environment missing. Run bash scripts/setup_env.sh --client-only")
    if not table.is_file():
        raise SystemExit(f"Dataset missing: {table}\nRun: bash scripts/download_data.sh --bench {args.bench}")
    main_model = Path(env.get("MAIN_MODEL_PATH", resource / "Qwen/Qwen3.5-27B"))
    draft_model = Path(env.get("DRAFT_MODEL_PATH", resource / "Qwen/Qwen3.5-9B"))
    required_models = [draft_model] if args.method == "draft-only" else [main_model]
    if args.method in {"spectool", "speceyes", "specreason", "dreamr"}:
        required_models.append(draft_model)
    for model in required_models:
        if not (model / "config.json").is_file():
            raise SystemExit(f"Model missing: {model}\nRun: bash scripts/download_models.sh, or configure MAIN_MODEL_PATH / DRAFT_MODEL_PATH")
    subprocess.run([python, str(ROOT / "scripts/doctor.py"), "--data", str(table)], env=env, check=True)
    if not args.reuse_services:
        probe_gpus = env["DRAFT_GPUS"] if args.method == "draft-only" else env["MAIN_GPUS"]
        subprocess.run([server, str(ROOT / "scripts/doctor.py"), "--server"],
                       env=dict(env, CUDA_VISIBLE_DEVICES=probe_gpus), check=True)
    if not args.full:
        subprocess.run([python, str(ROOT / "scripts/make_smoke_indices.py"), "--dataset", dataset,
                        "--table", str(table), "--count", str(args.samples), "--output", str(indices)], check=True)
    if args.method in {"specreason", "dreamr"}:
        adapter = ROOT / "baselines" / args.method
        if not (adapter / "runtime/manifest.json").exists():
            subprocess.run([python, str(adapter / "freeze_runtime.py"), "--source", str(ROOT / "spectool")], check=True)
    subprocess.run(command, env=env, check=True)


if __name__ == "__main__":
    main()
