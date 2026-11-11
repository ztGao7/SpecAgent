#!/usr/bin/env python3
"""CPU smoke by default; explicit method selects a small real benchmark run."""
import argparse
import json
import os
from pathlib import Path
import shlex
import socket
import subprocess
import time

from download_data import DATASETS

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=["offline", "spectool", "speceyes", "specreason", "baseline", "direct", "draft-only"], default="offline")
    parser.add_argument("--bench", choices=DATASETS, default="vstar")
    parser.add_argument("--samples", type=int, default=2)
    parser.add_argument("--full", action="store_true", help="Use the full selected dataset")
    parser.add_argument("--reuse-services", action="store_true", help="Use, never stop, existing endpoints on configured ports")
    parser.add_argument("--dry-run", action="store_true", help="Print the plan without installing, downloading or starting anything")
    parser.add_argument("--main-gpus")
    parser.add_argument("--main-tp", type=int)
    parser.add_argument("--draft-gpus")
    parser.add_argument("--draft-tp", type=int)
    args = parser.parse_args()
    if args.samples < 1 or (args.main_tp is not None and args.main_tp < 1) or (args.draft_tp is not None and args.draft_tp < 1):
        parser.error("sample count and tensor parallel sizes must be positive")
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
        "MAIN_GPUS": args.main_gpus or env.get("MAIN_GPUS", "0,1,2,3"),
        "MAIN_TP": args.main_tp or env.get("MAIN_TP", "4"),
        "DRAFT_GPUS": args.draft_gpus or env.get("DRAFT_GPUS", "4"),
        "DRAFT_TP": args.draft_tp or env.get("DRAFT_TP", "1"),
        "MODEL_MAX_LEN": env.get("MODEL_MAX_LEN", "32768"),
        "DRAFT_GPU_MEMORY_UTILIZATION": env.get("DRAFT_GPU_MEMORY_UTILIZATION", "0.9"),
        "MAIN_EXTRA_ARGS": env.get("MAIN_EXTRA_ARGS", "--dtype float16 --max-num-seqs 2 --enable-prefix-caching"),
        "DRAFT_EXTRA_ARGS": env.get("DRAFT_EXTRA_ARGS", "--dtype float16 --max-num-seqs 2 --enable-prefix-caching"),
        "MAIN_MAX_NUM_BATCHED_TOKENS": env.get("MAIN_MAX_NUM_BATCHED_TOKENS", "8192"),
        "DRAFT_MAX_NUM_BATCHED_TOKENS": env.get("DRAFT_MAX_NUM_BATCHED_TOKENS", "8192"),
    }.items():
        env[key] = str(value)
    resource = Path(env.get("RESOURCE_DIR", work / "resources"))
    data = Path(env.get("LMUData", resource / "LMUData"))
    dataset = DATASETS[args.bench][0]
    table = data / f"{dataset}.tsv"
    run_id = f"quick_{args.method}_{args.bench}_{time.strftime('%Y%m%d_%H%M%S')}_{time.time_ns()}"
    run = work / "runs" / run_id
    indices = run / "indices.json"
    output = run / "predictions"
    selected_method = "spectool" if args.method == "specreason" else args.method
    env.update(RUN_ID=run_id, RUN_DIR=str(run), LOG_DIR=str(work / "logs"),
               IMAGE_TRANSPORT="data", SPECTOOL_IMAGE_TRANSPORT="data", LMUData=str(data), RESOURCE_DIR=str(resource))
    # Data URLs work with both freshly started and externally supplied services;
    # they do not depend on those services' allowed-local-media-path setting.
    if args.method == "specreason":
        env["SPECAGENT_EVAL_DRIVER"] = "specreason"
    command = ["bash", str(ROOT / "start_test.sh"), f"--{selected_method}",
               "--bench", args.bench, "--mode", "infer", "--api-nproc", "1",
               "--work-dir", str(output)]
    if not args.full:
        command += ["--index-file", str(indices)]
    if args.reuse_services:
        command += ["--reuse-services"]
    print(json.dumps({"method": args.method, "dataset": dataset,
                      "samples": "all" if args.full else args.samples,
                      "output": str(output), "main_gpus": env["MAIN_GPUS"],
                      "main_tp": env["MAIN_TP"], "draft_gpus": env["DRAFT_GPUS"],
                      "draft_tp": env["DRAFT_TP"], "reusing_services": args.reuse_services}, indent=2), flush=True)
    print(shlex.join(command), flush=True)
    if args.dry_run:
        return
    if not Path(python).is_file():
        raise SystemExit("Client environment missing. Run bash scripts/setup_env.sh --client-only")
    if not table.is_file():
        raise SystemExit(f"Dataset missing: {table}\nRun: bash scripts/download_data.sh --bench {args.bench}")
    main_model = Path(env.get("MAIN_MODEL_PATH", resource / "Qwen/Qwen3.5-27B"))
    draft_model = Path(env.get("DRAFT_MODEL_PATH", resource / "Qwen/Qwen3.5-9B"))
    required_models = [draft_model] if args.method == "draft-only" else [main_model]
    if args.method in {"spectool", "speceyes", "specreason"}:
        required_models.append(draft_model)
    for model in required_models:
        if not (model / "config.json").is_file():
            raise SystemExit(f"Model missing: {model}\nRun: bash scripts/download_models.sh, or configure MAIN_MODEL_PATH / DRAFT_MODEL_PATH")
    subprocess.run([python, str(ROOT / "scripts/doctor.py"), "--data", str(table)], env=env, check=True)
    if not args.reuse_services:
        subprocess.run([server, str(ROOT / "scripts/doctor.py"), "--server"], env=env, check=True)
        ports = [int(env.get("MAIN_PORT", "8126"))]
        if args.method not in {"direct", "baseline"}:
            ports.append(int(env.get("DRAFT_PORT", "8129")))
        if args.method != "direct":
            ports += [int(env.get("SANDBOX_PORT", "18901")), int(env.get("SANDBOX_REDIS_PORT", "26379"))]
        for port in ports:
            with socket.socket() as listener:
                try:
                    listener.bind(("127.0.0.1", port))
                except OSError as exc:
                    raise SystemExit(f"Port {port} is busy. Choose free ports or explicitly use --reuse-services.") from exc
    if not args.full:
        subprocess.run([python, str(ROOT / "scripts/make_smoke_indices.py"), "--dataset", dataset,
                        "--table", str(table), "--count", str(args.samples), "--output", str(indices)], check=True)
    if args.method == "specreason":
        adapter = ROOT / "baselines/specreason"
        if not (adapter / "runtime/manifest.json").exists():
            subprocess.run([python, str(adapter / "freeze_runtime.py"), "--source", str(ROOT / "spectool")], check=True)
    subprocess.run(command, env=env, check=True)


if __name__ == "__main__":
    main()
