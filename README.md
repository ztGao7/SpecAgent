# SpecAgent

Spectool and the SpecEyes / SpecReason visual-tool baselines, with a local vLLM + Python sandbox evaluation stack.

**Start here: [环境配置、模型下载与启动测试（中文）](docs/QUICKSTART.md).** The repository now includes the shared runtime and sandbox; no access to the original experimental server is required.

```bash
git clone https://github.com/ztGao7/SpecAgent.git
cd SpecAgent
cp configs/env.example .env
# Edit .env: choose a writable data disk and your GPU/TP configuration.
set -a; source .env; set +a

# Fast CPU-only code checks; no weights or GPU needed.
bash scripts/setup_env.sh --core-only
bash scripts/run_test.sh

# Full Linux x86_64 evaluation + serving environments.
bash scripts/setup_env.sh
bash scripts/doctor.sh
bash scripts/doctor.sh --server
bash scripts/download_models.sh
bash scripts/download_data.sh --bench vstar

# Start required services and infer on two deterministic samples.
bash scripts/run_test.sh --method spectool --bench vstar --samples 2
```

| Method | Quick test |
| --- | --- |
| Spectool | `bash scripts/run_test.sh --method spectool --samples 2` |
| SpecEyes adaptation | `bash scripts/run_test.sh --method speceyes --samples 2` |
| SpecReason adaptation | `bash scripts/run_test.sh --method specreason --samples 2` |
| Main tool agent | `bash scripts/run_test.sh --method baseline --samples 2` |
| Main direct answer | `bash scripts/run_test.sh --method direct --samples 2` |
| Draft tool agent | `bash scripts/run_test.sh --method draft-only --samples 2` |

The default benchmark is VStar. `--dry-run` prints a plan without side effects; `--full` explicitly selects the complete dataset. Existing services require `--reuse-services`. New services are bound to loopback and cleaned up after the quick test. Real model tests require compatible GPUs, sufficient VRAM, drivers, weights and data. Predictions are written in inference-only mode; a completed inference job is not an accuracy result.

## Layout

```text
scripts/                    # setup, downloads, diagnostics, smoke and benchmark launchers
requirements/               # lightweight dependencies, Linux client lock, serving pins
configs/env.example         # portable local configuration
start_test.sh               # detailed service/evaluation orchestration
spectool/spectool_agent/    # Spectool and shared model/tool/prompt code
spectool/VLMEvalKit/        # modified, vendored evaluation framework
sandbox_repo/               # local Redis/Jupyter execution service
baselines/specreason/       # separate SpecReason controller and frozen-runtime preparation
baselines/speceyes/         # adaptation documentation and pinned official submodule
```

The included visual-agent pipeline does not require initializing the official submodules. To inspect/run the original upstream implementations separately, use `git submodule update --init --recursive`.

Official SpecEyes and SpecReason implementations remain pinned Git submodules; provenance is recorded in [UPSTREAMS.json](UPSTREAMS.json). Their original task/deployment instructions are separate from these visual-agent adaptations.

## Validation and scope

The default CPU smoke suite is explicitly selected in `scripts/test.sh`. The original research tests are retained under `spectool/tests`; some still target retired interfaces. `bash scripts/test.sh --full` exposes that complete historical suite, including known failures. Fresh GPU/model/benchmark validation must be performed on the deployment machine; no new accuracy or speedup is claimed by publishing this setup.

See [release scope](docs/RELEASE_SCOPE.md) and [third-party notices](THIRD_PARTY_NOTICES.md). Model weights, datasets, output files, credentials and private repository history are excluded. The sandbox executes generated Python with the permissions of the service account and is intended for a trusted, local experimental environment.
