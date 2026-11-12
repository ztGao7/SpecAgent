# SpecAgent

本仓库包含 Spectool，以及 SpecEyes、SpecReason、DreamR 在视觉工具任务上的基线适配实现，提供基于 vLLM 和 Python sandbox 的本地评测流程。

**首次使用请看：[环境配置、模型下载与启动测试](docs/QUICKSTART.md)。** 公共运行时和 sandbox 已包含在仓库中，无需访问原实验服务器。

## 快速开始

自动安装适用于 Linux x86_64 和 NVIDIA GPU。先配置数据盘路径，再安装环境、下载模型和测试数据；GPU 分组通过启动参数指定。

```bash
git clone https://github.com/ztGao7/SpecAgent.git
cd SpecAgent
cp .env.example .env

# 编辑 .env，设置可写的数据盘目录及模型、数据路径。GPU 编号在启动命令中指定。
# 所有入口自动读取仓库根目录 .env，无需 source 或激活环境。

# Ubuntu/Debian 系统依赖（已安装可跳过；无 sudo 权限请管理员准备）。
sudo apt-get update
sudo apt-get install -y curl git build-essential pkg-config util-linux \
  redis-server tesseract-ocr libgl1 libglib2.0-0 libcairo2-dev

# 先跑 CPU 快速检查，无需模型权重和 GPU。
bash scripts/setup_env.sh --core-only
bash scripts/run_test.sh

# 安装完整的评测环境和模型服务环境，并检查依赖与 CUDA。
bash scripts/setup_env.sh
bash scripts/doctor.sh
bash scripts/doctor.sh --server --gpus 0

# 下载 Qwen3.5-27B、Qwen3.5-9B 和 VStar 测试数据。
bash scripts/download_models.sh
bash scripts/download_data.sh --bench vstar
bash scripts/doctor.sh --paths --bench vstar

# 启动所需服务，选取两个固定样本进行真实模型推理。
bash scripts/run_test.sh --method spectool --bench vstar --samples 2 --main-gpus 0 --draft-gpus 1
```

国内网络下载较慢时，可在 `.env` 启用示例中的 `PIP_INDEX_URL`、`PYTORCH_INDEX_URL` 和 `HF_ENDPOINT`；GitHub wheel 下载受限时，按[快速入门](docs/QUICKSTART.md)设置 `VLLM_WHEEL` 本地包或 `VLLM_WHEEL_URL`。首次安装需要下载数 GB 依赖，耗时取决于网络。

已有模型和数据时，在 `.env` 中设置 `MAIN_MODEL_PATH`、`DRAFT_MODEL_PATH` 和 `LMUData`，跳过下载；用 `doctor.sh --paths --bench vstar` 检查实际路径。配置优先级为命令行参数、已导出的环境变量、`.env`、默认值。`.env` 被 Git 忽略；多份配置可用 `SPECAGENT_ENV_FILE=/绝对路径/实验.env bash scripts/run_test.sh ...` 选择。

环境安装会分别创建 `spectool` 客户端环境和 `vllm0251` 模型服务环境。模型、数据、缓存及环境默认保存在配置的数据盘目录中；详细版本和配置方法见[快速入门](docs/QUICKSTART.md)。

## A800 80GB：两卡运行完整 benchmark

完成环境安装并在 `.env` 配好模型和数据路径后，先设置结果根目录（替换为自己的可写数据盘路径）：

```bash
# 写入仓库根目录 .env；每轮结果自动保存在此目录的 <RUN_ID>/ 子目录。
SPECAGENT_RESULTS_ROOT=/data/your-user/specagent/results
```

然后在仓库根目录执行。下面使用两张空闲的 A800 80GB：GPU 0 部署 Main（默认 Qwen3.5-27B），GPU 1 部署 Draft（默认 Qwen3.5-9B），每个模型单卡 TP=1。

```bash
bash start_test.sh \
  --spectool --bench vstar --mode infer \
  --main-gpus 0 --main-tp 1 \
  --draft-gpus 1 --draft-tp 1 \
  --api-nproc 4
```

将 `0`、`1` 替换为分配给本轮实验的 GPU 编号。此命令运行完整 VStar 的推理，不需要传 `--full`；`--api-nproc 4` 表示同时评测 4 个样本。未覆盖配置时使用 80000 上下文、32768 批处理预算和每个服务最多 8 条在途序列。端口自动避让占用，结束后清理本轮服务，保留本轮目录中的预测、已有轨迹、vLLM 日志及 `token_usage.json`。该 A800 配置示例尚未在目标服务器实测。

## 支持的方法

| 方法 | 快速测试命令 |
| --- | --- |
| Spectool | `bash scripts/run_test.sh --method spectool --samples 2 --main-gpus 0 --draft-gpus 1` |
| SpecEyes 适配实现 | `bash scripts/run_test.sh --method speceyes --samples 2 --main-gpus 0 --draft-gpus 1` |
| DreamR 视觉工具适配 | `bash scripts/run_test.sh --method dreamr --samples 2 --main-gpus 0 --draft-gpus 1` |
| SpecReason 适配实现 | `bash scripts/run_test.sh --method specreason --samples 2 --main-gpus 0 --draft-gpus 1` |
| Main 工具 Agent 基线 | `bash scripts/run_test.sh --method baseline --samples 2 --main-gpus 0` |
| Main 直接回答基线 | `bash scripts/run_test.sh --method direct --samples 2 --main-gpus 0` |
| Draft 工具 Agent 基线 | `bash scripts/run_test.sh --method draft-only --samples 2 --draft-gpus 1` |

GPU 参数会传给 `start_test.sh`，Main / Draft 的 TP 默认均为 1；多卡部署需同时指定 `--main-tp` / `--draft-tp`。默认启动会自动选择空闲端口，遇到已有 vLLM 也会跳过；仅 `--reuse-services` 显式复用服务。

默认数据集为 VStar，可用 `--bench` 切换。常用选项：

- `--dry-run`：只查看执行计划，不安装、下载或启动服务。
- `--full`：使用完整数据集。
- `--reuse-services`：复用已确认配置的现有服务。
- `--output-dir DIR`：指定本轮总目录，预测、已有轨迹、服务日志和 bench token 汇总都保存在其中。

默认快速测试会启动需要的本地服务，并在结束后清理本次启动的服务。真实模型测试需要匹配的 GPU、足够的显存、兼容驱动、模型权重和数据。当前入口使用仅推理模式，生成预测文件不等于完成准确率评测。

两个启动入口读取 `.env` 的 `SPECAGENT_RESULTS_ROOT`，输出到其 `<RUN_ID>/` 子目录；未设置时使用 `$SPECAGENT_WORK_ROOT/runs/` 作为结果根目录。其中 `logs/` 保存 vLLM / sandbox / 评测日志，`predictions/` 保存预测和已有轨迹，`token_usage.json` 保存本轮输入、输出 token 总量及 Main / Draft 分量。按 vLLM 服务端计数器的评测前后差值统计；做方法对比时使用独占服务。目录结构和计数口径见[快速入门](docs/QUICKSTART.md#每轮结果目录与整个-benchmark-的-token-用量)。

## 目录结构

```text
scripts/                    # 环境安装、下载、诊断与测试入口
requirements/               # 轻量依赖、Linux 客户端版本锁定及服务端版本约束
.env.example                # 本地配置模板；复制成 .env 后自动加载
start_test.sh               # 完整的服务启动与评测编排脚本
spectool/spectool_agent/    # Spectool 与公共模型、工具、提示词代码
spectool/VLMEvalKit/        # 包含本项目修改的评测框架
sandbox_repo/               # 本地 Redis / Jupyter 代码执行服务
baselines/dreamr/           # DreamR 阈值验证与并行改写适配
baselines/specreason/       # 独立 SpecReason 控制器与运行时快照准备工具
baselines/speceyes/         # 适配说明及固定版本的官方代码子模块
```

运行本仓库的视觉工具评测流程无需初始化官方子模块。若需查看或运行 SpecEyes、SpecReason 官方原始实现，再执行：

```bash
git submodule update --init --recursive
```

官方仓库及固定提交记录在 [UPSTREAMS.json](UPSTREAMS.json) 中。官方原始任务的部署要求与本仓库的视觉工具适配流程不同，应分别参考对应说明。

## 验证情况与使用范围

默认 CPU 快速测试目前为 **187 项，已在 macOS 和 Linux 新客户端环境通过**，包括 `.env`、GPU 参数传递、真实端口占用、并发端口预留、整轮输出目录和 bench token 汇总回归。token 汇总已用实际 HTTP 指标服务验证正常及失败路径，并只读核对现有 vLLM 0.25.1 的指标格式。2026-11-11 已从 GitHub clone，使用新建的客户端与 vLLM 环境、现有模型及数据，在两张扩容 4090（约 48/45 GiB）上完成 Spectool / VStar 两样本真实推理，两个样本均生成有效预测，本次服务及通信目录自动清理。模型为 Qwen3.8-27B-FP8 + Qwen3.5-9B。DreamR 也已从 GitHub 全新 clone 后完成 VStar 两样本推理，核对了阈值验证、并行改写和取消记录。2026-11-12 的修改通过 CPU / socket / HTTP 检查，因 GPU 均被现有实验占用，本轮未重跑模型推理或全量 bench。详细范围见[验证说明](docs/VALIDATION.md)。

原始研究测试保留在 `spectool/tests` 中，其中部分仍针对已移除的接口，完整套件存在已知失败，可通过 `bash scripts/test.sh --full` 查看。目标机器仍需执行小样本验证；以上检查不代表全 benchmark 准确率、加速比或 A800 性能已经验证。

仓库不包含模型权重、数据集内容、实验输出、凭据或原私有仓库历史。发布范围见[代码与配置说明](docs/RELEASE_SCOPE.md)，第三方代码及许可信息见[第三方声明](THIRD_PARTY_NOTICES.md)。sandbox 以服务账号的权限执行生成的 Python 代码，适用于可信的本地实验环境。
