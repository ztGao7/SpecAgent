# 快速配置与启动测试

本仓库现已包含 Spectool、SpecEyes、SpecReason、DreamR 适配器、公共模型/工具代码、修改过的 VLMEvalKit 和本地 Python sandbox。所有路径可配置，无需访问原实验服务器。

自动安装面向 **Linux x86_64、glibc ≥ 2.28、NVIDIA GPU**。服务环境固定为 vLLM **0.25.1+cu129**、Torch **2.11.0+cu129**、Python 3.12；评测环境使用 Python 3.10。两个环境独立，避免依赖互相覆盖。预编译安装不要求另装 CUDA Toolkit，但驱动必须兼容，后面会实际检查 CUDA 运算。

## 1. Clone 并配置数据盘

```bash
git clone https://github.com/ztGao7/SpecAgent.git
cd SpecAgent
cp .env.example .env
```

启动本仓库管线无需额外初始化官方基线子模块；需要查看或运行官方原始实现时，再执行 `git submodule update --init --recursive`。

编辑 `.env`，至少设置：

```bash
SPECAGENT_WORK_ROOT=/data/你的用户名/specagent
```

`.env` 保存环境、资源路径和模型运行配置，GPU 编号及 TP 在每次启动时用参数指定。两张 80GB 卡可使用 `--main-gpus 0 --draft-gpus 1`，TP 默认均为 1。使用 24GB 卡时，可改为 `--main-gpus 0,1,2,3 --main-tp 4 --draft-gpus 4 --draft-tp 1`（共五张）；TP 必须与对应 GPU 数量一致，Main/Draft 不要共用 GPU。27B 权重约 52 GiB、9B 约 19 GiB，还需要 KV cache、视觉编码器和运行缓冲显存。先用 `nvidia-smi` 确认实际配置和空闲显存。

保存后直接运行脚本，**无需手动 `source .env` 或 `conda activate`**。`scripts/*.sh` 和根目录 `start_test.sh` 自动读取仓库根目录 `.env`，与当前终端所在目录无关。`.env` 是可信的本地 Bash 配置；路径含空格时加引号，不要加载来源不明的配置文件。

配置优先级为：**命令行参数 > 已导出的环境变量 > `.env` > 默认值**。如果之前手动导出过同名变量，先 `unset` 相应变量或换一个干净终端，避免旧值覆盖 `.env`。`.env` 中的路径建议填写绝对路径，不要依赖其他可覆盖变量推导。

需要切换实验配置时：

```bash
SPECAGENT_ENV_FILE=/绝对路径/实验.env bash scripts/run_test.sh --method spectool --bench vstar --samples 2 --main-gpus 0 --draft-gpus 1
```

显式指定的配置文件不存在会立即报错。默认 `.env` 不存在时，仍可只用环境变量或默认值运行。

未配置时，环境、缓存、模型和数据默认位于仓库下 `.local/`。建议将工作目录放在至少约 200 GiB 可用空间的数据盘，模型、Conda、pip/HF 缓存均使用该目录；脚本不修改 HOME 或 shell 启动文件。`.env` 不会上传 Git。

Ubuntu/Debian 的系统依赖（已有则跳过，无 sudo 权限请管理员安装）：

```bash
sudo apt-get update
sudo apt-get install -y curl git build-essential pkg-config util-linux \
  redis-server tesseract-ocr libgl1 libglib2.0-0 libcairo2-dev
```

## 2. 先跑 CPU 检查

只想检查代码能否启动，先安装轻量环境，无需 GPU、模型或数据集：

```bash
bash scripts/setup_env.sh --core-only
bash scripts/run_test.sh
```

这会创建独立的 `spectool` Python 环境，运行快速测试套件，包括解析、原生工具协议、基线控制逻辑、sandbox 状态和启动前检查。它使用 mock 模型/工具，不是模型准确率或 GPU 性能测试。

后续执行完整安装会在同一个客户端环境补齐依赖。只预览安装命令、不进行安装：

```bash
bash scripts/setup_env.sh --dry-run
```

## 3. 安装完整环境

```bash
bash scripts/setup_env.sh
bash scripts/doctor.sh
bash scripts/doctor.sh --server --gpus 0
```

先安装第 1 节的系统依赖，确保 `redis-server` 在 PATH 中（或已放在所选客户端环境的 `bin/` 中）。安装器不会通过 Conda 安装 Redis。它自动下载并校验 Miniconda，创建 `spectool`（客户端）和 `vllm0251`（模型服务）两个环境，安装已固定的核心版本。客户端完整依赖在 Linux x86_64 / Python 3.10 下解析并记录为 `requirements/client-linux.lock`；原始依赖清单位于 `requirements/client.txt`。客户端 Torch 仅用于库依赖，评测通过 HTTP 调用模型，不要求客户端 CUDA 可用。

`doctor.sh` 检查客户端导入、模型注册和 Redis / setsid 命令；`doctor.sh --server --gpus 0` 在所选 GPU 上检查 vLLM、Torch、TorchCodec 的 CUDA 12.9 版本，执行真实 CUDA tensor 运算及 `vllm serve --help` 导入检查，不加载模型。任何失败都应先处理，再启动正式模型测试。

可单独安装：

```bash
bash scripts/setup_env.sh --client-only
bash scripts/setup_env.sh --server-only
```

Miniconda 官方下载较慢时，可设置 `MINICONDA_URL` 为同名安装包的可信镜像地址；仍会校验固定 SHA256。pip 源可通过标准 `PIP_INDEX_URL` 环境变量配置。GitHub 下载慢时，可将同一官方 vLLM wheel 预先放到本机，在 `.env` 中设置 `VLLM_WHEEL=/绝对路径/vllm-0.25.1+cu129-cp38-abi3-manylinux_2_28_x86_64.whl`，或设置 `VLLM_WHEEL_URL` 为该文件的可信镜像地址；安装器始终校验固定 SHA256，不更换版本。默认下载使用 `.part` 文件，支持断点续传，低速超时后会报错而不会无限等待；已校验的本地 wheel 可在重复安装时复用。

PyTorch 官方下载源较慢时，可在 `.env` 中设置 CUDA 12.9 镜像索引（参见[交大镜像说明](https://mirror.sjtu.edu.cn/docs/pytorch-wheels)）：

```bash
PYTORCH_INDEX_URL=https://mirror.sjtu.edu.cn/pytorch-wheels/cu129/
```

这只更换下载来源，仍按约束安装 `torch==2.11.0+cu129`、`torchvision==0.26.0+cu129` 、`torchaudio==2.11.0+cu129` 和 `torchcodec==0.15.0+cu129`，不要把索引换成其他 CUDA 版本。未设置时使用 PyTorch 官方索引。

默认使用工作目录中的独立 Miniconda，两个环境分别是 `$CONDA_BASE/envs/spectool` 和 `$CONDA_BASE/envs/vllm0251`。已有合适的 Conda 时，可以设置 `CONDA_BASE`。也可将 `CLIENT_PYTHON` / `SERVER_PYTHON` 指向其他独立 Conda 环境的绝对 `bin/python` 路径；安装器在该前缀创建环境，启动器也按同一完整前缀激活，不再按环境名称查找，不依赖 `CONDA_ENVS_PATH`。

安装脚本会修改所选环境的依赖。需要隔离验证时，保留独立的 `CONDA_BASE`，不要指向正在运行实验的环境。

## 4. 下载模型和测试数据

```bash
bash scripts/download_models.sh
bash scripts/download_data.sh --bench vstar
```

默认下载 `MAIN_MODEL_ID=Qwen/Qwen3.5-27B` 和 `DRAFT_MODEL_ID=Qwen/Qwen3.5-9B`。默认目录为 `$RESOURCE_DIR/$MAIN_MODEL_ID` 和 `$RESOURCE_DIR/$DRAFT_MODEL_ID`；显式设置 `MAIN_MODEL_PATH` / `DRAFT_MODEL_PATH` 后，下载、离线权重检查和服务启动都使用相应目录。`download_models.sh --model 27B` / `--model 9B` 分别表示 Main / Draft 角色，保留旧入口命名。更换型号时同时修改仓库 ID 和路径，避免把不同型号下载进同一个目录。

下载器固定模型 commit，在模型目录内保存 `.specagent-revision.json`，单 worker 下载、失败重试，随后离线验证配置和全部权重分片。已有模型和 benchmark 时，**跳过下载**，按下一节填写现有路径。适配器和结果文件中的 `Qwen3.5` 是保留的注册名；实际使用的模型由 `MAIN_MODEL_PATH` / `DRAFT_MODEL_PATH` 决定，可在启动计划、vLLM 日志及 `/v1/models` 中核对。

网络不能访问 Hugging Face 时，在 `.env` 设置：

```bash
HF_ENDPOINT=https://hf-mirror.com
```

数据下载支持 `vstar`、`mmstar`、`hrbench4k`、`hrbench8k`、`pope`、`mathvista`。POPE / MathVista 的 OpenCompass 地址可能因网络无法访问；此时可将合法获得的对应完整 TSV 放入 `$LMUData`（默认 `$RESOURCE_DIR/LMUData`）。下载器会验证 TSV 格式、记录来源和 SHA256，不把失败的 HTML 响应当数据。

VStar 下载的是公开 TSV，不保证与历史实验中本地转换的 TSV 字节一致。复现历史结果时需使用同一数据/模型版本、图像策略和配置；快速测试不作历史结果复现声明。

### 复用已有模型与 benchmark

下面是路径写法示例，请替换成所在服务器上的真实路径：

```bash
# 写入 .env；环境保留在独立工作目录，模型和数据使用现有文件。
SPECAGENT_WORK_ROOT=/data/your-user/specagent-validation
RESOURCE_DIR=/data/your-user/existing-resources
MAIN_MODEL_ID=Qwen/Qwen3.8-27B-FP8
DRAFT_MODEL_ID=Qwen/Qwen3.5-9B
MAIN_MODEL_PATH=/data/your-user/existing-resources/Qwen/Qwen3.8-27B-FP8
DRAFT_MODEL_PATH=/data/your-user/existing-resources/Qwen/Qwen3.5-9B
LMUData=/data/your-user/existing-resources/LMUData
```

`LMUData` 必须直接包含 `VStarBench.tsv`、`MMStar.tsv` 等文件，不能只指向原始数据下载包。EvalKit 可能在该目录生成图片缓存；如需隔离缓存，复制所需 TSV 到独立数据目录，再将 `LMUData` 指向该目录。

安装完成后先检查路径：

```bash
bash scripts/doctor.sh --paths --bench vstar
bash scripts/doctor.sh --paths --bench mmstar
```

该检查不加载模型或启动服务；它会执行两个配置中的 Python，核对实际 `sys.prefix`，读取两个模型的 `config.json` 并验证指定 TSV。它不等于权重完整性检查或 GPU 推理成功。若需检查所有权重分片：

```bash
# 与其他入口一样加载 .env，使用配置的模型路径，全程离线。
bash scripts/download_models.sh --check-only
```

## 5. 启动少量真实样本测试

首先预览将运行的命令：

```bash
bash scripts/run_test.sh --method spectool --bench vstar --samples 2 --main-gpus 0 --draft-gpus 1 --dry-run
```

然后任选方法：

```bash
bash scripts/run_test.sh --method spectool --bench vstar --samples 2 --main-gpus 0 --draft-gpus 1
bash scripts/run_test.sh --method speceyes --bench vstar --samples 2 --main-gpus 0 --draft-gpus 1
bash scripts/run_test.sh --method specreason --bench vstar --samples 2 --main-gpus 0 --draft-gpus 1
bash scripts/run_test.sh --method dreamr --bench vstar --samples 2 --main-gpus 0 --draft-gpus 1
bash scripts/run_test.sh --method baseline --bench vstar --samples 2 --main-gpus 0
bash scripts/run_test.sh --method direct --bench vstar --samples 2 --main-gpus 0
```

每条命令独立运行，使用配置中的客户端 Python，先核对环境和数据，再自动分配端口、启动需要的 vLLM / Redis / sandbox 服务、等待就绪并执行推理。GPU 参数原样传给 `start_test.sh`；未指定所需 GPU 时直接报错，不会默认占用其他卡。新启动的服务会在测试结束后清理。

无需在 `.env` 中指定端口。Main / Draft 从 8126 / 8129 开始，sandbox / Redis 从 18901 / 26379 开始；遇到占用或其他实验的预留，默认每次加 100，最多尝试 50 次并检查端口范围。选择结果会显示在启动摘要，并写入本次 `RUN_DIR/pids/ports.ready`。跨工作目录的启动器通过同一用户的本机端口锁预留，避免模型加载期间的并发抢占；结束或启动失败时释放。其他不使用本启动器的程序仍可能在检查后抢占端口，此时应重试，不会自动复用其服务。

可用 `--main-port` / `--draft-port` / `--sandbox-port` / `--redis-port` 修改搜索起点；`--reuse-services` 和 `--no-services` 则使用给定的准确端口，不搜索替代端口。

默认固定选择 TSV 中前两个不同 index，不按正确答案挑选。输出目录每次唯一，位于 `$SPECAGENT_WORK_ROOT/runs/`，包含选择的 index、服务日志、预测表及轨迹。`infer` 模式只运行推理，不依赖外部评分 API；生成预测文件不自动等同于预测正确。

SpecEyes 的单批路由大小可能大于 smoke 样本数，快速测试仅用于验证管线。SpecReason 会先把本仓库公共代码冻结到自己的、被 Git 忽略的 `runtime/`，随后使用独立控制器运行；它没有使用 Spectool 的 QJ / early-exit 决策。

DreamR 使用独立适配控制器，按正标签概率阈值验证 Draft，并行生成 Main 改写；工具结果使跨轮执行保持串行，默认使用现有模型而非 SAPO 训练模型。可在 `.env` 设置 `DREAMR_ACCEPT_THRESHOLD=0.7`、`DREAMR_FIRST_MAIN_TURNS=0`。冻结运行时会自动生成，无需手动复制源码。实现范围与用量记录见 [DreamR 说明](../baselines/dreamr/README.md)。

已有符合配置的服务时：

```bash
bash scripts/run_test.sh --method spectool --bench vstar --samples 2 --reuse-services \
  --main-port 8126 --draft-port 8129 --sandbox-port 18901
```

此模式无需分配 GPU，验证并使用指定端口上的现有服务，不停止它们。应确认模型服务的实际模型、参数和负载；与其他实验共享服务不适合做严谨的耗时比较。

完整数据集需显式启用：

```bash
bash scripts/run_test.sh --method spectool --bench vstar --full --main-gpus 0 --draft-gpus 1
```

### 每轮结果目录与整个 benchmark 的 token 用量

两个入口均默认将本轮文件存入 `$SPECAGENT_WORK_ROOT/runs/<RUN_ID>/`。可指定一个新的、专用于本轮实验的目录：

```bash
# 整个 VStar benchmark；去掉 --full 则为默认两样本小测。
bash scripts/run_test.sh --method spectool --bench vstar --full \
  --main-gpus 0 --draft-gpus 1 \
  --output-dir /data/your-user/specagent/results/spectool_vstar_001

# 完整启动入口同样支持 --output-dir，默认使用整个 benchmark。
bash start_test.sh --spectool --bench vstar --main-gpus 0 --draft-gpus 1 \
  --output-dir /data/your-user/specagent/results/spectool_vstar_002
```

```text
本轮目录/
├── run_info.json             # 方法、bench、完整集/子集、起止时间和退出状态
├── token_usage.json          # 本轮输入/输出 token 总量及各模型服务的分量
├── indices.json              # 小测或显式选样时保存；全量运行没有此文件
├── logs/
│   ├── vllm_main.log
│   ├── vllm_draft.log        # 多 Draft 副本为 vllm_draft_1.log 等
│   ├── sandbox.log
│   ├── eval.log
│   └── tokens.log
├── predictions/             # EvalKit 预测表、耗时文件、已有轨迹和适配器 manifest
│   └── raw_trajectories/    # DreamR / SpecReason、标准 Agent 的独立 JSON 轨迹
├── metrics/
│   ├── before.json
│   ├── after.json
│   ├── before/*.prom        # 各模型服务的原始指标快照
│   └── after/*.prom
└── pids/                    # 本轮进程记录和最终端口选择
```

`token_usage.json` 的顶层 `input_tokens` / `output_tokens` 是 Main 和所有 Draft 服务的合计；`services` 保存各服务的分量。统计在服务就绪后、评测开始前取初值，评测结束后等待服务空闲并取末值，再停止本轮服务。使用 vLLM 的 `prompt_tokens_total` / `generation_tokens_total` 差值，无需依赖每个客户端响应是否返回 usage。指标口径见 [vLLM 0.25.1 实现](https://github.com/vllm-project/vllm/blob/v0.25.1/vllm/v1/metrics/loggers.py)，它表示服务端报告的 token 数，不代表 GPU FLOPs；输入量包含重复提交的历史上下文，不是只统计原始题目。

默认独占服务时可用于整个 bench 的汇总。使用 `--reuse-services` / `--no-services` 时仍会保存计数，但 `bench_attribution_complete=false`，因为时段内其他客户端的请求也会计入；原服务的 vLLM 日志保留在其原目录，不会复制成一份新服务日志。指标无法访问、服务重启/计数器重置或边界仍有在途请求时会注明原因，缺失值写为 `null`。推理失败或正常接收中断信号时也会尽量保存已取得的统计，并通过 `benchmark_completed` 和退出码标识未完成；强制 `kill -9` 无法执行收尾。

`--work-dir` 现在是完整启动入口中 `--output-dir` 的别名，指向本轮总目录，预测固定在其 `predictions/` 子目录。已有结果目录会被拒绝复用，避免覆盖或混入上一轮预测。服务退出时保留结果、轨迹和日志。Spectool / SpecEyes 现有轨迹仍在预测表的 `extra_records` 中，本次没有改为逐请求 token 记录。

## 6. 高级入口与故障处理

`start_test.sh --help` 是完整服务编排入口，支持 GPU/TP、端口、图像、请求/步数预算及门限等设置。快速入口额外固定了 32768 上下文、单样本并发、每个服务最多 2 条在途序列和 data URL 图片传输，便于新机器检查；这些是快速检查配置，不应与历史全量实验混用后直接比较耗时。

工作目录路径较长也可使用：脚本为每个 vLLM 服务自动选择数据盘上可写的短通信目录，避免 Unix socket 路径超过系统限制，结束后只清理本次创建的目录。若自动选择失败，可在 `.env` 指定短的可写父目录，例如 `SPECAGENT_IPC_ROOT=/data/your-user/ipc`。模型、数据和编译缓存仍使用原配置路径。

直接使用完整入口时，例如：

```bash
bash start_test.sh --spectool --bench vstar --main-gpus 0 --draft-gpus 1
```

保留服务、暂不评测时可追加 `--services-only`。该高级模式会保留服务进程和本轮目录的 `pids/` 记录，需自行管理。日常小测优先使用会自动清理自身服务的 `scripts/run_test.sh`。

| 问题 | 检查 |
| --- | --- |
| CUDA 不可用 / GPU OOM | `doctor.sh --server --gpus 0`、驱动、显存、GPU/TP 分组；必要时减小上下文和并发 |
| 下载失败 | 网络/代理、`HF_ENDPOINT` 和可用空间；模型下载中断后执行同一命令 |
| benchmark 缺失 | 运行 `download_data.sh --bench ...`，或检查 `LMUData` 指向的 TSV |
| Conda / Python 路径错误 | 检查 `.env` 及已导出的同名变量，运行 `setup_env.sh --dry-run`、`doctor.sh --paths --bench vstar` 查看路径 |
| 端口冲突 | 默认自动寻找空闲端口；检查本次 `logs/ports.log`，或用启动参数更换搜索起点。复用服务必须显式传 `--reuse-services` 和准确端口 |
| 客户端 pyarrow 被屏蔽 | 不要给客户端设置 `compat/vllm_no_pyarrow` 的 PYTHONPATH；兼容模块只用于 vLLM 进程 |

TorchCodec 固定为 `0.15.0+cu129`，避免解析到默认 CUDA 13 wheel 后出现 `libnvrtc.so.13` 缺失、导致模型服务无法启动的[上游问题](https://github.com/vllm-project/vllm/issues/63121)。

完整安装会运行依赖兼容检查。上游 [decord 0.6.0 的 Linux wheel 标签问题](https://github.com/dmlc/decord/issues/313) 会令原生 `pip check` 报平台不支持；脚本仅在 Linux x86_64、确认该版本及错误的 cp36 标签、且实际导入成功时接受这一条提示，日志会明确标记 `COMPAT`。任何其他依赖错误或导入失败仍会终止，不会跳过检查。

完整历史测试仍可执行 `bash scripts/test.sh --full`。当前导入的研究代码存在历史接口测试漂移，完整套件并非全绿；默认快速测试是单独列出的 smoke 范围，不隐藏或删除这些历史测试。GPU 全流程仍须在目标硬件、模型和数据准备好后执行验证。

参考：[vLLM wheel 及 SHA256](https://github.com/vllm-project/vllm/releases/expanded_assets/v0.25.1)、[Miniconda 安装包](https://repo.anaconda.com/miniconda/)、[Hugging Face 下载接口](https://huggingface.co/docs/huggingface_hub/guides/download)。
