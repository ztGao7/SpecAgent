# 快速配置与启动测试

本仓库现已包含 Spectool、SpecEyes、SpecReason 适配器、公共模型/工具代码、修改过的 VLMEvalKit 和本地 Python sandbox。所有路径可配置，无需访问原实验服务器。

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
MAIN_GPUS=0
MAIN_TP=1
DRAFT_GPUS=1
DRAFT_TP=1
```

上面 GPU 示例按两张 80GB 卡安排，Main / Draft 各占一张。使用 24GB 卡时，可将 Main 改为四卡 TP=4、Draft 单卡 TP=1（共五张）；TP 必须与对应 GPU 数量一致，Main/Draft 不要共用 GPU。27B 权重约 52 GiB、9B 约 19 GiB，还需要 KV cache、视觉编码器和运行缓冲显存；不是任意 GPU 组合都能运行。先用 `nvidia-smi` 确认机器的实际配置和空闲显存。

保存后直接运行脚本，**无需手动 `source .env` 或 `conda activate`**。`scripts/*.sh` 和根目录 `start_test.sh` 自动读取仓库根目录 `.env`，与当前终端所在目录无关。`.env` 是可信的本地 Bash 配置；路径含空格时加引号，不要加载来源不明的配置文件。

配置优先级为：**命令行参数 > 已导出的环境变量 > `.env` > 默认值**。如果之前手动导出过同名变量，先 `unset` 相应变量或换一个干净终端，避免旧值覆盖 `.env`。`.env` 中的路径建议填写绝对路径，不要依赖其他可覆盖变量推导。

需要切换实验配置时：

```bash
SPECAGENT_ENV_FILE=/绝对路径/实验.env bash scripts/run_test.sh --method spectool --bench vstar --samples 2
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
bash scripts/doctor.sh --server
```

先安装第 1 节的系统依赖，确保 `redis-server` 在 PATH 中（或已放在所选客户端环境的 `bin/` 中）。安装器不会通过 Conda 安装 Redis。它自动下载并校验 Miniconda，创建 `spectool`（客户端）和 `vllm0251`（模型服务）两个环境，安装已固定的核心版本。客户端完整依赖在 Linux x86_64 / Python 3.10 下解析并记录为 `requirements/client-linux.lock`；原始依赖清单位于 `requirements/client.txt`。客户端 Torch 仅用于库依赖，评测通过 HTTP 调用模型，不要求客户端 CUDA 可用。

`doctor.sh` 检查客户端导入、模型注册和 Redis / setsid 命令；`doctor.sh --server` 检查 vLLM、Torch、TorchCodec 的 CUDA 12.9 版本，执行真实 CUDA tensor 运算及 `vllm serve --help` 导入检查，不加载模型。任何失败都应先处理，再启动正式模型测试。

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
bash scripts/run_test.sh --method spectool --bench vstar --samples 2 --dry-run
```

然后任选方法：

```bash
bash scripts/run_test.sh --method spectool --bench vstar --samples 2
bash scripts/run_test.sh --method speceyes --bench vstar --samples 2
bash scripts/run_test.sh --method specreason --bench vstar --samples 2
bash scripts/run_test.sh --method baseline --bench vstar --samples 2
bash scripts/run_test.sh --method direct --bench vstar --samples 2
```

每条命令独立运行，使用配置中的客户端 Python，先核对环境、数据和端口，再启动需要的 vLLM / Redis / sandbox 服务、等待就绪并执行推理。新启动的服务会在该测试结束后清理；端口被占用时快速入口会报错，不会默认接入未知服务。Main/Draft 默认端口为 8126/8129，sandbox 为 18901，Redis 为 26379，可通过环境变量覆盖。

默认固定选择 TSV 中前两个不同 index，不按正确答案挑选。输出目录每次唯一，位于 `$SPECAGENT_WORK_ROOT/runs/`，包含选择的 index、服务日志、预测表及轨迹。`infer` 模式只运行推理，不依赖外部评分 API；生成预测文件不自动等同于预测正确。

SpecEyes 的单批路由大小可能大于 smoke 样本数，快速测试仅用于验证管线。SpecReason 会先把本仓库公共代码冻结到自己的、被 Git 忽略的 `runtime/`，随后使用独立控制器运行；它没有使用 Spectool 的 QJ / early-exit 决策。

已有符合配置的服务时：

```bash
bash scripts/run_test.sh --method spectool --bench vstar --samples 2 --reuse-services
```

此模式验证并使用配置端口上的现有服务，不停止它们。应确认模型服务的实际模型、参数和负载；与其他实验共享服务不适合做严谨的耗时比较。

完整数据集需显式启用：

```bash
bash scripts/run_test.sh --method spectool --bench vstar --full
```

## 6. 高级入口与故障处理

`start_test.sh --help` 是完整服务编排入口，支持 GPU/TP、端口、图像、请求/步数预算及门限等设置。快速入口额外固定了 32768 上下文、单样本并发、每个服务最多 2 条在途序列和 data URL 图片传输，便于新机器检查；这些是快速检查配置，不应与历史全量实验混用后直接比较耗时。

保留服务、暂不评测时可使用 `bash start_test.sh --spectool --services-only`。该高级模式会保留服务进程和 `.runs/<run-id>/pids` 记录，需自行管理。日常小测优先使用会自动清理自身服务的 `scripts/run_test.sh`。

| 问题 | 检查 |
| --- | --- |
| CUDA 不可用 / GPU OOM | `doctor.sh --server`、驱动、显存、GPU/TP 分组；必要时减小上下文和并发 |
| 下载失败 | 网络/代理、`HF_ENDPOINT` 和可用空间；模型下载中断后执行同一命令 |
| benchmark 缺失 | 运行 `download_data.sh --bench ...`，或检查 `LMUData` 指向的 TSV |
| Conda / Python 路径错误 | 检查 `.env` 及已导出的同名变量，运行 `setup_env.sh --dry-run`、`doctor.sh --paths --bench vstar` 查看路径 |
| 端口冲突 | 设置空闲的 `MAIN_PORT` / `DRAFT_PORT` / `SANDBOX_PORT` / `SANDBOX_REDIS_PORT`，或显式复用已确认的服务 |
| 客户端 pyarrow 被屏蔽 | 不要给客户端设置 `compat/vllm_no_pyarrow` 的 PYTHONPATH；兼容模块只用于 vLLM 进程 |

TorchCodec 固定为 `0.15.0+cu129`，避免解析到默认 CUDA 13 wheel 后出现 `libnvrtc.so.13` 缺失、导致模型服务无法启动的[上游问题](https://github.com/vllm-project/vllm/issues/63121)。

完整安装会运行依赖兼容检查。上游 [decord 0.6.0 的 Linux wheel 标签问题](https://github.com/dmlc/decord/issues/313) 会令原生 `pip check` 报平台不支持；脚本仅在 Linux x86_64、确认该版本及错误的 cp36 标签、且实际导入成功时接受这一条提示，日志会明确标记 `COMPAT`。任何其他依赖错误或导入失败仍会终止，不会跳过检查。

完整历史测试仍可执行 `bash scripts/test.sh --full`。当前导入的研究代码存在历史接口测试漂移，完整套件并非全绿；默认快速测试是单独列出的 smoke 范围，不隐藏或删除这些历史测试。GPU 全流程仍须在目标硬件、模型和数据准备好后执行验证。

参考：[vLLM wheel 及 SHA256](https://github.com/vllm-project/vllm/releases/expanded_assets/v0.25.1)、[Miniconda 安装包](https://repo.anaconda.com/miniconda/)、[Hugging Face 下载接口](https://huggingface.co/docs/huggingface_hub/guides/download)。
