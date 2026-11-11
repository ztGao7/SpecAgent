# Setup validation — 2026-11-11

## `.env` 与启动流程修复验证

- 配置入口自动读取根目录 `.env`，支持外部配置文件；安装、按完整前缀激活环境、下载模型、读取 TSV 和启动测试使用同一组路径。macOS 与 Linux 新客户端环境当前均通过 **149 项** CPU 回归。
- 在独立前缀新建 Python 3.10 环境，运行 `setup_env.sh --core-only`，随后按 `client-linux.lock` 完整安装客户端并安装本仓库 EvalKit。Redis 使用已有兼容二进制的独立副本作为系统依赖。完整安装通过兼容检查；decord 的已知 cp36 wheel 标签提示单独保留，其余依赖错误仍会失败。
- 修复启动前对 vLLM 版本的误判：检查 distribution metadata 的 `0.25.1+cu129` 和实际 Torch/CUDA 版本，而不是要求 `vllm.__version__` 带本地版本后缀。CUDA 检查只访问本次选择的 GPU。
- 真实 Spectool / VStar 测试使用 Main=Qwen3.8-27B-FP8、Draft=Qwen3.5-9B、两张 RTX 4090（此机器单卡约 48/45 GiB），上下文 32768、TP=1+1。固定第一个 index 得到有效预测 `A`，失败率 0/1，单样本推理墙钟约 9.07 秒；新服务在退出后清理。该轮服务端使用现有 vLLM 环境，未声称完成全新服务环境安装或 A800 验证。
- 新安装的完整客户端还运行了 SpecReason / MMStar 的固定单样本测试，推理失败率 0/1，墙钟约 107.99 秒，退出后自动清理服务；服务端同样复用现有 vLLM 环境。
- 从 GitHub 提交 `ffe621d` 全新 clone，未修改源文件，只配置被 Git 忽略的 `.env`，运行客户端安装、路径检查、默认 CPU 测试（128 项通过）及 README 的 Spectool / VStar 两样本命令。两个样本均生成预测，推理失败 0，退出后本次服务自动清理。该轮仍使用新客户端与已有服务环境。
- 同一 GitHub clone 的模型离线检查通过：Qwen3.8-27B-FP8 为 66 个分片 / 1606 个索引 tensor，Qwen3.5-9B 为 4 个分片 / 775 个索引 tensor。数据下载入口成功复用配置目录中的 VStar TSV（191 行），未重复下载。
- 上述 GPU 测试仅检查推理流程，不计算准确率；首次服务启动含编译、视觉预热，耗时数分钟，不能把单样本推理时间当作部署总耗时。

## 新建客户端与服务环境的最终验收

运行代码提交：`67a138b679e5d042cb7da4581653b2e81a93eaaa`。代码由公开 GitHub 仓库 clone / fast-forward 更新取得，工作树无源文件修改；仅配置被 Git 忽略的根目录 `.env`。

- 两个环境均在独立前缀创建并安装依赖，未克隆或修改原生产环境。客户端 Python 3.10；服务端 Python 3.12.16、vLLM distribution `0.25.1+cu129`、Torch `2.11.0+cu129`、TorchCodec `0.15.0+cu129`。安装器、客户端导入、服务端 CUDA 运算及实际 serving CLI 导入检查通过。
- 使用国内 pip / PyTorch 镜像与独立安装包缓存；vLLM 官方 wheel 的 SHA256 校验通过。模型权重和 TSV 复用已有文件，Redis 前置依赖为现有兼容二进制的独立副本。
- 根目录 `.env` 自动加载；未手动 source，也未额外手工修改模型服务命令。模型、bench、环境和输出目录均按配置读取。
- `bash scripts/run_test.sh`：**131 passed**，包含真实 Unix socket 绑定与连接、错误清理目标拒绝、`.env` 与导出变量的优先级回归。
- `bash scripts/run_test.sh --method spectool --bench vstar --samples 2`：两个样本完成并输出 `A`、`C`，推理失败 **0 个**，无准确率评判。模型为 Main Qwen3.8-27B-FP8 + Draft Qwen3.5-9B，GPU 4 / 5，各 TP=1，上下文 32768；使用新环境的独立编译缓存。推理墙钟约 49.95 秒，含 EvalKit 初始化的评测步骤约 60 秒，首次模型服务启动与编译另需数分钟。
- 退出后停止本次模型、Redis、sandbox 服务，并清理本次私有短通信目录。模型、benchmark 和预测记录保留。

本次实际修复包括：安装与激活环境前缀不一致、`.env` 未自动加载、模型下载目录不跟随配置、Conda Redis 包不可用、vLLM 版本字符串误判、TorchCodec 默认 CUDA 13 wheel 与 CUDA 12.9 栈不匹配，以及长工作路径超过 Unix socket 长度上限。临时通信目录由脚本自动选择并按 run / service 隔离。

验收范围是这台扩容 RTX 4090 服务器上的小样本启动流程；未验证 A800、16 卡并发调度、完整 benchmark 矩阵或准确率 / 加速比。

## DreamR 补充验收

运行代码提交：`7f7897c3943be5fcfaec68321d271e9d444fc038`。从 GitHub 全新 clone，未初始化可选上游子模块，复制已有路径配置到根目录 `.env`，复用已独立安装的客户端和 vLLM 环境、现有模型与数据。冻结运行时由脚本自动生成，工作树无源文件修改。

- 默认 CPU 入口：macOS 和 Linux 均 **149 passed**，包含 DreamR 阈值边界、候选历史隔离、并行改写、取消、正标签概率解析、请求参数及用量完整性标记测试；方法选择会覆盖遗留的 driver 环境变量。
- `bash scripts/run_test.sh --method dreamr --bench vstar --samples 2`：固定索引 0、1，分别输出 `A`、`C`，均以 `stop_reason=answer` 完成，无推理失败。使用 Qwen3.8-27B-FP8 + Qwen3.5-9B、GPU 4 / 5、TP=1+1、32768 上下文，接受阈值 0.7，起始 Main-only 轮数 0。
- 轨迹与 manifest 确认为 `mode=dreamr`：共 5 轮验证，正标签概率约 0.9424–0.9965，均接受 Draft；5 个 Main 改写请求都在验证之前启动，并在接受后取消。两个样本均标记 `usage_complete=false`，因为取消请求未返回完整 token 消耗。本轮未触发真实拒绝改写分支，该分支由 CPU 测试覆盖。
- 两样本推理墙钟合计约 48.03 秒（不含部署与 benchmark judge）。只作小样本流程验收，不计算准确率或加速比。
- 本次 Main、Draft、Redis、sandbox 已自动停止，两个私有通信目录均已删除，GPU 4–7 重新空闲；GPU 0–3 上的原实验未动。

DreamR 是本仓库的视觉工具适配：实现同轮阈值验证与 Main 改写并行，工具依赖使跨轮串行；使用现有 Qwen 权重，没有运行 SAPO 训练或官方多步前瞻流水线。官方来源、固定提交及实现范围见 [DreamR 说明](../baselines/dreamr/README.md)。全量 benchmark、A800 和 16 卡并发尚未验证。

## 初次发布的历史检查


Checks performed for this release:

- Cloned the published GitHub repository into new directories on Linux and macOS and ran the default test entry point against the separately prepared client environments: **121 passed on each**. The Linux clone did not initialize optional upstream submodules; the macOS clone did, confirming neither route relies on the old worktree.

- Created a new, isolated Linux x86_64 Conda installation and Python 3.10 client environment using `setup_env.sh --core-only`. The Miniconda checksum was verified and `pip check` passed. Trusted download/package mirrors were supplied through the documented environment variables because direct downloads were slow.
- Ran the public CPU quick-test entry point in that environment: **121 passed**. The same suite passed in a separate clean Python 3.10 environment on macOS.
- Resolved the complete client requirement graph for Linux x86_64 / Python 3.10 into `requirements/client-linux.lock`. This is a dependency-resolution check, not a claim of a fresh full GPU-stack installation.
- Imported the complete public runtime, registered adapters and client dependencies using an existing compatible full client environment with CUDA hidden. The public paths, rather than the production source checkout, were placed first on the Python import path.
- Checked both existing Qwen3.5 model directories with the published offline checker: 27B has 11 shards / 1199 indexed tensors; 9B has 4 shards / 775 indexed tensors. No model inference was run by this check.
- Parsed 804 Python source files under Python 3.10 and checked shell syntax. Scanned the publication for credential patterns, private machine paths and unresolved merge markers.
- Verified side-effect-free plans for all supported quick-test methods, deterministic sample selection, missing-data failure before service startup, explicit full-dataset selection and portable environment paths.

The original, unmodified research snapshot's broader test audit reported 445 passes, 35 failures and three collection errors. Several failures reference removed interfaces; the remaining cases still need individual review. Those tests remain available through `scripts/test.sh --full`. The default 121-test smoke scope is explicit in `scripts/test.sh`; it is not represented as the entire historical suite being green.

Not claimed by the initial checks below: new accuracy/speedup results, universal GPU memory compatibility, or byte-identical reproduction of historical model/data/configuration snapshots. See the current limited GPU smoke scope above. Run the documented real two-sample test on the deployment hardware after environment, models and data are ready.
