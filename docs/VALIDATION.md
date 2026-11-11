# Setup validation — 2026-11-11

## `.env` 与启动流程修复验证

- 配置入口自动读取根目录 `.env`，支持外部配置文件；安装、按完整前缀激活环境、下载模型、读取 TSV 和启动测试使用同一组路径。macOS 与 Linux 新客户端环境当前均通过 **128 项** CPU 回归。
- 在独立前缀新建 Python 3.10 环境，运行 `setup_env.sh --core-only`，随后按 `client-linux.lock` 完整安装客户端并安装本仓库 EvalKit。Redis 使用已有兼容二进制的独立副本作为系统依赖。完整安装通过兼容检查；decord 的已知 cp36 wheel 标签提示单独保留，其余依赖错误仍会失败。
- 修复启动前对 vLLM 版本的误判：检查 distribution metadata 的 `0.25.1+cu129` 和实际 Torch/CUDA 版本，而不是要求 `vllm.__version__` 带本地版本后缀。CUDA 检查只访问本次选择的 GPU。
- 真实 Spectool / VStar 测试使用 Main=Qwen3.8-27B-FP8、Draft=Qwen3.5-9B、两张 RTX 4090（此机器单卡约 48/45 GiB），上下文 32768、TP=1+1。固定第一个 index 得到有效预测 `A`，失败率 0/1，单样本推理墙钟约 9.07 秒；新服务在退出后清理。该轮服务端使用现有 vLLM 环境，未声称完成全新服务环境安装或 A800 验证。
- 新安装的完整客户端还运行了 SpecReason / MMStar 的固定单样本测试，推理失败率 0/1，墙钟约 107.99 秒，退出后自动清理服务；服务端同样复用现有 vLLM 环境。
- 上述 GPU 测试仅检查推理流程，不计算准确率；首次服务启动含编译、视觉预热，耗时数分钟，不能把单样本推理时间当作部署总耗时。

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
