# SpecAgent

本仓库包含 Spectool，以及 SpecEyes、SpecReason 在视觉工具任务上的基线适配实现，提供基于 vLLM 和 Python sandbox 的本地评测流程。

**首次使用请看：[环境配置、模型下载与启动测试](docs/QUICKSTART.md)。** 公共运行时和 sandbox 已包含在仓库中，无需访问原实验服务器。

## 快速开始

自动安装适用于 Linux x86_64 和 NVIDIA GPU。先配置数据盘路径及 GPU 分组，再安装环境、下载模型和测试数据。

```bash
git clone https://github.com/ztGao7/SpecAgent.git
cd SpecAgent
cp configs/env.example .env

# 编辑 .env，设置可写的数据盘目录，以及 GPU 编号和张量并行数（TP）。
set -a; source .env; set +a

# 先跑 CPU 快速检查，无需模型权重和 GPU。
bash scripts/setup_env.sh --core-only
bash scripts/run_test.sh

# 安装完整的评测环境和模型服务环境，并检查依赖与 CUDA。
bash scripts/setup_env.sh
bash scripts/doctor.sh
bash scripts/doctor.sh --server

# 下载 Qwen3.5-27B、Qwen3.5-9B 和 VStar 测试数据。
bash scripts/download_models.sh
bash scripts/download_data.sh --bench vstar

# 启动所需服务，选取两个固定样本进行真实模型推理。
bash scripts/run_test.sh --method spectool --bench vstar --samples 2
```

环境安装会分别创建 `spectool` 客户端环境和 `vllm0251` 模型服务环境。模型、数据、缓存及环境默认保存在配置的数据盘目录中；详细版本和配置方法见[快速入门](docs/QUICKSTART.md)。

## 支持的方法

| 方法 | 快速测试命令 |
| --- | --- |
| Spectool | `bash scripts/run_test.sh --method spectool --samples 2` |
| SpecEyes 适配实现 | `bash scripts/run_test.sh --method speceyes --samples 2` |
| SpecReason 适配实现 | `bash scripts/run_test.sh --method specreason --samples 2` |
| Main 工具 Agent 基线 | `bash scripts/run_test.sh --method baseline --samples 2` |
| Main 直接回答基线 | `bash scripts/run_test.sh --method direct --samples 2` |
| Draft 工具 Agent 基线 | `bash scripts/run_test.sh --method draft-only --samples 2` |

默认数据集为 VStar，可用 `--bench` 切换。常用选项：

- `--dry-run`：只查看执行计划，不安装、下载或启动服务。
- `--full`：使用完整数据集。
- `--reuse-services`：复用已确认配置的现有服务。

默认快速测试会启动需要的本地服务，并在结束后清理本次启动的服务。真实模型测试需要匹配的 GPU、足够的显存、兼容驱动、模型权重和数据。当前入口使用仅推理模式，生成预测文件不等于完成准确率评测。

## 目录结构

```text
scripts/                    # 环境安装、下载、诊断与测试入口
requirements/               # 轻量依赖、Linux 客户端版本锁定及服务端版本约束
configs/env.example         # 本地配置示例
start_test.sh               # 完整的服务启动与评测编排脚本
spectool/spectool_agent/    # Spectool 与公共模型、工具、提示词代码
spectool/VLMEvalKit/        # 包含本项目修改的评测框架
sandbox_repo/               # 本地 Redis / Jupyter 代码执行服务
baselines/specreason/       # 独立 SpecReason 控制器与运行时快照准备工具
baselines/speceyes/         # 适配说明及固定版本的官方代码子模块
```

运行本仓库的视觉工具评测流程无需初始化官方子模块。若需查看或运行 SpecEyes、SpecReason 官方原始实现，再执行：

```bash
git submodule update --init --recursive
```

官方仓库及固定提交记录在 [UPSTREAMS.json](UPSTREAMS.json) 中。官方原始任务的部署要求与本仓库的视觉工具适配流程不同，应分别参考对应说明。

## 验证情况与使用范围

Linux 新建环境及全新 clone 后，默认 CPU 快速测试的 **121 项检查全部通过**。该测试范围在 `scripts/test.sh` 中明确列出，使用模拟模型和工具响应。详细记录见[验证说明](docs/VALIDATION.md)。

原始研究测试保留在 `spectool/tests` 中，其中部分仍针对已移除的接口，完整套件存在已知失败，可通过 `bash scripts/test.sh --full` 查看。真实 GPU 推理仍需在部署机器上验证；本次发布不代表获得了新的准确率或加速比结果。

仓库不包含模型权重、数据集内容、实验输出、凭据或原私有仓库历史。发布范围见[代码与配置说明](docs/RELEASE_SCOPE.md)，第三方代码及许可信息见[第三方声明](THIRD_PARTY_NOTICES.md)。sandbox 以服务账号的权限执行生成的 Python 代码，适用于可信的本地实验环境。
