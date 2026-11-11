# DREAM-R 视觉工具适配

官方实现：[HuYunhai-Alex/DREAM-R](https://github.com/HuYunhai-Alex/DREAM-R)，固定提交 `8a37dd2d7fd5659d28538f46419660b29a193463`，可选子模块位于 `upstream/`。本目录迁入现有服务器上的视觉工具适配实现，使用独立的 `dreamr_adapter` 包，运行时从本仓库公共代码冻结，不读取原服务器源码。

## 启动

与其他方法共用根目录 `.env`、客户端和 vLLM 环境、模型与 benchmark 路径。完成[快速配置](../../docs/QUICKSTART.md)后，在仓库根目录执行：

```bash
bash scripts/run_test.sh --method dreamr --bench vstar --samples 2
# 只查看计划：
bash scripts/run_test.sh --method dreamr --bench mmstar --samples 2 --dry-run
```

入口自动准备本地 `runtime/`、启动 Main / Draft / sandbox、运行冻结的 EvalKit，并在退出时清理本次服务。无需初始化上游子模块，也无需安装上游 RL 训练依赖。完整 benchmark 使用 `--full`，复用已部署服务使用 `--reuse-services`。

`.env` 中可配置：

```bash
DREAMR_ACCEPT_THRESHOLD=0.7
DREAMR_FIRST_MAIN_TURNS=0
```

前者是接受 Draft 的正标签概率阈值，后者是前多少轮仅由 Main 生成；默认从第一轮使用 Draft。模型服务需返回 top-20 logprobs，统一启动器默认的 vLLM 配置满足此要求。

## 实现范围

每轮由 Draft 提出下一步，然后 Main 同时进行阈值验证（TBVM）和独立改写。正标签归一化概率达到阈值且候选格式有效时接受 Draft，并取消或丢弃改写；否则使用 Main 改写。被拒候选不会执行工具，也不会进入已提交历史。

这是面向视觉工具 Agent 的适配：一轮可包含工具调用或最终答案。跨轮要等待工具结果，没有官方跨多步前瞻流水线。这里使用 `.env` 配置的现有模型，没有执行 SAPO 训练；官方训练和原始推理实现保留在可选上游子模块中。因此结果应标注为本仓库的视觉工具适配，不直接引用官方论文的复现或性能结论。

公共提示词、工具、图像处理和计时接口与其他视觉适配共用；DreamR 的验证和并行改写由自己的控制器执行，不调用 Spectool 的 QJ / early-exit 决策。该实现沿用本地 SpecReason 适配器的传输与账本结构，SpecReason 目录保持独立。

## 记录与检查

`experiment_manifest.json` 记录阈值、模型实际路径、服务地址、图像配置、数据 SHA256 和冻结运行时哈希。轨迹记录验证概率、采用 Draft / Main 的来源、改写取消状态和请求用量。被取消的请求可能没有返回 token usage，账本会标记 `usage_complete=false`，不能把已知 token 数当作完整消耗。

默认 CPU 套件包含 DreamR 的阈值边界、候选隔离、并行改写、取消、概率解析和实际 OpenAI 请求参数检查：

```bash
bash scripts/run_test.sh
```

单独离线检查配置可在冻结后运行（不发送模型请求）：

```bash
python baselines/dreamr/freeze_runtime.py --source spectool
bash baselines/dreamr/run_dreamr.sh --check-only --data vstar \
  --work-dir /path/to/new-output --method dreamr \
  --main-model /path/to/main --draft-model /path/to/draft
```

冻结快照和输出被 Git 忽略。小样本验收结果另见[验证记录](../../docs/VALIDATION.md)。
