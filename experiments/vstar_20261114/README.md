# VStarBench 实验(2026-11-14)

完整 VStarBench(191 题)上的实验代码:异步 Spectool 最终配置、步同步协议(fast 开/关)、2×2 消融,以及"第一次 BAD 即 fallback"开关。环境安装、模型和数据下载按仓库根目录 README 与 [docs/QUICKSTART.md](../../docs/QUICKSTART.md) 完成。

## 与 main 分支的代码差异

| 内容 | 位置 | 说明 |
|---|---|---|
| 实验用的 Spectool 主循环 | `spectool/spectool_agent/speculative_agent.py`、`model.py`、`prompt.py`、`VLMEvalKit/vlmeval/api/spectool_speculative.py` | 换成实验时实际运行的版本。**不含 main 上写死开启的统一前缀**(所有 Main 请求共用 fallback 系统提示开头);实验均在无统一前缀下运行 |
| 缺陷修复 | 同上 | fast 交接只查冲突不查过期;判断请求可取消;fallback 生成超时 `SPECTOOL_GENERATE_TIMEOUT`(默认 max(API 超时, 600s));MathVision Draft 步数 |
| 最终配置开关 | 同上 | `SPECTOOL_QJ_ONLY`(1 token GOOD/BAD 步骤判断)、`SPECTOOL_IMAGE_FIRST`、`SPECTOOL_JUDGE_ANCHOR`、`SPECTOOL_FAST_PROMPT_FILE`、`SPECTOOL_FAST_INSTR`、`SPECTOOL_REVIEW_SYSTEM_FILE`、`SPECTOOL_REVIEW_INSTRUCTION_FILE`、`SPECTOOL_FAST_HANDOFF_DETACH` |
| 步同步协议 | `spectool/spectool_agent/sync_testing/` | `SPECTOOL_SYNC=1`:fast 先跑完再启动 Draft;Draft 每个工具步等 Main 决定;每次生成只执行第一个工具调用 |
| 消融开关 | `speculative_agent.py` | `SPECTOOL_ABL_NO_STEP_JUDGE=1`、`SPECTOOL_ABL_NO_VERIFIER=1` |
| BAD 连续次数 | `speculative_agent.py` | `SPECTOOL_MAX_CONSECUTIVE_QUALITY_BAD`,默认 2;设 1 则第一次 BAD 即由 Main fallback,异步、同步都生效 |
| 提示词 | `experiments/vstar_20261114/prompts/` | 中等 fast 提示词、宽松校验、GOOD/BAD 判断提示词 |
| CPU 测试 | `spectool/tests/experiments_20261114/` | `$CLIENT_PYTHON <测试>.py spectool` |

## 运行

```bash
# .env 里配好 SPECAGENT_WORK_ROOT、模型与数据路径(见 QUICKSTART);GPU 在命令行指定
export MAIN_GPUS=0,1 DRAFT_GPUS=2,3          # 参考结果:27B TP2、9B TP2(4090 24GB)
bash experiments/vstar_20261114/run_vstar.sh services
bash experiments/vstar_20261114/run_vstar.sh baseline async_final sync_fast_on sync_fast_off
QBAD=1 bash experiments/vstar_20261114/run_vstar.sh async_final sync_fast_on sync_fast_off
```

- 结果在 `$SPECAGENT_RESULTS_ROOT/vstar_20261114/<变体>[_qbad1]_<时间>/`,日志在同名 `.out`。
- 各变体请依次运行:它们共用两个 vLLM 服务,并行会改变耗时。
- 80GB 卡可用 `MAIN_TP=1 DRAFT_TP=1 MAIN_GPUS=0 DRAFT_GPUS=1`,但耗时不能和下表直接比较。
- 打分:`$CLIENT_PYTHON experiments/vstar_20261114/compare_vstar.py baseline=<目录> async=<目录> sync_on=<目录> ...`(第一个为参照,输出准确率、平均/中位耗时、加速比、各路线、配对符号检验)。

## 参考结果(8×4090,27B/9B 各 TP2,4 并发)

| 配置 | 答对 | 平均 / 中位耗时(s) | 加速比(平均 / 中位) |
|---|---|---|---|
| Agent-27B baseline | 183(95.81%) | 84.4 / 74.1 | 1.00× |
| 异步最终版 | 183(95.81%) | 52.1 / 23.2 | 1.62× / 3.20× |
| 同步 + fast | 178(93.19%) | 51.9 / 27.2 | 1.63× / 2.73× |
| 同步,关 fast | 184(96.34%) | 65.6 / 42.4 | 1.29× / 1.75× |
| 消融:去掉 fast | 181(94.76%) | 64.7 / 36.4 | 1.30× / 2.03× |
| 消融:去掉步骤判断与答案校验 | 178(93.19%) | 35.5 / 14.2 | 2.38× / 5.22× |
| 消融:两者都去掉 | 179(93.72%) | 42.1 / 26.5 | 2.01× / 2.80× |

与 baseline 或异步最终版的配对符号检验 p 均 ≥ 0.12,准确率差异都不显著。

## 测试状态

- 实验测试 `spectool/tests/experiments_20261114/` 三个文件全部通过;仓库冒烟测试 `bash scripts/test.sh` 218 项全部通过。
- `bash scripts/test.sh --full` 的 spectool 历史测试:main 上原有 37 项失败、3 项收集错误;本分支 64 项失败、4 项收集错误。新增的都是针对 main 行为写的断言,原因是本分支有意不同:
  - 无统一前缀:`test_shared_prefix_and_replay.py`(8 项);
  - 实验代码的 fast/校验提示词和请求布局(系统提示不再是 fallback 提示):`test_initial_fast_answer_scope.py`(2)、`test_speculative_fallback.py`(5)、`test_option_candidate.py`(收集错误,`_describe_option_candidate` 已移除);
  - 缺陷修复:fast 交接不再按"过期"丢弃(`test_fast_answer_handoff.py` 3 项),生成超时至少 600s(`test_runtime_generation_config.py` 9 项)。
