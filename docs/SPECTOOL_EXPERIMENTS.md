# Spectool 实验配置(2026-11)

本分支的 `start_test.sh --spectool` 默认使用 2026-11 实验的最终配置,并新增同步协议和"BAD 几次进 fallback"两个参数。环境安装、模型和数据下载见 [QUICKSTART](QUICKSTART.md)。

## 启动参数

| 参数 | 作用 | 默认 |
|---|---|---|
| `--spectool-config final\|orig` | `final`:最终配置(1 token GOOD/BAD 步骤判断、中等 fast 提示词、宽松答案校验,提示词在 `configs/spectool_final/`,fast 接受阈值 0.9);`orig`:不开这些开关的代码默认行为(阈值 0.95) | `final` |
| `--sync` | 步同步协议:fast 先跑完、未接受才启动 Draft;Draft 每个工具步等 Main 判断后再继续;每次生成只执行第一个工具调用 | 关(异步) |
| `--quality-bad-streak N` | 连续 N 次 BAD 步骤判断后由 Main 接管(fallback);`1` = 第一次 BAD 即 fallback;同步、异步都生效 | 2 |
| `--no-fast-answer` | 关闭 fast answer(可与 `--sync` 组合) | 开 |
| `--answerability-threshold N` | fast 接受阈值;显式给出时覆盖配置默认值 | 0.9(final) |

已导出的 `SPECTOOL_*` 环境变量优先于配置默认值;消融可用 `SPECTOOL_ABL_NO_STEP_JUDGE=1`、`SPECTOOL_ABL_NO_VERIFIER=1`。启动摘要里的 `Spectool config` 一行会显示实际配置、协议和 BAD 次数。

## MMStar / HRBench4K / MathVista 上的三组实验

```bash
# 每条命令自动启动并在结束后清理服务;GPU 按实际分配替换
for b in mmstar hrbench4k mathvista; do
  bash start_test.sh --spectool --bench $b --mode infer --main-gpus 0,1 --main-tp 2 --draft-gpus 2,3 --draft-tp 2 --api-nproc 4                          # 异步最终版
  bash start_test.sh --spectool --sync --bench $b --mode infer --main-gpus 0,1 --main-tp 2 --draft-gpus 2,3 --draft-tp 2 --api-nproc 4                   # 同步(加 --no-fast-answer 为同步关 fast)
  bash start_test.sh --spectool --quality-bad-streak 1 --bench $b --mode infer --main-gpus 0,1 --main-tp 2 --draft-gpus 2,3 --draft-tp 2 --api-nproc 4   # bad1
done
```

- 参考结果的服务设置:27B、9B 各 TP2(4090 24GB),显存占比 0.9 / 0.8(`DRAFT_GPU_MEMORY_UTILIZATION=0.8`),每个服务 8 条在途序列,4 题并发。80GB 卡可用 TP1,但耗时不能直接比较。
- 几组实验想共用一套服务时,先 `bash start_test.sh --spectool --services-only <同样的 GPU/端口参数>`,之后每条命令加 `--reuse-services` 和相同的端口参数,依次运行。
- `--mode infer` 只推理;MMStar、HRBench4K 用 `--mode all` 可直接得到准确率。MathVista 判分需要 LLM judge(在 `.env` 配 `OPENAI_API_BASE`、`OPENAI_API_KEY`、`LOCAL_LLM`),用推理型 judge 时注意 max_tokens 截断。
- 每题耗时在预测目录的 `*_sample_wall_clock.csv`,路线(fast / 校验通过 / fallback)在预测表的 `extra_records` 的 `stop_reason`。

## 与 main 的代码差异

- `speculative_agent.py`、`model.py`、`prompt.py`、`VLMEvalKit/vlmeval/api/spectool_speculative.py` 换成实验时实际运行的版本,**不含 main 上写死开启的统一前缀**;包含缺陷修复:fast 交接只查冲突不查过期、判断请求可取消、fallback 生成超时 `SPECTOOL_GENERATE_TIMEOUT`(默认 max(API 超时, 600s))、MathVision Draft 步数。
- 新增 `spectool/spectool_agent/sync_testing/`(步同步协议)、`configs/spectool_final/`(提示词)、`spectool/tests/experiments_20261114/`(CPU 测试:`$CLIENT_PYTHON <测试>.py spectool`)。

## 参考结果(完整 VStarBench 191 题,8×4090,4 题并发)

| 配置 | 答对 | 平均 / 中位耗时(s) | 加速比(平均 / 中位) |
|---|---|---|---|
| Agent-27B baseline | 183(95.81%) | 84.4 / 74.1 | 1.00× |
| 异步最终版 | 183(95.81%) | 52.1 / 23.2 | 1.62× / 3.20× |
| 同步 + fast | 178(93.19%) | 51.9 / 27.2 | 1.63× / 2.73× |
| 同步,关 fast | 184(96.34%) | 65.6 / 42.4 | 1.29× / 1.75× |

与 baseline 或异步最终版的配对符号检验 p 均 ≥ 0.12。

## 测试状态

- 实验测试三个文件全部通过;`bash scripts/test.sh` 冒烟测试全部通过;三个 bench × 三种配置(异步 / `--sync` / `--quality-bad-streak 1`)各 2 题经 `start_test.sh` 实跑通过。
- `bash scripts/test.sh --full` 的 spectool 历史测试:main 原有 37 项失败、3 项收集错误;本分支 64 项失败、4 项收集错误。新增的都是针对 main 行为的断言,原因是有意的差异:无统一前缀(`test_shared_prefix_and_replay.py`)、实验提示词与请求布局(`test_initial_fast_answer_scope.py`、`test_speculative_fallback.py`、`test_option_candidate.py`)、缺陷修复(`test_fast_answer_handoff.py`、`test_runtime_generation_config.py`)。
