# Complete runtime and setup release

This revision includes Spectool's runtime, shared model/tool/image/prompt helpers, the local sandbox, the modified VLMEvalKit, and environment/download/test entry points. It supersedes the initial baseline-only release, whose shared runtime was intentionally omitted.

The original research behavior is retained. Deployment changes remove fixed machine paths, read Redis configuration from the environment, bind services to loopback, clean dependency-file conflict markers and add portable environment initialization. The quick entry point explicitly uses a 32768 context, one evaluation worker and data-URL image transport; these are smoke-test settings, not a reproduction of any historical timed experiment.

SpecReason runs its own sequential controller on a checksum-recorded copy of the shared runtime. Its launcher uses the active environment, model IDs are explicit, and its data/media directories are configurable. The shared service launcher now supports dispatch to that controller without importing Spectool's speculative decision policy into the baseline.

Each evaluation now keeps its predictions, existing trajectories, service logs and benchmark token report in one run directory. Token accounting uses before/after vLLM server counters, with explicit missing/reset/shared-service status, and does not change the agents' generation policies or add per-request token instrumentation.

The default smoke suite covers startup failures and selected current protocol/transport/baseline tests. Historical tests for retained features remain available and have known retired-interface failures. Tests belonging exclusively to explicitly removed scripts are removed with those scripts; this does not make the remaining full suite green.

Excluded: private Git history, model weights, benchmark contents, outputs/trajectories, generated frozen runtimes, caches, backups, credentials, environment files and experiment-specific queues. AgentVista feasibility experiments are not included.

脚本清理（2026-11-12）：删除旧随机子集工具 `spectool/scripts/make_subset.py` 和独立 F1 评分工具 `spectool/scripts/score_mmsearch_evalkit.py`；按发布范围要求进一步移除 VDR 数据转换脚本 `build_vdr_tsv.py`、VDR 协议评分脚本 `score_mmsearch_vdr_judge.py` 及其专属测试。启动入口不再支持 `--bench vdr`，在部署服务前明确报错。快速测试仍使用 `scripts/make_smoke_indices.py` 选择固定样本；其余评测、数据转换、NUMA 脚本和 `compat/` 中的 vLLM 兼容补丁继续保留。

The vendored VLMEvalKit keeps its Apache-2.0 license and upstream notices. The recorded original upstream pin in its VENDORED.md is historical; this tree includes later local modifications and should not be treated as an untouched checkout at that pin.
