# Complete runtime and setup release

This revision includes Spectool's runtime, shared model/tool/image/prompt helpers, the local sandbox, the modified VLMEvalKit, and environment/download/test entry points. It supersedes the initial baseline-only release, whose shared runtime was intentionally omitted.

The original research behavior is retained. Deployment changes remove fixed machine paths, read Redis configuration from the environment, bind services to loopback, clean dependency-file conflict markers and add portable environment initialization. The quick entry point explicitly uses a 32768 context, one evaluation worker and data-URL image transport; these are smoke-test settings, not a reproduction of any historical timed experiment.

SpecReason runs its own sequential controller on a checksum-recorded copy of the shared runtime. Its launcher uses the active environment, model IDs are explicit, and its data/media directories are configurable. The shared service launcher now supports dispatch to that controller without importing Spectool's speculative decision policy into the baseline.

The default smoke suite covers startup failures and selected current protocol/transport/baseline tests. The historical full test suite is retained and has known retired-interface failures. No tests were deleted to make the full suite appear green.

Excluded: private Git history, model weights, benchmark contents, outputs/trajectories, generated frozen runtimes, caches, backups, credentials, environment files and experiment-specific queues. AgentVista feasibility experiments are not included.

脚本清理（2026-11-12）：删除未接入当前启动流程、测试或文档的旧随机子集工具 `spectool/scripts/make_subset.py` 和独立 F1 评分工具 `spectool/scripts/score_mmsearch_evalkit.py`。快速测试仍使用 `scripts/make_smoke_indices.py` 选择固定样本；保留启动器实际调用的评测入口、数据转换及 NUMA 脚本，以及已有回归测试依赖的 MMSearch judge 工具。`compat/` 中的 vLLM 兼容补丁继续保留。

The vendored VLMEvalKit keeps its Apache-2.0 license and upstream notices. The recorded original upstream pin in its VENDORED.md is historical; this tree includes later local modifications and should not be treated as an untouched checkout at that pin.
