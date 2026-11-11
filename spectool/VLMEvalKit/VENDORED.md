# Vendored VLMEvalKit (Spectool fork)

This directory was cloned from [open-compass/VLMEvalKit](https://github.com/open-compass/VLMEvalKit)
and is tracked as **plain files** in the parent `spectool` repo (no nested `.git`).

## Upstream pin

| Field | Value |
|-------|-------|
| Remote | `https://github.com/open-compass/VLMEvalKit.git` |
| Commit | `2d1b192` — `[FIx] fix anchor links (#1507)` |
| Branch | `main` (at clone time) |

## Spectool-local patches

Files changed relative to upstream:

| File | Change |
|------|--------|
| `vlmeval/api/spectool_agent.py` | **New** — `SpectoolAgentAPI` wrapper |
| `vlmeval/api/__init__.py` | Export `SpectoolAgentAPI` |
| `vlmeval/config.py` | Register `Spectool-Agent-Qwen3-VL-8B`, `Spectool-Raw-Qwen3-VL-8B` |
| `vlmeval/api/gpt.py` | Fallback to `reasoning` when `content` is null (Qwen3-VL / vLLM) |
| `vlmeval/dataset/image_mcq.py` | Disable VStarBench MD5 check for local TSV |
| `vlmeval/dataset/image_vqa.py` | Disable MathVision(_MINI) MD5 check for local TSV |

To refresh from upstream later: re-clone or merge from VLMEvalKit, then re-apply the patches above.
