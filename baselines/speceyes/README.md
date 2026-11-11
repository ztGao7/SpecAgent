# SpecEyes visual-agent adapter

Official implementation: [MAC-AutoML/SpecEyes](https://github.com/MAC-AutoML/SpecEyes), pinned in `upstream/` to the commit recorded in the repository's `UPSTREAMS.json`.

## Files

```text
overlay/
  spectool_agent/speceyes.py
  VLMEvalKit/vlmeval/api/spectool_speceyes.py
upstream/                          # official Git submodule
```

The overlay contains baseline-specific files only. It preserves the existing package layout because the orchestration uses relative imports from the shared runtime. It is not an independently installable Python package.

## Adaptation behavior

1. Main routes the question using the official reliability prompt.
2. Direct-answer candidates are sent to Draft with top-K log probabilities.
3. The SpecEyes confidence gate decides whether to accept Draft's answer.
4. Remaining samples enter the shared Main-agent sandbox fallback.

The adapter uses OpenAI-compatible vLLM requests and the shared visual-tool runtime rather than the upstream implementation's local Transformers execution. See the source for batch barriers, concurrency, image handling and runtime overrides; do not treat this adaptation as an unchanged execution of upstream code.

## Requirements and use

To run the official baseline, initialize `upstream/` and follow its README.

To run this visual-agent adaptation, an existing compatible common runtime containing `spectool_agent` and the patched `VLMEvalKit` is required. That runtime is deliberately not published in this release. The two overlay files belong at the matching paths within that runtime; registration and launching also come from that runtime. **Do not overwrite a live experimental runtime with this overlay.** Use a separate copy after version compatibility has been established.

No standalone end-to-end execution or new benchmark result is claimed for this public tree.
