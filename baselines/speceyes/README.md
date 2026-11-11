# SpecEyes visual-agent adaptation

Official implementation: [MAC-AutoML/SpecEyes](https://github.com/MAC-AutoML/SpecEyes), pinned in `upstream/` to the commit in `UPSTREAMS.json`.

The maintained adaptation now lives in the complete runtime:

- `spectool/spectool_agent/speceyes.py`: router, Draft confidence gate and Main-agent fallback orchestration.
- `spectool/VLMEvalKit/vlmeval/api/spectool_speceyes.py`: evaluation wrapper.

The former duplicate overlay files were removed to avoid maintaining two copies. The shared runtime and sandbox are included in this revision.

Follow [the quickstart](../../docs/QUICKSTART.md), then run:

```bash
bash scripts/run_test.sh --method speceyes --bench vstar --samples 2
```

The adaptation uses vLLM HTTP requests and shared visual-tool helpers instead of upstream local Transformers execution. Routing, confidence, batch barriers and fallback settings are defined in the maintained implementation. A small smoke run is not a throughput benchmark or a claim of unchanged upstream execution.

For the original upstream tasks, initialize `upstream/` and follow its README separately.
