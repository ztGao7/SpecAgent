# SpecReason visual-agent adapter

Official implementation: [ruipeterpan/specreason](https://github.com/ruipeterpan/specreason), pinned in `upstream/` to the commit recorded in the repository's `UPSTREAMS.json`.

The local `specreason_adapter/` implements sequential **Draft → Main score → accept or Main single-turn rewrite → Draft**. Rejected candidates are not executed. Main and Draft share the common runtime's task prompt; a separate constrained 0–9 scoring request evaluates each candidate before execution. The visual-agent task prompt, native tool calls and constrained digit scoring are adaptations of upstream SpecReason.

## Dependencies

The adapter requires Python 3.10+, OpenAI SDK, Pillow and a compatible common agent/VLMEvalKit runtime. **The common runtime is not included in this release.** The prompt, parser, image processing, tool sessions and timing helpers are imported from that separately supplied runtime.

For the original upstream task and installation, initialize `upstream/` and follow its own README. Those instructions are separate from the visual-agent adapter.

## Preparing an existing private runtime

Only if you already have the compatible common runtime:

```bash
cd baselines/specreason
python freeze_runtime.py --source /path/to/common-runtime
python -m unittest discover -s tests -v
```

The helper copies the supplied runtime locally, applies the sample-ID timing compatibility patch and records file hashes. It refuses to overwrite an existing snapshot. The resulting `runtime/` directory is ignored by Git and must not be uploaded as part of this baseline-only release.

## Checking and running

Activate your evaluation environment first. Model services and the sandbox must already be available. The runner does not deploy, restart or reset them.

```bash
export LMUData=/path/to/LMUData
export SPECTOOL_SANDBOX_URL=http://127.0.0.1:18901/run_jupyter

bash run_specreason.sh --check-only --data vstar \
  --work-dir /path/to/new-output \
  --main-model MAIN_SERVED_MODEL_ID --draft-model DRAFT_SERVED_MODEL_ID \
  --main-url http://127.0.0.1:8126/v1 \
  --draft-url http://127.0.0.1:8129/v1
```

`--check-only` checks the supplied runtime, dataset and configuration without making model requests. Use `--execute` in place of `--check-only` to run, only after dependencies and services have been validated. An existing output directory is refused.

Available controls include `--threshold`, `--max-turns`, `--max-tool-calls`, `--max-tokens`, `--max-total-output-tokens`, `--wall-seconds` and role-specific thinking flags. `--method main-only` and `--method draft-only` provide matched neutral-prompt controls. Defaults and accounting are recorded by the runner. No total output-token cap is active when `--max-total-output-tokens` is zero.

The included tests use mock model/tool responses, but still import the common runtime. They are not standalone tests in a fresh clone. No new GPU evaluation is claimed by this release.
