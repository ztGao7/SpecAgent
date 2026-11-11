# SpecAgent — baseline implementations

This initial release contains **SpecEyes** and **SpecReason** baseline adapters and pinned references to their official implementations.

**Spectool's core implementation is not included in this release.** The visual-agent adapters depend on a separately supplied common runtime. Cloning this repository alone is therefore not sufficient to reproduce the complete visual-tool experiments.

| Baseline | Official implementation | Local adaptation | Availability |
| --- | --- | --- | --- |
| SpecEyes | `baselines/speceyes/upstream` | vLLM routing, small-model confidence gate, and shared Main-agent fallback | Adapter source included; common runtime excluded |
| SpecReason | `baselines/specreason/upstream` | Sequential Draft candidate → Main score → accept or one-turn Main rewrite | Adapter source and tests included; common runtime excluded |

## Get the official baseline code

```bash
git clone --recurse-submodules https://github.com/ztGao7/SpecAgent.git
cd SpecAgent
```

For an existing clone:

```bash
git submodule update --init --recursive
```

Upstreams are Git submodules pinned to specific commits, rather than copied into this repository with their history. Follow each upstream's README for its original tasks, dependencies, model deployment and execution. The original upstream instructions may use different models and vLLM versions from the visual-agent adaptations.

## Visual-agent adaptations

- [SpecEyes adapter](baselines/speceyes/README.md)
- [SpecReason adapter](baselines/specreason/README.md)
- [Dependency and release boundaries](docs/RELEASE_SCOPE.md)
- [Upstream provenance](UPSTREAMS.json)

These adapters use OpenAI-compatible model endpoints. Models, datasets, model-serving processes, sandbox servers and the shared agent/VLMEvalKit runtime are not bundled. No model weights, benchmark records, experiment outputs, credentials or production configuration files are included.

SpecReason's adapter also supports matched `main-only` and `draft-only` controls. Historical Spectool baseline results should not be assumed to use the same control flow or budgets.

## Licenses

Each upstream retains its own license and notices in its submodule. No license for upstream code is replaced or extended by this repository. No project-wide license has been selected for the local adaptations yet; their presence here does not imply an additional license grant.
