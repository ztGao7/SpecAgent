# Initial baseline-only release

## Included

- Pinned Git submodule references to the official SpecEyes and SpecReason repositories.
- SpecEyes-specific orchestration and VLMEvalKit wrapper, kept as an overlay so their existing package-relative imports remain unchanged.
- The SpecReason visual-agent controller, endpoint client, score prompt, wrapper, runner, runtime-freezing helper and CPU tests.
- Setup boundaries and provenance documentation.

## Excluded

- The Spectool speculative controller, model/tool helpers, generation prompts, pool, sandbox implementation and launcher.
- Any frozen `runtime/` copy, including copies hidden inside other method directories.
- Vendored VLMEvalKit, private repository history, environment files, credentials and machine-specific paths.
- Model weights, benchmarks, trajectories, caches, experiment logs and backups.
- AgentVista experiments and feasibility work, which are outside this release.

The baseline adapters still import modules named `spectool_agent` and `vlmeval`. These imports identify external dependencies; the dependency implementations are not part of this tree. They have intentionally not been replaced by new approximations.

## Portability edits

Baseline inference, scoring, routing and confidence logic are preserved. Changes to the SpecReason preparation/runner only:

- `freeze_runtime.py --source` is required instead of defaulting to a private server path.
- `run_specreason.sh` uses the active Python environment and does not source a private Conda initializer.
- `--main-model` and `--draft-model` are required and must match the serving endpoints' model IDs.
- Dataset location defaults to `~/LMUData`; generated media defaults under the selected output directory. Both can be overridden through the existing environment settings.

The common runtime must be supplied separately before running adapter tests or complete visual-agent evaluations. Installing the latest stock VLMEvalKit alone does not provide the private agent helpers or timing interface these adapters expect.
