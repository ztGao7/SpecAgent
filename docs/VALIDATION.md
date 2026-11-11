# Setup validation — 2026-11-11

Checks performed for this release:

- Created a new, isolated Linux x86_64 Conda installation and Python 3.10 client environment using `setup_env.sh --core-only`. The Miniconda checksum was verified and `pip check` passed. Trusted download/package mirrors were supplied through the documented environment variables because direct downloads were slow.
- Ran the public CPU quick-test entry point in that environment: **121 passed**. The same suite passed in a separate clean Python 3.10 environment on macOS.
- Resolved the complete client requirement graph for Linux x86_64 / Python 3.10 into `requirements/client-linux.lock`. This is a dependency-resolution check, not a claim of a fresh full GPU-stack installation.
- Imported the complete public runtime, registered adapters and client dependencies using an existing compatible full client environment with CUDA hidden. The public paths, rather than the production source checkout, were placed first on the Python import path.
- Checked both existing Qwen3.5 model directories with the published offline checker: 27B has 11 shards / 1199 indexed tensors; 9B has 4 shards / 775 indexed tensors. No model inference was run by this check.
- Parsed 804 Python source files under Python 3.10 and checked shell syntax. Scanned the publication for credential patterns, private machine paths and unresolved merge markers.
- Verified side-effect-free plans for all supported quick-test methods, deterministic sample selection, missing-data failure before service startup, explicit full-dataset selection and portable environment paths.

The original, unmodified research snapshot's broader test audit reported 445 passes, 35 failures and three collection errors. Several failures reference removed interfaces; the remaining cases still need individual review. Those tests remain available through `scripts/test.sh --full`. The default 121-test smoke scope is explicit in `scripts/test.sh`; it is not represented as the entire historical suite being green.

Not claimed by these checks: a new end-to-end GPU benchmark run, new accuracy/speedup results, universal GPU memory compatibility, or byte-identical reproduction of historical model/data/configuration snapshots. Run the documented real two-sample test on the deployment hardware after environment, models and data are ready.
