# R Publication Code

Use this directory for R code that directly supports a manuscript table,
figure, benchmark, sensitivity analysis, or ablation. Each protocol should
live in a separate subdirectory and provide:

1. a README defining the scientific question and fixed protocol;
2. a dataset registry containing source, version, license, and preprocessing;
3. a preflight that fails on missing APIs or dependencies;
4. deterministic analysis scripts with recorded seeds;
5. aggregation code that preserves failed and timed-out cells;
6. machine-readable timing, memory, quality, and provenance outputs;
7. exact commands for local or scheduler execution.

Do not commit datasets, credentials, container images, generated job scripts,
logs, RDS workspaces, or complete result directories here.
