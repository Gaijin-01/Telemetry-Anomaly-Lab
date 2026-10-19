# LLM Telemetry Anomaly Lab

Python 3.11+ offline toolkit for exploratory analysis of explicitly selected numeric telemetry, including LLM/GPU benchmark metrics such as latency, throughput, utilization, and memory. CSV input is supported. Outputs are statistical observations—not proof of regression, root cause, hardware failure, or causality.

## Core Rule

**An anomaly detector produces statistical evidence, not a diagnosis.**

```
Anomaly               != Regression
Anomaly               != Root Cause
Same Cluster          != Equivalence
No Significant Diff   != Equivalence
Change Point          != Causal Event
Detector Consensus    != Ground Truth
```

## Install (Ubuntu / WSL)

```bash
git clone https://github.com/Gaijin-01/Telemetry-Anomaly-Lab.git
cd Telemetry-Anomaly-Lab

python3 -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip
python -m pip install .
anomaly-lab --help
```

For contributors who want edits under `src/` to take effect without reinstalling, use `python -m pip install -e '.[dev]'` instead. The included example `examples/synthetic_telemetry.csv` is synthetic; no private/local telemetry is required. Every command prints a concise summary. `--output-json PATH` writes detailed strict JSON. Selected feature columns must be numeric and finite; they are never imputed.

## Correct Workflow

```bash
# 1. Inspect data structure
anomaly-lab inspect data.csv

# 2. Choose applicable analysis
#    (inspect output tells you what is valid)

# 3. Run analysis
anomaly-lab analyze data.csv --detector robust-zscore --features latency

# 4. Quantify uncertainty
anomaly-lab compare baseline.csv candidate.csv --paired --equivalence-margin 0.05

# 5. Report evidence class
#    (output includes evidence_class field)
```

## Workflow manual: CSV to analysis results

The commands in this workflow answer related but different questions; use it as a manual rather than a five-minute benchmark regression test. The three-row sample below illustrates a CSV shape only: concurrency changes between runs, it is not a controlled comparison, and it does not feed the commands below. Those runnable commands use the repository's 15-row synthetic fixture or separate compatible input files.

```csv
timestamp,run_id,model,concurrency,p50_s,p95_s,tokens_per_second,gpu_util_pct,vram_mib,error_rate
2026-10-19T10:00:00,r001,qwen,1,0.31,0.44,42.1,21,7200,0
2026-10-19T10:05:00,r002,qwen,4,0.52,0.91,114.2,51,10400,0
2026-10-19T10:10:00,r003,qwen,8,0.92,1.71,184.0,83,16800,0
```

Save as `runs.csv`. For commands that require repeated comparable runs, use datasets collected with controlled, matching settings. The repository's `examples/synthetic_telemetry.csv` is available for runnable smoke examples.

### 1. Find unusual individual runs

First inspect each subcommand's built-in help as needed:

```bash
anomaly-lab --help
anomaly-lab analyze --help
anomaly-lab compare --help
anomaly-lab baseline --help
anomaly-lab change-points --help
anomaly-lab regimes --help
```

Then run the simplest analysis:

```bash
anomaly-lab analyze examples/synthetic_telemetry.csv \
  --detector robust-zscore --features p95_s,tokens_per_second
```

This finds statistical outliers in the selected feature space; `--features` is required and explicitly selects numeric columns. To save full strict JSON output, add:

```bash
  --output-json result.json
```

Other supported detectors are `isolation-forest` and `dbscan`. Statistical unusualness is not by itself evidence of a regression or cause.

### 2. Compare baseline and candidate

Prepare two CSVs containing comparable repeated runs. Supply explicit metadata and per-metric direction. Values below are illustrative; use the actual values for the runs.

```bash
anomaly-lab compare baseline.csv candidate.csv \
  --features p95_s,tokens_per_second \
  --baseline-metadata model=qwen --baseline-metadata version=v1 \
  --baseline-metadata quantization=q4 --baseline-metadata concurrency=4 \
  --candidate-metadata model=qwen --candidate-metadata version=v1 \
  --candidate-metadata quantization=q4 --candidate-metadata concurrency=4 \
  --metric p95_s:lower:0.10 --metric tokens_per_second:higher:0.10 \
  --output-json comparison.json
```

The current API requires all four metadata values to match exactly. A version change itself therefore returns `NOT_COMPARABLE`; the comparator does not estimate the effect of changing versions or configurations. For the example command above, hold all four fields constant and compare repeated samples under that declared identity; do not describe it as a comparison across a version change.

### 3. Look for gradual drift

```bash
anomaly-lab baseline baseline.csv --features p95_s \
  --method ewma --alpha 0.2 --order-by timestamp --output-json baseline.json
```

Each estimate uses earlier rows only; initial warm-up estimates are null. Inspect residuals alongside the time series. This is a baseline, not an automatic regression verdict.

### 4. Locate candidate change boundaries

```bash
anomaly-lab change-points baseline.csv --feature p95_s \
  --min-segment 3 --threshold 0.10 --order-by timestamp \
  --output-json change-points.json
```

This retrospective scan reports candidate boundaries when segment medians differ by at least the chosen threshold. It needs at least two observations per side, so short samples can validly produce no candidates. A change point is not a root cause.

### 5. Explore performance regimes

```bash
anomaly-lab regimes examples/synthetic_telemetry.csv \
  --features p95_s,tokens_per_second --eps 0.7 --min-samples 2 \
  --metadata run_id --output-json regimes.json
```

DBSCAN clusters density after feature standardization and labels noise `-1`. Cluster IDs are opaque, not named or ordered performance modes; results depend on features and parameters. Metadata is returned separately and is not part of the clustering feature matrix.

## Data format

Input is a headered CSV. Select numeric columns explicitly via `--features` (or `--feature` for change-point scanning). Timestamp and run identifiers can be used for ordering or metadata but should not be treated as numeric features unless intentional. Missing or non-finite selected values are rejected; they are not imputed. Time-aware commands use CSV row order. `--order-by COLUMN`, when supplied, validates that the existing order follows that column; it does not sort rows. The order column must be non-null, unique, and strictly increasing. `compare` takes explicit run metadata on the command line; it does not infer comparability from CSV metadata columns.

## CLI reference

- `anomaly-lab analyze CSV --detector isolation-forest|dbscan|robust-zscore --features COL,...`
- `anomaly-lab compare BASELINE.csv CANDIDATE.csv --features COL,... --baseline-metadata KEY=VALUE ... --candidate-metadata KEY=VALUE ... --metric COL:higher|lower:THRESHOLD ...`
- `anomaly-lab baseline CSV --features COL,... --method rolling|ewma [--window N | --alpha A] [--order-by COL]`
- `anomaly-lab change-points CSV --feature COL --min-segment N --threshold VALUE [--order-by COL]`
- `anomaly-lab regimes CSV --features COL,... [--eps VALUE] [--min-samples N] [--metadata COL ...]`

Run `anomaly-lab COMMAND --help` for exact options. Each analysis command accepts `--output-json PATH`. JSON output rejects non-standard NaN/Infinity values; intentional baseline warm-up values serialize as `null`.

## Interpretation and limitations

- `analyze` finds statistical anomalies in selected features; it does not determine operational severity.
- `compare` evaluates an explicitly declared baseline/candidate relation; insufficient samples or incompatible metadata limit conclusions.
- `baseline` uses past observations only; residuals do not explain causes.
- `change-points` is retrospective; candidate boundaries do not identify root causes.
- `regimes` returns density clusters with opaque IDs and no intrinsic semantics.
- Regression is not causation. Comparison output states `Causality: NOT ESTABLISHED`.
- Phase G real RTX 6000 benchmark data is optional external validation and is intentionally not included. Only synthetic example data is shipped.

## Development

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
pytest -q
ruff check .
ruff format --check .
```

## Acknowledgements / Inspiration

This project was inspired by and informed by the public repository [`IndraniB2020/All_machine_learning`](https://github.com/IndraniB2020/All_machine_learning), particularly its anomaly-detection examples. Telemetry Anomaly Lab is an independent clean-room implementation. No upstream source code or data is redistributed here. The upstream repository had no explicit software license when reviewed, so it served only as reference and inspiration. Credit for the original reference goes to IndraniB2020.

## Project documentation

- [Getting started](docs/getting-started.md)
- [Five-minute tutorial](docs/tutorial.md)
- [Data format](docs/data-format.md)
- [CLI reference](docs/cli-reference.md)
- [Interpretation guide](docs/interpretation-guide.md)
- [Architecture](docs/architecture.md)
- [Detector semantics](docs/detector_semantics.md)
- [Basic analysis example](examples/basic_analysis.md)
- [Provenance and clean-room boundary](PROVENANCE.md)
- [Changelog](CHANGELOG.md)
