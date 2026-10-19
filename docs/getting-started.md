# Getting started

This guide installs the published package from a clean checkout and runs one analysis. The project is an exploratory telemetry library: findings are statistical signals, not diagnoses or proof of causation.

## Prerequisites

- Ubuntu/WSL instructions are shown; other platforms may need environment-specific activation commands
- Python 3.11 or newer
- Git

## Install from GitHub

```bash
git clone https://github.com/Gaijin-01/Telemetry-Anomaly-Lab.git
cd Telemetry-Anomaly-Lab
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install .
anomaly-lab --help
```

For a release-pinned install without cloning:

```bash
python -m pip install \
  "git+https://github.com/Gaijin-01/Telemetry-Anomaly-Lab.git@v0.2.0"
anomaly-lab --help
```

## First analysis

The repository includes synthetic sample data, so no private telemetry or GPU is needed:

```bash
anomaly-lab analyze examples/synthetic_telemetry.csv \
  --detector robust-zscore \
  --features p95_s,tokens_per_second \
  --output-json anomalies.json
```

Inspect the concise terminal summary and `anomalies.json`. An anomaly is a row that stands out under the selected detector and features; it is not automatically a regression or a faulty run.

## Next steps

- [Five-minute tutorial](tutorial.md): compare compatible baseline and candidate runs, then explore temporal changes.
- [Input data format](data-format.md): columns, types, ordering, and missing values.
- [CLI reference](cli-reference.md): commands and key options.
- [Interpretation guide](interpretation-guide.md): what the outputs do—and do not—mean.
