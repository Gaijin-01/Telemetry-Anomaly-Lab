# Scenario Harness

The scenario harness is a TOML-driven integration test framework for anomaly-lab experiments. It provides **semantic correctness** verification beyond simple exit-code checking.

---

## Overview

```
harness/
├── scenarios/          # *.toml scenario definitions
│   ├── 01_rz_point_anomaly.toml
│   ├── 02_rz_no_anomaly.toml
│   └── ...
├── fixtures/           # Deterministic CSV input data
│   ├── point_anomaly.csv
│   └── ...
├── expected/           # Expected result metadata (JSON)
│   └── ...
├── results/            # Created at runtime by `experiment run`
│   ├── scenario_name.json          # Raw result JSON
│   └── scenario_name_eval.json     # Semantic evaluation outcome
├── runner.py           # Scenario discovery and execution
├── evaluator.py        # Semantic correctness checks
└── __init__.py
```

---

## Quick Start

```bash
# Validate a single scenario TOML
anomaly-lab experiment validate harness/scenarios/01_rz_point_anomaly.toml

# Show what would run (without executing)
anomaly-lab experiment plan harness/scenarios/01_rz_point_anomaly.toml

# Run all scenarios
anomaly-lab experiment run

# Run a specific scenario
anomaly-lab experiment run harness/scenarios/01_rz_point_anomaly.toml

# Re-evaluate a previously saved result
anomaly-lab experiment replay harness/scenarios/01_rz_point_anomaly.toml harness/results/01_rz_point_anomaly.json
```

---

## Scenario TOML Format

Each scenario is a TOML file with a declarative contract:

```toml
name = "rz_point_anomaly_single"
type = "analyze"
detector = "robust-zscore"
features = ["x"]
fixture = "point_anomaly.csv"

expected_evidence_class = "POINT_ANOMALY"
expected_confidence_min = 0.5
expected_confidence_max = 1.0
expected_detector = "robust-zscore"
anomaly_count_min = 1
anomaly_count_max = 3

required_evidence_fields = [
    "evidence_class",
    "confidence",
    "detector",
]

required_assumptions = ["median_and_MAD"]
forbidden_assumptions = ["causal", "root_cause"]
```

See [experiment-contract.md](./experiment-contract.md) for the full schema.

---

## Fixture Data

Fixtures are small, deterministic CSV files in `harness/fixtures/`. They are:

- **Deterministic**: Same fixture always produces the same detector output (modulo stochastic seeds)
- **Minimal**: Small enough to understand at a glance (typically 5–15 rows)
- **Self-contained**: No external dependencies

Reference a fixture from a scenario with:

```toml
fixture = "point_anomaly.csv"   # resolves to harness/fixtures/point_anomaly.csv
```

Or use an absolute/relative path:

```toml
input = "path/to/my/data.csv"
```

---

## Semantic Evaluation

The evaluator (`harness/evaluator.py`) checks **meaning**, not just **mechanics**:

| Check | What it verifies |
|-------|-----------------|
| `evidence_class` | Result's evidence class matches expected (with superset relaxation) |
| `confidence` | Confidence is within `[min, max]` bounds |
| `detector` | Detector name matches exactly |
| `anomaly_count` | Anomaly count is within `[min, max]` |
| `required_assumptions` | Required assumption substrings appear in `assumptions` list |
| `forbidden_assumptions` | Forbidden substrings do NOT appear in `assumptions` |
| `supporting_metrics` | Numeric values match within relative tolerance |
| `required_evidence_fields` | Dot-path fields are present and non-null |

Exit codes are **not** checked by the evaluator — those are mechanical concerns handled by `cli_experiment.py`.

---

## Writing a New Scenario

1. **Create a fixture** in `harness/fixtures/my_fixture.csv`:

```csv
x
1.0
1.1
0.95
10.0
```

2. **Create a scenario TOML** in `harness/scenarios/NN_name.toml`:

```toml
name = "my_scenario"
type = "analyze"
detector = "robust-zscore"
features = ["x"]
fixture = "my_fixture.csv"

expected_evidence_class = "POINT_ANOMALY"
expected_confidence_min = 0.5
expected_confidence_max = 1.0
anomaly_count_min = 1
anomaly_count_max = 2

required_evidence_fields = ["evidence_class", "confidence"]
```

3. **Validate** the scenario:

```bash
anomaly-lab experiment validate harness/scenarios/NN_name.toml
```

4. **Run** it:

```bash
anomaly-lab experiment run harness/scenarios/NN_name.toml
```

5. **Check** the result in `harness/results/NN_name.json` and evaluation in `harness/results/NN_name_eval.json`.

---

## Compare Scenarios

For `type = "compare"` scenarios, the TOML looks like:

```toml
name = "compare_model_v1_vs_v2"
type = "compare"
baseline = "harness/fixtures/baseline.csv"
candidate = "harness/fixtures/candidate.csv"
features = ["x", "y"]

baseline_metadata = { model = "my-model", version = "v1", quantization = "none", concurrency = "1" }
candidate_metadata = { model = "my-model", version = "v2", quantization = "none", concurrency = "1" }

[[metrics]]
name = "x"
direction = "higher"
threshold = 0.1

[[metrics]]
name = "y"
direction = "higher"
threshold = 0.1

expected_evidence_class = "COMPARISON_DIFFERENCE"
```

---

## Result Files

After running `experiment run`:

- `harness/results/<name>.json` — The raw JSON output from the CLI (detector result or comparison result)
- `harness/results/<name>_eval.json` — The `EvaluationOutcome` dict with per-check details

The `_eval.json` file contains:

```json
{
  "scenario": "rz_point_anomaly_single",
  "passed": true,
  "checks": [
    {
      "field": "evidence_class",
      "expected": "POINT_ANOMALY",
      "actual": "POINT_ANOMALY",
      "passed": true,
      "tolerance": null,
      "note": "exact match"
    }
  ],
  "errors": [],
  "warnings": []
}
```

---

## Caching

Results are cached in `harness/results/`. To force re-running:

```bash
anomaly-lab experiment run --rerun
```

---

## CI Integration

In a CI pipeline, run the harness as:

```bash
anomaly-lab experiment run --output-dir ./harness-results
# Exit code 0 = all passed; exit code 1 = any scenario failed
```

The `--rerun` flag can be used to bypass the cache in CI environments.
