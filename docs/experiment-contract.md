# Experiment Contract

A **scenario contract** is a machine-readable specification (TOML) that defines:

- What input data to use (inline fixture or file path)
- Which anomaly-lab command to invoke (`analyze`, `compare`, etc.)
- The expected **evidence contract** — the semantic properties the result must satisfy
- Tolerance and bounds for numeric comparisons

The contract is **not** a test script. It does not encode assertions as Python code; instead it declares what the output *should mean* if the experiment is behaving correctly.

---

## Contract Schema

A scenario TOML has the following top-level keys:

| Key | Required | Description |
|-----|----------|-------------|
| `name` | Yes | Human-readable scenario identifier |
| `type` | Yes | One of `analyze`, `compare` |
| `fixture` | For `analyze` | Filename in `harness/fixtures/` (inline data) |
| `input` | For `analyze` | Absolute or relative path to a CSV |
| `detector` | For `analyze` | `robust-zscore`, `dbscan`, `isolation-forest` |
| `features` | Yes | List of feature column names |
| `seed` | No | Integer seed for stochastic detectors |
| `baseline` | For `compare` | Path to baseline CSV |
| `candidate` | For `compare` | Path to candidate CSV |
| `baseline_metadata` | For `compare` | Table of `model`, `version`, `quantization`, `concurrency` |
| `candidate_metadata` | For `compare` | Same keys as baseline_metadata |
| `metrics` | For `compare` | List of metric specs with `name`, `direction`, `threshold` |
| `expected_evidence_class` | Recommended | Evidence class the result must have |
| `expected_confidence_min` | No | Lower bound on confidence (0.0–1.0) |
| `expected_confidence_max` | No | Upper bound on confidence |
| `expected_detector` | No | Detector name the result must carry |
| `anomaly_count_min` | No | Minimum acceptable anomaly count |
| `anomaly_count_max` | No | Maximum acceptable anomaly count |
| `required_evidence_fields` | No | List of dot-path fields that must be present |
| `required_assumptions` | No | Substrings that must appear in the `assumptions` list |
| `forbidden_assumptions` | No | Substrings that must NOT appear in `assumptions` |
| `supporting_metrics` | No | Numeric tolerances for `supporting_metrics` entries |
| `tolerance` | No | Relative tolerance for numeric comparisons (default 1e-5) |

---

## Evidence Class Values

Evidence classes are defined in `anomaly_lab.evidence.EvidenceClass`:

| Class | Meaning |
|-------|---------|
| `POINT_ANOMALY` | Single isolated statistical anomaly |
| `LOCAL_OUTLIER` | Point(s) different from immediate neighbours |
| `GLOBAL_OUTLIER` | Majority of points are anomalous (distribution shift) |
| `DENSITY_NOISE` | DBSCAN noise label — low local density |
| `REGIME_SHIFT` | Structural regime change |
| `DISTRIBUTION_SHIFT` | Statistical distribution difference |
| `VARIANCE_COLLAPSE` | Zero-variance or degenerate cluster |
| `COMPARISON_DIFFERENCE` | Statistical difference detected between two runs |
| `COMPARISON_EQUIVALENT` | TOST equivalence within margin |
| `INCONCLUSIVE` | Insufficient evidence to draw a conclusion |
| `NOT_APPLICABLE` | No anomaly/noise/no difference found |

---

## Semantic Supersets

Some evidence classes form a hierarchy. For example, `LOCAL_OUTLIER` is a subset of `GLOBAL_OUTLIER`. The evaluator allows a result to satisfy a contract by returning any class in the same superset family. This prevents false negatives from minor algorithmic differences in boundary cases.

The exact superset mapping is defined in `harness/evaluator.py` `_EVIDENCE_CLASS_HIERARCHY`.

---

## Example Contract

```toml
# harness/scenarios/01_rz_point_anomaly.toml
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

required_assumptions = [
    "median_and_MAD",
]

forbidden_assumptions = [
    "causal",
    "root_cause",
    "regression",
]
```

---

## Evaluation Protocol

1. **Run** — Execute the CLI command defined by the scenario TOML.
2. **Capture** — Write the result JSON to `harness/results/<name>.json`.
3. **Evaluate** — Run `harness/evaluator.evaluate_scenario(spec, result)` to check semantic correctness.
4. **Report** — Write `harness/results/<name>_eval.json` with per-check outcomes.

A scenario **passes** only when all semantic checks pass. Exit-code checks are a separate, mechanical concern.
