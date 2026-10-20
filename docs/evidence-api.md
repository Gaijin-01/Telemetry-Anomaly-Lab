# Evidence API

The Evidence API translates raw anomaly-detection and comparison results into structured `EvidenceRecord` objects that carry an **evidence class** label and explicit assumptions/limitations.

> **Important**: Evidence records are statistical observations, NOT diagnoses, causal claims, or regression verdicts.

---

## EvidenceRecord

```python
@dataclass(frozen=True, kw_only=True)
class EvidenceRecord:
    evidence_class: str  # EvidenceClass enum value
    confidence: float | None  # 0.0–1.0, or None
    detector: str | None  # e.g. "robust-zscore", "dbscan"
    score: float | None  # Top anomaly score
    threshold: float | None  # Detection threshold used
    supporting_metrics: dict[str, float]
    assumptions: tuple[str, ...]  # What was assumed
    limitations: tuple[str, ...]  # What this record cannot claim
    interpretation: str  # Human-readable meaning
```

### Converting to dict

```python
record.to_dict()  # Returns JSON-serializable dict
```

---

## EvidenceClass Enum

| Value | Detection scenario |
|-------|--------------------|
| `POINT_ANOMALY` | Single isolated exceedance |
| `LOCAL_OUTLIER` | Few points deviating from neighbours |
| `GLOBAL_OUTLIER` | Majority of points flagged |
| `DENSITY_NOISE` | DBSCAN noise label (-1) |
| `REGIME_SHIFT` | Structural regime change detected |
| `DISTRIBUTION_SHIFT` | Statistical distribution difference |
| `VARIANCE_COLLAPSE` | Degenerate variance / constant feature |
| `COMPARISON_DIFFERENCE` | Statistical difference between two runs |
| `COMPARISON_EQUIVALENT` | Equivalence within margin (TOST) |
| `INCONCLUSIVE` | Cannot draw a conclusion |
| `NOT_APPLICABLE` | No anomaly / noise / difference found |

---

## Using the Evidence API

### Classify a DetectionResult

```python
from anomaly_lab.evidence import classify_detection_evidence
from anomaly_lab.detectors import robust_zscore as rz

result = rz.fit_predict(df, features=["x"])
record = classify_detection_evidence(result)

print(record.evidence_class)  # e.g. "POINT_ANOMALY"
print(record.confidence)  # e.g. 0.85
print(record.interpretation)  # Human-readable explanation
print(record.assumptions)  # Tuple of assumptions
print(record.limitations)  # Tuple of what this cannot claim
```

### Classify a ComparisonResult

```python
from anomaly_lab.evidence import classify_comparison_evidence

comparison_result = compare_runs(baseline_df, candidate_df, ...)
record = classify_comparison_evidence(comparison_result)

print(record.evidence_class)  # e.g. "COMPARISON_DIFFERENCE"
```

---

## Assumptions and Limitations

Every evidence record includes explicit `assumptions` and `limitations` tuples. These are the **contract** between the detector and the consumer: what must be true for the evidence class to be meaningful, and what claims the evidence cannot support.

### Common assumptions (robust-zscore)

- `median_and_MAD_are_appropriate_central_and_scale_estimators`
- `data_follows_a_symmetric_heavy_tailed_distribution`

### Common limitations (all detectors)

- `anomaly_label_is_not_regression_or_root_cause`
- `causality_not_established`
- `statistical_difference_is_not_a_causal_claim`

Consumers MUST check that their use case satisfies the assumptions before acting on the evidence.

---

## Confidence Scores

Confidence is a float in [0.0, 1.0]:

| Range | Meaning |
|-------|---------|
| 0.0 | Inconclusive — no conclusion possible |
| 0.5–0.7 | Low confidence — weak evidence |
| 0.7–0.9 | Medium confidence — plausible |
| 0.9–1.0 | High confidence — strong statistical signal |

Confidence does NOT encode the probability that an anomaly is "real" in a domain sense — it only reflects the strength of the statistical signal.

---

## Integration with the Scenario Harness

The harness evaluator (`harness/evaluator.py`) uses the evidence API to check:

1. `evidence_class` matches `expected_evidence_class` (with superset relaxation)
2. `confidence` is within `[expected_confidence_min, expected_confidence_max]`
3. `required_assumptions` appear in the `assumptions` tuple
4. `forbidden_assumptions` do NOT appear in `assumptions`
5. `anomaly_count` is within bounds

The evaluator does NOT check exit codes or mechanical CLI success — those are handled separately.
