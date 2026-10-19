# Statistical Contract

This document defines what each `EvidenceClass` means, what conclusions ARE valid,
and what conclusions MUST NOT be emitted.

## EvidenceClass Reference

### POINT_ANOMALY

**Definition**: A single observation (or small set) is statistically unusual relative
to the rest of the dataset.

**Valid conclusions**:
- "Point X was flagged as a statistical outlier by detector Y."
- "The anomaly score for row X was Z, above threshold T."
- "This is point-level unusualness only."

**Forbidden conclusions**:
- "This is a regression caused by the recent code change." ❌
- "The hardware is failing." ❌
- "This caused the latency spike." ❌
- "We should roll back." ❌

---

### LOCAL_OUTLIER

**Definition**: A subset of points is unusual relative to their immediate neighbours,
suggesting local distributional change.

**Valid conclusions**:
- "A cluster of N points was flagged as a local outlier."
- "The local neighbourhood density differs from the global pattern."

**Forbidden conclusions**:
- "The system changed operating regime due to a specific cause." ❌
- "These points caused a cascade failure." ❌

---

### GLOBAL_OUTLIER

**Definition**: A large fraction (>50%) of points are simultaneously flagged,
indicating a global distributional shift or threshold miscalibration.

**Valid conclusions**:
- "The majority of points were flagged, suggesting global distribution change."
- "The detection threshold may be miscalibrated for this data."

**Forbidden conclusions**:
- "A regression occurred." ❌
- "The system is fundamentally broken." ❌

---

### DENSITY_NOISE

**Definition**: DBSCAN identified points as low-density noise (cluster label -1).

**Valid conclusions**:
- "N points were labelled as low-density noise by DBSCAN."
- "These points are in regions of low local density."

**Forbidden conclusions**:
- "These noise points are errors or failures." ❌
- "They caused system degradation." ❌

---

### REGIME_SHIFT

**Definition**: DBSCAN identified multiple distinct density clusters, suggesting
different operating regimes in the data.

**Valid conclusions**:
- "Multiple distinct operating regimes were detected."
- "The data contains N clusters with different density profiles."

**Forbidden conclusions**:
- "The system transitioned to a new regime because of X." ❌
- "This caused the performance change." ❌
- "Regime 2 is better than Regime 1." ❌

---

### DISTRIBUTION_SHIFT

**Definition**: Statistical evidence of a change in the underlying data distribution.

**Valid conclusions**:
- "Evidence suggests a distributional change between segment A and segment B."
- "The empirical CDF changed significantly."

**Forbidden conclusions**:
- "The distribution shifted because of the deployment." ❌
- "This caused the regression." ❌

---

### VARIANCE_COLLAPSE

**Definition**: Statistical evidence of reduced variance (e.g., variance halved).

**Valid conclusions**:
- "Variance in segment B is lower than in segment A."
- "The variance reduction is statistically significant."

**Forbidden conclusions**:
- "Variance collapse caused the regression." ❌
- "The system is now more stable." ❌

---

### COMPARISON_DIFFERENCE

**Definition**: A statistical difference was detected between baseline and candidate
runs on one or more metrics.

**Valid conclusions**:
- "Metric X showed a relative change of Y% between baseline and candidate."
- "The difference is statistically detectable."

**Forbidden conclusions**:
- "The candidate caused the difference." ❌
- "This is a confirmed regression requiring action." ❌
- "The difference is due to the new algorithm." ❌
- "Causality is established." ❌

---

### COMPARISON_EQUIVALENT

**Definition**: TOST equivalence testing concluded that baseline and candidate
are equivalent within a declared margin.

**Valid conclusions**:
- "Within the declared equivalence margin of ±5%, no meaningful difference exists."
- "The two populations are statistically equivalent for metric X."

**Forbidden conclusions**:
- "The two systems are identical." ❌
- "No difference exists whatsoever." ❌
- "The systems are causally equivalent." ❌

---

### INCONCLUSIVE

**Definition**: Insufficient evidence to draw any statistical conclusion.

**Valid conclusions**:
- "Not enough data to reach a conclusion."
- "The analysis was inconclusive."

**Forbidden conclusions**:
- "Therefore, there is no regression." ❌
- "Therefore, the systems are equivalent." ❌

---

### NOT_APPLICABLE

**Definition**: The analysis was not applicable to this data (e.g., metadata mismatch).

**Valid conclusions**:
- "The baseline and candidate are not comparable due to metadata differences."

**Forbidden conclusions**:
- "Therefore, no anomaly exists." ❌

---

## Corpus Fixture Validity Table

The synthetic corpus (`tests/synthetic_corpus/`) establishes ground truth for
validation. The following table maps each fixture to valid and invalid conclusions:

| Fixture                    | Expected Evidence Class     | Forbidden Conclusion Types                    |
|----------------------------|---------------------------|----------------------------------------------|
| spike.csv                  | POINT_ANOMALY             | Regression, root cause, causal               |
| negative_spike.csv         | POINT_ANOMALY             | Regression, causal                          |
| mean_shift.csv             | REGIME_SHIFT / POINT_ANOMALY | Root cause, causal                      |
| drift.csv                  | POINT_ANOMALY or INCONCLUSIVE | Regression, causal                    |
| variance_increase.csv      | POINT_ANOMALY or LOCAL_OUTLIER | Regression, causal                |
| variance_collapse.csv      | VARIANCE_COLLAPSE or POINT_ANOMALY | Regression, causal            |
| local_density_anomaly.csv  | DENSITY_NOISE             | Regression, causal, equivalence             |
| correlated_multivariate.csv | POINT_ANOMALY           | Regression, causal                          |
| repeated_anomaly_cluster.csv | POINT_ANOMALY / LOCAL_OUTLIER | Regression, causal                 |
| constant_feature.csv       | FeatureSelectionError      | Any anomaly claim                           |
| paired_ab.csv              | COMPARISON_DIFFERENCE     | Equivalence, causal                         |
| equivalent_ab.csv          | NOT COMPARISON_EQUIVALENT | COMPARISON_EQUIVALENT (without TOST)       |
| known_regression.csv       | COMPARISON_DIFFERENCE     | Equivalence, causal                         |
| insufficient_short.csv     | INCONCLUSIVE              | Any strong claim                            |
