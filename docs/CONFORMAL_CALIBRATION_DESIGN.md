# Conformal Calibration (EXPERIMENTAL / DEFERRED)

## Goal

Convert arbitrary anomaly scores into empirically calibrated tail probability estimates
when the exchangeability assumption holds.

## Assumption: Exchangeability

Conformal prediction requires that all observations are exchangeable (i.e., i.i.d.
or at least permutation invariant). This assumption **fails** when:

- Distribution shift between calibration and test set
- Time-series autocorrelation
- Dependent observations (e.g., overlapping windows)
- Changing data collection conditions

## Algorithm

Given a calibration set `C` of size `n` and a test point `x`:

1. **Split data** into proper calibration set and test set
2. **Compute non-conformity scores** on calibration set:
   `s_i = non_conformity_score(x_i)` for each calibration point
3. **Compute non-conformity score** for test point: `s_test`
4. **Compute empirical p-value** (fraction of calibration scores ≥ test score):
   ```
   p(x_test) = |{ i in C : s_i >= s_test }| / (n + 1)
   ```
5. **Apply Bonferroni correction** for multiple testing:
   `p_corrected = p_raw * m` where `m` = number of tests

## Output Schema

```json
{
  "p_value": 0.023,
  "calibrated_under_exchangeability_assumption": true,
  "assumption_warnings": [
    "exchangeability_assumption_may_be_violated_due_to_temporal_structure",
    "distribution_shift_detected_between_calibration_and_test"
  ],
  "caveats": [
    "calibration_set_must_be_larger_than_20_for_defensible_results",
    "finite_sample_guarantees_only",
    "p_value_is_conditional_on_calibration_set"
  ]
}
```

## Implementation Notes

- Use the **split conformal** variant (not full conformal) for computational efficiency.
- The non-conformity score can be any detection score (z-score, isolation forest score, etc.).
- For anomaly detection, `s_i = -score_i` (lower score = more anomalous → higher s = more extreme).

## Deferral Notice

If a scipy- or numpy-based rigorous implementation cannot be verified against a trusted
reference fixture, write this design document and **defer the code** until:

1. A reference implementation can be found for validation
2. A synthetic calibration fixture with known p-values can be constructed

## Verification Requirements

Before shipping any implementation, it MUST:

- Reproduce against a known reference fixture (e.g., conformal prediction on Gaussian data
  where exact p-values are known analytically)
- Pass `pytest tests/test_conformal_calibration.py`
- Document the reference fixture used for verification in the test file

## Relationship to Anomaly Lab

This calibration layer sits **above** the detector output:

```
Raw Score → Non-Conformity Score → Empirical P-Value → Calibrated Evidence
```

The resulting p-value supplements but does not replace the `EvidenceRecord` class.
It adds a calibrated probability dimension to existing evidence classes.

## Limitations

- Exchangeability is almost never satisfied for time-series telemetry data.
- Temporal autocorrelation is the common failure mode for telemetry.
- Use only when data has been checked for exchangeability violations.
- Even when valid, provides only finite-sample coverage guarantees.
- Not a substitute for domain knowledge about when anomalies are operationally significant.
