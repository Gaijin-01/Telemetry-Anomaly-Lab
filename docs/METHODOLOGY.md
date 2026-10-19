# Methodology

This document describes the statistical principles underlying Telemetry Anomaly Lab.

## Statistical Anomaly Detection Principles

Telemetry Anomaly Lab applies statistical anomaly detection methods to numeric telemetry
data. These methods share a common structure:

1. **Score computation**: Each detector computes an anomaly score for each observation.
   Higher scores indicate greater statistical unusualness.
2. **Thresholding**: Scores above a threshold are flagged as anomalous.
3. **Evidence classification**: The flagged pattern is mapped to an `EvidenceClass`.

The detectors are:

- **Robust Z-score**: Uses median and MAD (Median Absolute Deviation) for
  outlier-resistant scoring. Score = |x - median| / (1.4826 × MAD).
- **Isolation Forest**: Tree-ensemble anomaly scorer. Anomalies are easier to
  isolate and receive higher scores.
- **DBSCAN**: Density-based clustering. Points labelled -1 (noise) are density outliers.

## Why Detector Consensus Is Not Ground Truth

A common misconception is that when multiple detectors agree, the result must be
correct. This is not the case:

- Detectors can systematically misfire on certain data distributions.
- All detectors may be inappropriate for the data at hand.
- Agreement raises confidence but does not establish ground truth.
- Each detector's assumptions must be checked independently.

The `EvidenceRecord.confidence` field caps at 0.95 to reflect that even unanimous
detector agreement does not constitute certainty.

## Effect Size vs. Statistical Significance

A statistically significant result (p < 0.05) does not imply a large or operationally
meaningful effect. Telemetry Anomaly Lab separates these concepts:

- **Significance**: The `compare` command reports whether a difference is statistically
  distinguishable from zero.
- **Effect size**: The `delta` and `relative_change` fields report magnitude.
- **Equivalence**: TOST (Two One-Sided Tests) is required to claim equivalence,
  not merely non-significance.

The `COMPARISON_EQUIVALENT` evidence class is only emitted when an explicit equivalence
margin is provided and TOST testing passes.

## TOST Equivalence Testing

TOST (Two One-Sided Tests) is a parametric equivalence test:

1. Define an equivalence margin Δ (e.g., 5% relative change).
2. Test: is μ_below - μ_above > -Δ AND μ_above - μ_below < Δ?
3. If both one-sided tests pass, conclude equivalence within Δ.

Non-significant p-value (p > 0.05) is **not** equivalence. It merely means
"insufficient evidence to detect a difference" — the true difference could still
be large.

## Bootstrap Confidence Intervals

For small or non-normal samples, bootstrap resampling provides CI estimates:

1. Resample with replacement from the data `B` times (typically B=10000).
2. Compute the statistic of interest for each resample.
3. Use the α/2 and 1-α/2 percentiles as the confidence interval bounds.

Bootstrap CIs do not require normality but do require:
- Independent observations (exchangeability)
- A sufficiently large resample count
- Representative resampling (no systematic bias in sampling procedure)

## VUS-PR for Time Series

VUS-PR (Volume Under the Surface for Precision-Recall) is a rank-based evaluation
metric for time-series anomaly detection. It measures the area under the
precision-recall surface across all possible score thresholds.

**Status**: VUS-PR is noted as a potential future enhancement. It requires
ground-truth labelled anomaly windows and is not currently implemented.

## Change-Point Detection

The `change-points` command uses a retrospective two-sided median shift scan:

1. Enumerate all splits with at least `min_segment` observations on each side.
2. For each split, compute |median(right) - median(left)|.
3. Rank by shift magnitude (primary), support/size balance (secondary), position (tertiary).

A change point candidate is statistical evidence of a distributional shift,
**not** a root cause, causal event, or explanation.

## Regime Detection

DBSCAN clustering on standardized features identifies density-connected regions.
Regime detection treats each DBSCAN cluster as a distinct operating regime.

- Multiple regimes imply distributional non-stationarity.
- Regime detection does not identify *which* feature or process caused the regime.
- Regime transitions are **not** causal events.
