# Interpretation guide

This project reports statistical structure in explicitly selected telemetry. It does not label runs as “good” or “bad,” determine hardware health, identify root causes, or establish causation.

## The five commands answer different questions

- **`analyze` — which individual rows look unusual?** Detector results depend on selected features, population, preprocessing, and parameters. An outlier is an investigation lead.
- **`compare` — how do two explicitly comparable samples differ?** Metadata must match exactly for model, version, quantization, and concurrency. The current comparator cannot directly compare a changed version/configuration because such runs are `NOT_COMPARABLE`. Threshold status is not proof of a regression.
- **`baseline` — how does each observation compare with prior observations?** Rolling/EWMA estimates use past rows only. Warm-up nulls are expected. Residuals do not identify a cause.
- **`change-points` — where might the series have shifted?** The scan is retrospective and returns candidate boundaries under a median-shift threshold. It does not explain the boundary.
- **`regimes` — what density groups appear under these features and settings?** DBSCAN cluster numbers are opaque, arbitrary labels. Noise `-1` is not a diagnosis; clusters are not inherently named performance modes.

## Avoid invalid comparisons

Control benchmark conditions and collect repeated runs. `compare` requires matching model, version, quantization, and concurrency metadata. If any differ, fix the experimental design or report the mismatch; do not omit metadata merely to force a result. The three sample rows in the README vary concurrency, so they illustrate CSV shape only and cannot isolate the effect of another change.

## Metrics and thresholds

Set metric direction deliberately: lower latency is generally desirable, while higher throughput may be desirable. A threshold of `0.10` means a relative 10% threshold. Current summaries use arithmetic means; outliers can influence means. Check sample counts, warnings, per-metric status, raw measurements, and uncertainty before acting.

## Temporal claims

- A past-only baseline prevents future-data leakage into prior estimates, but does not eliminate confounding or measurement issues.
- A retrospective change point is a candidate time boundary, not an intervention analysis.
- A sequence of anomalies, a comparison status, and a detected boundary together still do not prove causality.

## Reproducibility

Record the exact CSV, feature list, metadata, detector parameters, package version, and environment. Synthetic examples are for understanding command behavior, not for claims about real hardware. Phase G's real RTX 6000 data is intentionally omitted and remains optional external validation.
