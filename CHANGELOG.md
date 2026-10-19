# Changelog

## v0.2.0 — Comparison, drift, change-points, regimes

- Compare compatible repeated telemetry runs using explicit metadata, metric direction and regression thresholds. `model`, `version`, `quantization` and `concurrency` must match; different versions are `NOT_COMPARABLE`.
- Add strictly past-only rolling and EWMA baselines, retrospective change-point candidates and scaled DBSCAN regime clusters with opaque IDs.
- Expose `compare`, `baseline`, `change-points`, and `regimes` CLI workflows alongside `analyze`.
- Add five user guides and Python 3.11–3.13 CI matrix.
- All reported differences are observational. `Causality: NOT ESTABLISHED`. Real RTX 6000 data is not included.

## v0.1.0 — Clean-room anomaly foundation

- Validate explicitly selected finite numeric telemetry features.
- Provide Isolation Forest, scaled DBSCAN and robust MAD/Z-score detectors, with structured results and strict JSON.
- Add the `anomaly-lab analyze` CLI, synthetic sample and provenance acknowledgement.
- Include typed validation errors, datetime JSON identity handling, NumPy integer random-state handling and finite-float robust MAD safety.
