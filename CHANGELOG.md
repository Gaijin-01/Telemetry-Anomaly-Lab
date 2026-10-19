# Changelog

## v0.4.0 — Anomaly Evidence Engine

- **DatasetSemantics** (Phase A): six semantic types — `UNORDERED_TABULAR` / `ORDERED_SERIES` / `PAIRED_EXPERIMENT` / `INDEPENDENT_EXPERIMENT` / `GROUPED` / `STREAMING_ORDERED`. Temporal detectors silently reject unordered data. Add `anomaly-lab inspect` CLI.
- **Applicability Registry** (Phase B): preflight check returns `APPLICABLE` / `NOT_APPLICABLE` / `INSUFFICIENT_DATA` / `INVALID_INPUT` per detector before execution.
- **Evidence Taxonomy** (Phase C): ten evidence classes — `POINT_ANOMALY` / `LOCAL_OUTLIER` / `GLOBAL_OUTLIER` / `DENSITY_NOISE` / `REGIME_SHIFT` / `DISTRIBUTION_SHIFT` / `VARIANCE_COLLAPSE` / `COMPARISON_DIFFERENCE` / `COMPARISON_EQUIVALENT` / `INCONCLUSIVE` / `NOT_APPLICABLE`.
- **Statistical Engine** (Phase D): paired bootstrap CI, Mann-Whitney U, Wilcoxon signed-rank, BCa bootstrap intervals, effect sizes (Cohen's d, Cliff's delta).
- **Equivalence Engine** (Phase H): TOST (Two One-Sided Tests) for explicit ±margin equivalence testing; `EQUIVALENT_WITHIN_MARGIN` outcome.
- **Consensus** (Phase I): rank-based detector consensus with pairwise agreement, Spearman rank correlation, and detector family diversity.
- **Regimes** (Phase F): PELT change-point detection via `ruptures` (optional dependency); rejects unordered data, requires ≥10 rows.
- **Evaluate** (Phase G): PR-AUC / ROC-AUC with ground-truth temporal labels; VUS-PR for range-annotated time series.
- **Synthetic Corpus** (Phase K): 18 deterministic fixtures with documented valid and prohibited conclusions.
- **Reproducibility** (Phase P): input SHA256, dependency versions, deterministic seed support, analysis UUID.
- **Output schema v0.4** (Phase N): `schema_version: "0.4"` on all outputs; full versioning of all result structures.
- **PyOD adapters** (Phase L): ECOD, LOF, KNN as optional detectors (optional dependency `pyod>=1.1`).
- **Conformal Calibration** (Phase M): design document only — rigorous implementation deferred.
- **Core rule preserved**: anomaly detector produces statistical evidence, not a diagnosis. `CLAIM_IS_NOT_EVIDENCE`, `REGRESSION_IS_NOT_CAUSATION`, `CHANGE_POINT_IS_NOT_ROOT_CAUSE` invariants maintained.
- Python 3.11 / 3.12 / 3.13 CI matrix; ruff + format + compileall gates.

## v0.3.0 — Controlled Comparison and Regression Gates

- **Controlled cross-version comparison** (Phase A): `--allow-difference` flag for explicitly declared change dimensions (`version`, `runtime_version`, `commit`, `driver`, `cuda`). Match vs. change dimensions recorded in output JSON. Backward-compatible with existing v0.2 commands.
- **CI/regression gate semantics** (Phase B): `--fail-on` exit codes: `0` = normal, `1` = regression found, `2` = configuration error. Add `--config` for TOML regression rules. Add `--direction higher|lower` per metric.
- **Modified Z-score** (Phase C): robust univariate evidence using median + 1.4826×MAD; explicit degenerate-input handling (MAD=0, constant series, NaN/inf); `method_family: robust_univariate` in metadata.
- **Streaming-ordered support** (Phase D): `STREAMING_ORDERED` dataset semantics; sequential processing with state; regime segmentation in streaming context.
- **Output schema**: `ResultStatus` enum (`NORMAL` / `REGRESSION` / `NOT_COMPARABLE` / `INSUFFICIENT_SAMPLE`), structured `anomalies` array, `run_info` block with SHA and dependency versions.
- All v0.2.0 invariants preserved: `METADATA_IS_NOT_FEATURE_DATA`, `REQUIRED_STATE_IS_EXPLICIT`, `WINDOWS_CANNOT_LEAK_FUTURE_DATA`, `SAME_INPUT_CONFIG_SEED_SAME_OUTPUT`, `CLUSTER_ID_HAS_NO_SEMANTIC_MEANING`.

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
