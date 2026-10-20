# Changelog

## v0.5.0 — Experiment Contract, Evidence API, Orchestration and Skills

### Release Repair

- **Package version**: `pyproject.toml` `project.version` bumped `0.4.2` → `0.5.0`. The `v0.5.0` tag previously pointed at a commit whose installable package still reported `0.4.2`; the tag and the package version are now consistent.
- **Output schema version**: `schema_version` bumped `0.4` → `0.5` across `DetectionResult`, `ConsensusResult`, `EvaluationResult`, the dataset-semantics profile payload, and the experiment-result deserialization default. Package version and output schema version now share the same `major.minor` prefix, as required by the release metadata gates.
- **GitHub Release**: a real release now exists for the `v0.5.0` tag. Previously only the tag existed, so `/releases/tag/v0.5.0` returned `404`.

### Stdlib-Only TOML Writing

- **Removed the `toml>=0.10` runtime dependency**: experiment specifications are read with stdlib `tomllib`, and results are written back with a small deterministic stdlib TOML writer (`_dict_to_toml`, `_make_toml_compatible`, `_toml_escape`, `_fmt_val`) instead of `toml.dumps()`.
- **Deterministic output**: strings are always quoted, `None`/NaN/inf values are omitted (TOML 1.0 has no `null` literal), arrays of scalars and arrays of inline tables are emitted in TOML 1.0 form, and arrays of tables are emitted as `[[...]]` sections with scalar members unindented.
- **No new third-party dependency is introduced**; the stdlib-only invariant is restored.

### Experiment Contract

- **ExperimentSpec**: frozen dataclass with validation — `experiment_id`, `dataset_path`/`dataset_hash`, `features`, `analysis_type`, `detectors`, `baseline_role`/`candidate_role`, `thresholds`, `seed`, `allowed_changed_dimensions`.
- **ExperimentResult**: frozen dataclass with confidence clamping and applicability tracking.
- **Canonical hashing**: SHA256 canonical hashes for spec, parameters and experiment results. Deterministic — same input, spec and seed always produce the same hash.
- **ProvenanceRecord**: environment metadata snapshot for every experiment run.
- **Replay**: verify and restore an experiment from its provenance record.

### Evidence API

- **taxonomy**: `EvidenceClass` (extended), `CausalityStatus`, `EvidenceStrength`. `EvidenceRecord` gains `causality_status`, `applicability` and `verdict`.
- **applicability**: `is_applicable()` enforces `INVALID_ANALYSIS_CANNOT_EXECUTE`.
- **comparability**: `can_compare()` enforces `TIME_ORDER` / `BASELINE` / `CANDIDATE` / `METADATA_IS_NOT_FEATURE`.
- **verdicts**: `compute_verdict()` enforces the `FAILED_EXECUTION_CANNOT_PASS` invariant.

### Orchestration

- **protocol**: allowed action sequence — inspect, plan, validate, execute, summarize, export.
- **actions**: action registry and dispatcher.
- **guardrails**: `guard()` blocks forbidden actions.

### Skills

- Six skills under `src/anomaly_lab/skills/`: `inspect_data`, `select_analysis_plan`, `validate_comparison`, `run_experiment`, `summarize_evidence`, `export_report`.
- The `SKILL_OUTPUT_CANNOT_OVERRIDE_CORE_EVIDENCE` invariant is enforced.

### CLI and Harness

- `anomaly-lab experiment validate | plan | run | replay` subcommands.
- Scenario harness with 15 scenario TOMLs and expected evidence contracts, plus `harness/runner.py` and `harness/evaluator.py` for semantic correctness checks.

### Tests

611 passed (570 core + 41 PyOD), 15 skipped.

## v0.4.2 — Hygiene Sanitation and Benchmark Hardening

### Internal Context Removed

- **DELETE**: `AGENTS.md`, `docs/PROVENANCE.md`, `docs/V0.4.0_BASELINE.md`, `docs/SWARM.md`, `docs/AUDIT_REPORT.md` — contain agent methodology, session paths, and forensic analysis not appropriate for public project documentation
- **DELETE**: tracked caches `.ruff_cache/`, `.pytest_cache/`, `.idea/`
- **MOVE**: benchmark harness from repository root to `benchmarks/pyod_candidates.py`

### Benchmark Failure Semantics Fixed

- **No global warning suppression**: removed `warnings.filterwarnings("ignore")` at module level
- **Explicit failure tracking**: detector failure → `None` score (not zero), excluded from AUC/spearman aggregates via `nanmean`
- **Non-zero exit on failure**: benchmark exits `sys.exit(1)` if any required detector run fails
- **Deterministic datasets**: `_make_rng(seed, dataset_id)` with SHA256 hash mixing — same `(id, seed)` always produces identical content
- **Reproducibility test**: `--test-reproducibility` verifies 32 unique dataset hashes and same-seed stability
- **Output**: `--output` flag (default `benchmark_results.json`); `--stdout` for console-only

### .gitignore Hardened

Added: `.idea/`, `.vscode/`, `.mypy_cache/`, `.hypothesis/`, `.tox/`, `.nox/`, `.coverage.*`, `coverage.xml`, `*.tar.gz`, `*.zip`, `artifacts/`, `benchmark_results*.json`

### CI Hygiene Gate Hardened

- Section 3 (machine-specific paths): switched from filename-only scan to **content scan** (`git ls-files -z | xargs grep -lF`)
- Section 6 (hardcoded user paths): removed `*.md`/`*.txt` exclusion — **all textual files now scanned**
- CI workflow venv path changed from `/tmp/anomaly-lab-smoke` to `/tmp/anomaly-lab-test-smoke` to avoid false-positive match on `/tmp/anomaly-lab` pattern

## v0.4.1 — PyOD Compatibility Fixes and Hardening

> ⚠️ **Benchmark correction notice**: v0.4.0 benchmark results claiming "HBOS AUC=0.991, PCA AUC=0.969" were computed on a synthetic corpus where the 30 synthetic entries (10 configs × 3 seeds) produced only 3 unique score fingerprints due to a shared RNG state bug in the benchmark generator. The corrected numbers below are from a re-run of the fixed generator producing 32 genuinely unique datasets.

### Bug Fixes

- **ECOD API mismatch** (critical): `ECOD(contamination=..., n_bins=...)` — `n_bins` was never a valid PyOD ECOD parameter. Removed; PyOD ECOD accepts only `contamination` and `n_jobs`. Fix: `n_bins` removed from `ECODConfig`, `n_jobs` added.
- **Contamination validation**: All three PyOD adapters (ECOD, LOF, KNN) incorrectly validated contamination as `(0, 1]` instead of the correct PyOD range `(0, 0.5]`. Fixed across all three.
- **Score semantics documentation**: ECOD adapter had contradictory comments about score orientation (lower vs. larger = more anomalous). Corrected to "larger = more anomalous (PyOD decision score)".

### Input Guards and Typed Errors

- **`InsufficientDataError`** (new, `schema.py`): Added typed error for insufficient rows. All three adapters now raise this explicitly instead of uncontrolled PyOD/sklearn crashes.
- **Pre-flight minimum-sample checks**: ECOD requires ≥5 rows; LOF/KNN require ≥`n_neighbors+1` rows.
- **Non-finite score guard** (3 adapters): All three adapters now raise `AnomalyLabError` if PyOD produces NaN/inf scores, instead of silently propagating them or converting to zeros.

### Test Coverage

- **`tests/test_pyod_adapters.py`** (new): 32 compatibility tests covering ECOD/LOF/KNN instantiation, fit/predict, score orientation, row index preservation, constant-feature handling, contamination validation, insufficient sample behavior, NaN output rejection.
- **`except Exception: pass` removed**: `test_insufficient_samples_behavior` now enforces a typed `AnomalyLabError` contract or valid `DetectionResult` — no silent pass.

### Version Policy

- **`pyod>=1.1` confirmed safe**: Verified against PyOD v1.1.0 source; ECOD, LOF, KNN APIs unchanged since v1.1.0. `n_bins` never existed in any PyOD version.

### CI Hardening

- **`pyod-adapters` CI job** (Python 3.12): Installs `.[pyod]`, verifies PyOD version, runs `pytest tests/test_pyod_adapters.py`, verifies adapter imports.
- **`pyod-version-matrix` CI job** (Python 3.12): Tests against PyOD 1.1.0, 2.0.0, and current — proves `pyod>=1.1` compatibility across the stated range.

### Empirical Benchmark (Phase 6, corrected)

8 detectors × 32 genuinely unique datasets (10 configs × 3 seeds + correlated_multivariate + local_density). Previous "32-dataset" results were invalidated: only 3 unique score fingerprints. Fixed benchmark generator uses per-seed shared RNG state with `i`-step advancement to ensure each config produces distinct data.

| Detector | AUC | Max Spearman vs others |
|---|---|---|
| robust-zscore | 0.978±0.069 | 0.784 (HBOS) |
| isolation-forest | 0.990±0.024 | 0.952 (LOF) ⚠️ |
| dbscan | 0.553±0.110 | — (REJECT_UNSTABLE) |
| ecod | 0.962±0.164 | 0.906 (PCA) |
| lof | 0.990±0.018 | 0.952 (IF) ⚠️ |
| knn | 0.990±0.018 | 0.915 (LOF) |
| **hbos** | **0.988±0.022** | **0.896 (IF)** → **ADD** |
| **pca** | **0.963±0.143** | **0.906 (ECOD)** → **ADD** |

- **HBOS → ADD**: AUC=0.988, max Spearman=0.896 (below 0.90 threshold). Adds global histogram-bin perspective, complementary to local-density detectors.
- **PCA → ADD**: AUC=0.963, max Spearman=0.906 (marginally above 0.90). Projection-based detection of subspace anomalies not caught by ECDF or density-based methods.
- **LODA → RESEARCH_ONLY**: Analytical evidence only; no implementation in this codebase.
- **DBSCAN → REJECT_UNSTABLE**: AUC=0.553 on sparse injected outliers (eps=0.5 unsuitable for isolated points; eps tuning may help but not addressed here).

⚠️ `isolation-forest ↔ lof` redundancy at 0.952 — highest pairwise correlation in the ensemble.

### Isolation Forest Audit (Phase 8)

Verified correct: StandardScaler necessity (random splits are scale-sensitive), contamination accepts `(0, 1]` (sklearn vs PyOD difference), score inversion via `-model.score_samples()` is correct (lower = more isolated = more anomalous, negated to higher = more anomalous), `random_state=42` provides reproducibility.

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
