# Dataset Semantics

This document describes the semantic types of datasets that Telemetry Anomaly Lab
supports, when each analysis is applicable, and how to declare pairing keys.

## DatasetSemantics Types

Telemetry Anomaly Lab distinguishes between several semantic dataset types:

### 1. Univariate Series

A single numeric feature observed over time (or index order).

**Structure**: `CSV` with one feature column + optional ordering column.

**Valid analyses**:
- `robust-zscore` — flag individual outlier points
- `isolation-forest` — flag individual outlier points
- `dbscan` — flag density noise points (1D DBSCAN uses 1-NN distance)
- `change-points` — detect step-function median shifts

**Invalid analyses**:
- Multivariate regime detection (requires multiple features)
- Paired comparison (requires two runs)

---

### 2. Multivariate Feature Matrix

Multiple numeric features per observation.

**Structure**: `CSV` with multiple feature columns.

**Valid analyses**:
- `robust-zscore` — per-feature z-scores, row-level maximum
- `isolation-forest` — joint feature-space anomaly scoring
- `dbscan` — joint density-based clustering
- All multivariate detectors (ECOD, LOF, KNN when pyod installed)

**Invalid analyses**:
- Temporal change-point detection on individual features (requires `order-by`)
- Paired comparison without pairing keys

---

### 3. Temporal / Ordered Series

Observations ordered by time or another strictly increasing column.

**Declaration**: Use `--order-by COLUMN` CLI flag.

**Requirements**:
- The ordering column must be strictly increasing.
- No duplicates allowed in the ordering column.
- All values must be non-null.

**Valid analyses**:
- `change-points` — retrospective median shift detection
- `baseline` — EWMA / rolling baseline estimation
- Temporal regime detection

**Invalid analyses**:
- `dbscan` / `isolation-forest` as "temporal" — these are order-agnostic
- Treating unordered data as if it were ordered

---

### 4. Paired / Before-After Comparison

Two runs of the same system under different conditions, with explicit pairing.

**Declaration**: Use `--paired` and `--pairing-key COLUMN` in `compare` command.

**Requirements**:
- Each pair must share the same pairing key value.
- Each pair must have the same number of observations (balanced).
- Features must be numeric and finite.

**Valid analyses**:
- `compare` with `--paired` — paired difference test
- `compare` with `--equivalence-margin` — TOST equivalence test
- Bootstrap CI on paired differences

**Invalid analyses**:
- Treating unpaired data as paired (violates independence assumption)
- Claiming equivalence from non-significant unpaired test

---

### 5. Streaming / Sequential Data

Observations arriving in order, evaluated sequentially.

**Characteristics**:
- No backward-looking statistics (rolling windows allowed)
- Each point is evaluated as it arrives
- Retrospective analysis is not applicable

**Valid analyses**:
- EWMA baseline
- Sequential change-point detection
- Online anomaly detection

**Invalid analyses**:
- Batch DBSCAN on the full stream (uses future data)
- Offline global statistics

---

## Pairing Key Declaration

When comparing paired runs, declare the pairing key:

```bash
anomaly-lab compare baseline.csv candidate.csv \
  --features latency \
  --pairing-key run_id \
  --baseline-metadata model=qwen --baseline-metadata version=v1 \
  --candidate-metadata model=qwen --candidate-metadata version=v1 \
  --metric latency:lower:0.10
```

**Requirements for pairing keys**:
- Must be present in both baseline and candidate CSVs
- Must uniquely identify each pair (no missing or duplicate pair IDs)
- Must be non-null across all rows

## Streaming Data Requirements

For streaming data analysis:

1. **Order is required**: `--order-by timestamp` (or equivalent column)
2. **No future leakage**: detectors must only use past/current data
3. **Warm-up period**: initial rows before sufficient history returns `null`
4. **Memory-bounded**: operations should be O(1) memory per feature

## What Each Analysis Is NOT

| Analysis        | Is NOT a substitute for                    |
|----------------|--------------------------------------------|
| anomaly detect | regression verdict, root cause, diagnosis  |
| compare        | causal claim, explanation, fix recommendation |
| change-points  | root cause, causal event, explanation       |
| regimes        | equivalence claim, performance ranking      |
| baseline       | causal explanation, root cause             |
