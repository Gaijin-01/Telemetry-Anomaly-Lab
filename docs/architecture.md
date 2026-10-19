# Architecture

```text
CSV / DataFrame
      ↓
explicit schema/feature validation
      ↓
finite numeric matrix; constants removed with metadata
      ↓
selected detector + detector-specific scaling/statistics
      ↓
DetectionResult (row identity, anomaly, score/cluster, config, metadata)
      ↓
CLI console and optional JSON serialization
```

`ingest.read_csv` is the local CSV boundary. `features.prepare_features` owns selection, dtype, finite-value, empty-data, and constant-column checks; it returns a copied matrix and original index. Each detector uses that common preparation and does not mutate the caller frame. The Isolation Forest and DBSCAN pipelines standardize active columns. The robust detector uses per-feature median/MAD. `DetectionResult` makes missing score semantics explicit and preserves duplicate row index labels alongside row position in JSON. The CLI catches domain input/configuration errors and emits concise errors with non-zero status; underlying unexpected programming errors are not swallowed.

The package is local/offline: no external service, network ingestion, persistence, dashboard, or causal diagnosis.
