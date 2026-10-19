# Input data format

## CSV structure

Provide a CSV file with one header row and one observation per row. Keep the header names stable across runs. The parser uses pandas defaults; use ordinary comma-separated text with numeric feature values. Example columns include:

```csv
timestamp,run_id,model,concurrency,p50_s,p95_s,tokens_per_second,gpu_util_pct,vram_mib,error_rate
2026-10-19T10:00:00,r001,qwen,1,0.31,0.44,42.1,21,7200,0
```

The library reads CSV values into a table; select numeric feature columns explicitly for analyses. Do not assume every column in this illustration is a required schema field.

## Features versus metadata

- Features are the numeric measurements explicitly passed through `--features` or `--feature` (for example `p95_s,tokens_per_second`). Selected feature data must be numeric and finite. Missing/non-finite feature values are rejected, not imputed.
- Metadata describes run identity or context (for example `run_id`, `model`, `version`, `quantization`, `concurrency`). Metadata must not be smuggled into a numeric feature matrix merely because it is present in the CSV.
- `compare` accepts explicit `--baseline-metadata` and `--candidate-metadata` values. The current comparability contract requires exact matches for model, version, quantization, and concurrency. Mismatches produce `NOT_COMPARABLE`; this command does not infer a permissible changed version.
- For `regimes`, optional `--metadata COLUMN` values can be included in result rows but are excluded from DBSCAN features.

## Time and order

For `baseline` and `change-points`, CSV row order is used by default. You can supply `--order-by timestamp` to validate the existing row order against that column; it does **not** sort the CSV for you. The column must be non-null, unique, and strictly increasing as encountered. Sort the input file yourself before running a time-aware command. Avoid mixing runs from different configurations in one series unless intentional.

## Benchmark comparability

For a meaningful baseline/candidate comparison, collect repeated runs with the same model, version, quantization, and concurrency metadata. If evaluating a version or configuration change, the present comparator will intentionally reject it as non-comparable because version/configuration are part of the exact-match identity. Do not remove metadata or claim a causal regression just to force a comparison.

The included `examples/synthetic_telemetry.csv` is synthetic and illustrates different concurrency levels; it is not suitable as evidence of a controlled regression.
