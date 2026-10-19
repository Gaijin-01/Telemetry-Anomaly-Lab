# CLI reference

Run `anomaly-lab --help` or append `--help` to a subcommand for the installed version's exact options. Each analysis command supports `--output-json PATH`.

## `analyze`

```text
anomaly-lab analyze CSV --detector isolation-forest|dbscan|robust-zscore --features COL[,COL...]
```

Finds statistical anomalies using explicitly selected numeric features. Detectors have different score semantics; see [detector semantics](detector_semantics.md).

## `compare`

```text
anomaly-lab compare BASELINE.csv CANDIDATE.csv \
  --features COL[,COL...] \
  --baseline-metadata KEY=VALUE (repeatable) \
  --candidate-metadata KEY=VALUE (repeatable) \
  --metric COL:higher|lower:THRESHOLD (repeatable)
```

Required metadata keys are `model`, `version`, `quantization`, and `concurrency`; all must match exactly. Each selected feature must have exactly one metric rule. Threshold is a non-negative relative proportion (`0.10` means 10%); comparisons use arithmetic means. `--min-samples N` defaults to 2. Incompatible metadata returns `NOT_COMPARABLE`; low samples are non-assertive. Output states `Causality: NOT ESTABLISHED`.

## `baseline`

```text
anomaly-lab baseline CSV --features COL[,COL...] --method rolling|ewma
```

For `rolling`, pass `--window N` (positive integer). For `ewma`, pass `--alpha A` in `(0, 1]`. Optional `--order-by COLUMN` validates that the existing row order is non-null, unique, and strictly increasing by that column; it does not sort the input. Each estimate uses earlier observations only; initial warm-up is null.

## `change-points`

```text
anomaly-lab change-points CSV --feature COL --min-segment N --threshold VALUE [--order-by COLUMN]
```

Performs a retrospective median-shift candidate scan. `--min-segment` is the minimum size on each side; `--threshold` is the explicit minimum absolute median difference.

## `regimes`

```text
anomaly-lab regimes CSV --features COL[,COL...] [--eps VALUE] [--min-samples N] [--metadata COLUMN ...]
```

Clusters standardized selected numeric features with DBSCAN. `--metadata` may repeat, but those columns are excluded from clustering. Cluster IDs are opaque; noise label is `-1`.

## Common data/output behavior

- CSV must have a header; selected numeric features must be finite and non-missing. No feature imputation is performed.
- Feature names are explicit. Metadata is separate from feature data.
- `--output-json PATH` writes strict JSON; non-finite numbers are rejected. Intentional warm-up values are serialized as `null`.
- Handled input/analysis errors may exit with code 2 and write an explanation to stderr. `--help` exits successfully; argument-parsing failures use argparse's error handling.
