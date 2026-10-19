# Five-minute tutorial: from CSV to a regression report

This walkthrough shows the five workflows on a common example. They are complementary analyses, not a causal pipeline: a finding in one stage does not automatically prove the next claim.

## 1. Inspect the example CSV

`examples/synthetic_telemetry.csv` contains 15 synthetic rows and these columns:

```text
run_id, concurrency, p95_s, tokens_per_second, gpu_util_pct, vram_mib
```

It has low-, medium-, and high-concurrency runs, plus planted unusual observations (`latency-spike`, `throughput-collapse`, and `gpu-isolated`). Since concurrency differs between groups, this fixture is useful for CLI demonstration but is **not** a controlled baseline/candidate regression dataset.

## 2. Find unusual individual runs (`analyze`)

```bash
anomaly-lab analyze examples/synthetic_telemetry.csv \
  --detector robust-zscore \
  --features p95_s,tokens_per_second \
  --output-json anomalies.json
```

This identifies observations that are unusual relative to the selected data and detector. Review them as investigation leads, not confirmed regressions or faulty runs.

## 3. Compare repeated compatible runs (`compare`)

For a meaningful comparison, create `baseline.csv` and `candidate.csv` from repeated runs with exactly matching model, version, quantization, and concurrency metadata. The comparator cannot directly compare a changed version or configuration. It can compare repeated runs under the same declared metadata; it does not estimate the effect of changing versions.

Example command (replace paths and metadata with genuinely matching runs):

```bash
anomaly-lab compare baseline.csv candidate.csv \
  --features p95_s,tokens_per_second \
  --baseline-metadata model=qwen --baseline-metadata version=v1 \
  --baseline-metadata quantization=q4 --baseline-metadata concurrency=4 \
  --candidate-metadata model=qwen --candidate-metadata version=v1 \
  --candidate-metadata quantization=q4 --candidate-metadata concurrency=4 \
  --metric p95_s:lower:0.10 \
  --metric tokens_per_second:higher:0.10 \
  --output-json comparison.json
```

The report is statistical, not a regression finding across versions. The threshold `0.10` means a relative 10% threshold. Summaries use arithmetic means; review sample counts, status fields, and raw data. Output states `Causality: NOT ESTABLISHED`.

## 4. Inspect temporal drift (`baseline`)

For a single chronological series, compute an EWMA based only on prior values:

```bash
anomaly-lab baseline baseline.csv \
  --features p95_s --method ewma --alpha 0.2 \
  --order-by timestamp --output-json baseline-trace.json
```

Use `--method rolling --window 20` for a rolling mean. If using `--order-by timestamp`, first sort the CSV in chronological order: the option validates existing row order, it does not sort for you. Initial warm-up estimates are null, and future rows do not affect earlier estimates. The residual is observed value minus the prior-only estimate.

## 5. Locate candidate shifts (`change-points`)

```bash
anomaly-lab change-points baseline.csv \
  --feature p95_s --min-segment 3 --threshold 0.10 \
  --order-by timestamp --output-json shifts.json
```

The offline scan reports candidate boundaries where left/right segment medians differ by at least the supplied threshold. It requires at least two observations on each side, so a short series can return no candidates. This is retrospective; a candidate change point does not identify root cause.

## 6. Explore clusters (`regimes`)

```bash
anomaly-lab regimes examples/synthetic_telemetry.csv \
  --features p95_s,tokens_per_second \
  --eps 0.7 --min-samples 2 \
  --metadata run_id --output-json regimes.json
```

DBSCAN runs on standardized selected numeric features. The optional metadata is returned separately, not used to form clusters. Cluster IDs are opaque; noise is `-1`. Do not infer named or ordered “performance modes” from label numbers alone.

## Reading the results responsibly

- A statistical anomaly is not necessarily a regression.
- A comparison is meaningful only if the runs satisfy the explicit comparability contract and have enough repeated samples.
- A temporal residual or change point can guide inspection but cannot establish a cause.
- Cluster labels summarize density structure under selected features and parameters; they are not semantic categories.
- Correlation, regression status, and temporal ordering do not establish causation.
