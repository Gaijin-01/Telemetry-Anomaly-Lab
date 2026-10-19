# Basic analysis example

The adjacent CSV is synthetic, intentionally small, and contains three normal load regimes (concurrency 1, 4, and 8) plus three hand-planted unusual observations:

- `latency-spike`: p95 latency is much larger than other concurrency-4 runs.
- `throughput-collapse`: tokens/second is much lower than other concurrency-4 runs.
- `gpu-isolated`: GPU utilization and VRAM are unusually high for its concurrency regime.

Run:

```bash
anomaly-lab analyze examples/synthetic_telemetry.csv \
  --detector robust-zscore --features p95_s,tokens_per_second,gpu_util_pct,vram_mib \
  --output-json result.json
```

The result is evidence of unusual measurements under the chosen features and detector only. The fixture is not a calibrated benchmark or ground truth for a real deployment; investigate the runs and their context before drawing conclusions.
