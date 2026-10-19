# LLM Telemetry Anomaly Lab

A small, offline Python library and CLI for exploratory anomaly analysis of explicitly selected numeric telemetry. It can analyze LLM/GPU benchmarks (latency, throughput, utilization, VRAM) or arbitrary numeric measurements.

It reports *observed statistical anomalies*, not hardware failures, regressions, bad configurations, or causal diagnoses.

## 60-second example

```bash
python3 -m venv .venv
. .venv/bin/activate
python -m pip install .
anomaly-lab analyze examples/synthetic_telemetry.csv \
  --detector robust-zscore --features p95_s,tokens_per_second,gpu_util_pct,vram_mib \
  --output-json result.json
```

The selected feature names are explicit. Inspect the result as a lead for investigation, not a diagnosis.

## Detectors
- `isolation-forest`: deterministic sklearn Isolation Forest; scores are negated `score_samples`, so larger means more anomalous.
- `dbscan`: StandardScaler followed by DBSCAN; noise label `-1` is an anomaly, with no continuous score.
- `robust-zscore`: median/MAD modified Z-score; row score is the largest absolute feature deviation; larger is more anomalous. Zero-MAD deviations use an ordinal capped score, as documented in detector semantics.

Scaling matters: distance-based DBSCAN is sensitive to units. The detector standardizes selected features, but scaling does not make an inappropriate feature set appropriate. Select features whose units/relationships support the question you are asking. Constant selected columns are removed and reported; an all-constant selection is an error. Missing and non-finite feature values are rejected, not imputed.

## Reproducibility and limitations
Isolation Forest has a fixed local random seed by default. Same input, configuration, and dependency versions produce repeatable results. Unsupervised results depend on population composition, feature choice, scaling, and parameters; results are not ground truth or causal explanations. Robust marginal deviations can miss multivariate relationships; DBSCAN is sensitive to density/eps; Isolation Forest scores are model-relative.

## Acknowledgements / Inspiration

This project was inspired by and informed by the public repository [`IndraniB2020/All_machine_learning`](https://github.com/IndraniB2020/All_machine_learning), particularly its anomaly-detection examples. Telemetry Anomaly Lab is an independent clean-room implementation. No upstream source code or data is redistributed here. The upstream repository had no explicit software license when reviewed, so it served only as reference and inspiration. Credit for the original reference goes to IndraniB2020.

## Documentation
See [`docs/architecture.md`](docs/architecture.md), [`docs/detector_semantics.md`](docs/detector_semantics.md), [`examples/basic_analysis.md`](examples/basic_analysis.md), and [`PROVENANCE.md`](PROVENANCE.md).
