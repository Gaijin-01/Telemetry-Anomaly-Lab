import json

import pandas as pd

from anomaly_lab.compare import MetricRule, RunMetadata, compare_runs


def meta(**overrides):
    values = dict(model="m", version="1", quantization="q4", concurrency="1")
    values.update(overrides)
    return RunMetadata(**values)


def test_comparable_runs_report_directional_threshold_status_and_noncausality():
    result = compare_runs(
        pd.DataFrame({"latency": [10, 12], "model": [99, 99]}),
        pd.DataFrame({"latency": [15, 17], "model": [0, 0]}),
        meta(),
        meta(),
        ["latency"],
        [MetricRule("latency", "lower", 0.2)],
    )
    assert result.status == "REGRESSION"
    metric = result.metrics[0]
    assert metric.baseline == 11 and metric.candidate == 16 and metric.delta == 5
    assert metric.status == "REGRESSION"
    assert "model" not in [m.name for m in result.metrics]
    assert result.causality == "NOT ESTABLISHED"


def test_metadata_mismatch_is_not_comparable():
    result = compare_runs(
        pd.DataFrame({"x": [1, 2]}),
        pd.DataFrame({"x": [9, 10]}),
        meta(),
        meta(version="2"),
        ["x"],
        [MetricRule("x", "higher", 1)],
    )
    assert result.status == "NOT_COMPARABLE"
    assert "version" in result.mismatches


def test_insufficient_sample_does_not_assert_regression():
    result = compare_runs(
        pd.DataFrame({"x": [1]}),
        pd.DataFrame({"x": [100]}),
        meta(),
        meta(),
        ["x"],
        [MetricRule("x", "lower", 1)],
        min_samples=2,
    )
    assert result.status == "INSUFFICIENT_SAMPLE"
    assert result.metrics[0].status == "INSUFFICIENT_SAMPLE"
    assert "small sample" in result.warnings[0].lower()


def test_compare_cli_accepts_explicit_metadata_and_metric_rule(tmp_path, capsys):
    from anomaly_lab.cli import main

    baseline, candidate, output = tmp_path / "b.csv", tmp_path / "c.csv", tmp_path / "out.json"
    baseline.write_text("latency,model\n10,7\n12,7\n")
    candidate.write_text("latency,model\n15,1\n17,1\n")
    args = [
        "compare",
        str(baseline),
        str(candidate),
        "--features",
        "latency",
        "--baseline-metadata",
        "model=m",
        "--baseline-metadata",
        "version=1",
        "--baseline-metadata",
        "quantization=q4",
        "--baseline-metadata",
        "concurrency=1",
        "--candidate-metadata",
        "model=m",
        "--candidate-metadata",
        "version=1",
        "--candidate-metadata",
        "quantization=q4",
        "--candidate-metadata",
        "concurrency=1",
        "--metric",
        "latency:lower:0.2",
        "--output-json",
        str(output),
    ]
    assert main(args) == 0
    assert "NOT ESTABLISHED" in capsys.readouterr().out
    payload = json.loads(output.read_text())
    assert payload["status"] == "REGRESSION"
    assert payload["causality"] == "NOT ESTABLISHED"
    assert [m["name"] for m in payload["metrics"]] == ["latency"]
