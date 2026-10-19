import pandas as pd
import pytest

from anomaly_lab.compare import MetricRule, RunMetadata, compare_runs
from anomaly_lab.schema import AnomalyLabError


def metadata():
    return RunMetadata(model="m", version="1", quantization="q4", concurrency="1")


def test_insufficient_sample_reports_computed_delta_without_regression():
    result = compare_runs(
        pd.DataFrame({"x": [10.0]}),
        pd.DataFrame({"x": [12.0]}),
        metadata(),
        metadata(),
        ["x"],
        [MetricRule("x", "lower", 0.1)],
        min_samples=2,
    )
    metric = result.metrics[0]
    assert result.status == "INSUFFICIENT_SAMPLE"
    assert metric.status == "INSUFFICIENT_SAMPLE"
    assert metric.delta == 2.0
    assert metric.relative_change == pytest.approx(0.2)


@pytest.mark.parametrize(
    "values",
    [([-1e308, -1e308], [1e308, 1e308])],
)
def test_nonfinite_computed_summary_is_rejected(values):
    baseline_values, candidate_values = values
    with pytest.raises(AnomalyLabError, match="finite"):
        compare_runs(
            pd.DataFrame({"x": baseline_values}),
            pd.DataFrame({"x": candidate_values}),
            metadata(),
            metadata(),
            ["x"],
            [MetricRule("x", "lower", 0.1)],
        )


def test_large_finite_means_remain_finite_and_json_serializable():
    import json

    result = compare_runs(
        pd.DataFrame({"x": [1e308, 1e308]}),
        pd.DataFrame({"x": [1e308, 1e308]}),
        metadata(),
        metadata(),
        ["x"],
        [MetricRule("x", "lower", 0.1)],
    )
    assert result.metrics[0].baseline == 1e308
    json.dumps(result.to_dict(), allow_nan=False)
