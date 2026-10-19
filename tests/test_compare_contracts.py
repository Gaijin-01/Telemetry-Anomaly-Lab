import math

import pandas as pd
import pytest

from anomaly_lab.compare import MetricRule, RunMetadata, compare_runs
from anomaly_lab.schema import AnomalyLabError


def metadata():
    return RunMetadata(model="m", version="1", quantization="q4", concurrency="1")


@pytest.mark.parametrize("bad", [1, None, ""])
def test_metadata_requires_nonempty_strings(bad):
    with pytest.raises(AnomalyLabError, match="metadata"):
        RunMetadata(model=bad, version="1", quantization="q4", concurrency="1")


@pytest.mark.parametrize(
    "direction,threshold",
    [("sideways", 1), ("higher", math.inf), ("lower", math.nan), ("lower", -1)],
)
def test_metric_rules_reject_invalid_direction_or_threshold(direction, threshold):
    with pytest.raises(AnomalyLabError, match="metric"):
        MetricRule("x", direction, threshold)


@pytest.mark.parametrize("values", [[], ["not numeric"], [float("inf")], [float("nan")]])
def test_comparison_rejects_empty_nonnumeric_and_nonfinite_features(values):
    frame = pd.DataFrame({"x": values})
    with pytest.raises(AnomalyLabError, match="feature"):
        compare_runs(frame, frame, metadata(), metadata(), ["x"], [MetricRule("x", "lower", 0.09)])


def test_threshold_is_relative_change_proportion():
    result = compare_runs(
        pd.DataFrame({"x": [100, 100]}),
        pd.DataFrame({"x": [110, 110]}),
        metadata(),
        metadata(),
        ["x"],
        [MetricRule("x", "lower", 0.09)],
    )
    assert result.metrics[0].relative_change == pytest.approx(0.1)
    assert result.metrics[0].status == "REGRESSION"
    assert result.metrics[0].threshold_kind == "relative_change_proportion"


def test_equal_to_threshold_is_no_regression():
    result = compare_runs(
        pd.DataFrame({"x": [100, 100]}),
        pd.DataFrame({"x": [110, 110]}),
        metadata(),
        metadata(),
        ["x"],
        [MetricRule("x", "lower", 0.11)],
    )
    assert result.metrics[0].status == "NO_REGRESSION"
