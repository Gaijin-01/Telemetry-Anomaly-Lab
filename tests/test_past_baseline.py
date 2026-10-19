import numpy as np
import pandas as pd
import pytest

from anomaly_lab.past_baseline import compute_past_baseline
from anomaly_lab.schema import (
    EmptyDataError,
    FeatureSelectionError,
    InvalidDetectorConfigError,
    NonFiniteValueError,
)


def test_ewma_baseline_uses_strictly_prior_observations_and_current_residual():
    frame = pd.DataFrame({"latency": [2.0, 10.0, 4.0]})
    result = compute_past_baseline(frame, ["latency"], method="ewma", alpha=0.5)
    assert np.isnan(result.baseline["latency"].iloc[0])
    assert result.baseline["latency"].iloc[1:].tolist() == [2.0, 6.0]
    assert np.isnan(result.residual["latency"].iloc[0])
    assert result.residual["latency"].iloc[1:].tolist() == [8.0, -2.0]


def test_mutating_future_rows_cannot_change_earlier_baselines_or_residuals():
    frame = pd.DataFrame({"x": [1.0, 3.0, 5.0, 7.0]})
    changed = frame.copy()
    changed.loc[2:, "x"] = [1000.0, -1000.0]
    for method, config in [("ewma", {"alpha": 0.3}), ("rolling", {"window": 2})]:
        first = compute_past_baseline(frame, ["x"], method=method, **config)
        second = compute_past_baseline(changed, ["x"], method=method, **config)
        pd.testing.assert_frame_equal(first.baseline.iloc[:2], second.baseline.iloc[:2])
        pd.testing.assert_frame_equal(first.residual.iloc[:2], second.residual.iloc[:2])


def test_rolling_uses_only_previous_window_and_marks_warmup_missing():
    result = compute_past_baseline(
        pd.DataFrame({"x": [1.0, 3.0, 9.0, 11.0]}), ["x"], method="rolling", window=2
    )
    assert result.baseline["x"].iloc[:2].isna().all()
    assert result.baseline["x"].iloc[2:].tolist() == [2.0, 6.0]


def test_timestamp_order_is_explicit_and_must_be_strictly_increasing():
    frame = pd.DataFrame({"t": pd.to_datetime(["2026-01-02", "2026-01-01"]), "x": [2.0, 1.0]})
    with pytest.raises(FeatureSelectionError, match="strictly increasing"):
        compute_past_baseline(frame, ["x"], method="ewma", alpha=0.5, order_by="t")
    result = compute_past_baseline(frame, ["x"], method="ewma", alpha=0.5)
    assert result.baseline["x"].iloc[1] == 2.0  # caller order is preserved


@pytest.mark.parametrize(
    "kwargs",
    [
        {"method": "rolling"},
        {"method": "rolling", "window": 0},
        {"method": "ewma"},
        {"method": "ewma", "alpha": 0},
        {"method": "ewma", "alpha": 1.1},
    ],
)
def test_invalid_configuration_is_rejected(kwargs):
    with pytest.raises(InvalidDetectorConfigError):
        compute_past_baseline(pd.DataFrame({"x": [1.0]}), ["x"], **kwargs)


def test_empty_and_nonfinite_inputs_are_rejected():
    with pytest.raises(EmptyDataError):
        compute_past_baseline(pd.DataFrame({"x": []}), ["x"], method="ewma", alpha=0.5)
    with pytest.raises(NonFiniteValueError, match="non-finite"):
        compute_past_baseline(pd.DataFrame({"x": [np.inf]}), ["x"], method="ewma", alpha=0.5)
