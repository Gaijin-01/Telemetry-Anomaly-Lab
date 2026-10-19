import numpy as np
import pandas as pd
import pytest

from anomaly_lab.past_baseline import compute_past_baseline
from anomaly_lab.schema import InvalidDetectorConfigError, NonFiniteValueError


def test_alpha_string_raises_domain_config_error():
    with pytest.raises(InvalidDetectorConfigError):
        compute_past_baseline(pd.DataFrame({"x": [1.0, 2.0]}), ["x"], method="ewma", alpha="0.5")


def test_ewma_overflow_is_rejected_not_emitted_as_finite_baseline():
    frame = pd.DataFrame({"x": [1.7e308, -1.7e308, 1.7e308]})
    with pytest.raises(NonFiniteValueError):
        compute_past_baseline(frame, ["x"], method="ewma", alpha=0.5)


def test_ewma_overflowing_residual_is_rejected():
    frame = pd.DataFrame({"x": [-1.7e308, 1.7e308]})
    with pytest.raises(NonFiniteValueError):
        compute_past_baseline(frame, ["x"], method="ewma", alpha=0.5)


def test_numpy_integer_rolling_window_is_accepted():
    result = compute_past_baseline(
        pd.DataFrame({"x": [1.0, 2.0, 3.0]}), ["x"], method="rolling", window=np.int64(2)
    )
    assert result.baseline["x"].iloc[2] == 1.5
