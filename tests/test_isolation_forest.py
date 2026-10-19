import pandas as pd
import pytest

from anomaly_lab.detectors.isolation_forest import IsolationForestConfig, fit_predict
from anomaly_lab.schema import InvalidDetectorConfigError


def test_isolation_forest_is_deterministic_and_ranks_extreme_point_highest():
    frame = pd.DataFrame({"x": [0.0, 0.1, -0.1, 0.05, 0.2, -0.2, 30.0]}, index=list("abcdefg"))
    before = frame.copy(deep=True)
    first = fit_predict(frame, ["x"])
    second = fit_predict(frame, ["x"])

    assert first.is_anomaly == second.is_anomaly
    assert first.scores == second.scores
    assert first.row_indices == tuple("abcdefg")
    assert first.scores[-1] == max(first.scores)
    assert first.is_anomaly[-1]
    pd.testing.assert_frame_equal(frame, before)


def test_isolation_forest_validates_configuration():
    with pytest.raises(InvalidDetectorConfigError):
        fit_predict(pd.DataFrame({"x": [1, 2]}), ["x"], IsolationForestConfig(contamination=0.8))
