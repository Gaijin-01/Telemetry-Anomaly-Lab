import numpy as np
import pandas as pd
import pytest

from anomaly_lab.detectors.robust_zscore import RobustZScoreConfig, fit_predict
from anomaly_lab.schema import InvalidDetectorConfigError


def test_robust_zscore_detects_obvious_spike_and_threshold_is_strict():
    frame = pd.DataFrame({"x": [10, 10, 10, 10, 11, 12, 100]})
    result = fit_predict(frame, ["x"], RobustZScoreConfig(threshold=3.5))

    assert result.is_anomaly[-1]
    assert result.scores[-1] == max(result.scores)
    assert not result.is_anomaly[0]


def test_robust_zscore_zero_mad_deviation_gets_documented_ordinal_score():
    result = fit_predict(
        pd.DataFrame({"x": [4, 4, 4, 4, 5]}), ["x"], RobustZScoreConfig(threshold=2.0)
    )

    assert result.scores == (0.0, 0.0, 0.0, 0.0, 3.0)
    assert result.is_anomaly == (False, False, False, False, True)
    assert "ordinal" in result.metadata["zero_mad"]


def test_robust_zscore_supports_multifeature_row_score_and_config_error():
    result = fit_predict(pd.DataFrame({"x": [0, 0, 0, 20], "y": [5, 5, 5, 5]}), ["x", "y"])
    assert result.is_anomaly[-1]
    assert result.scores[-1] > result.scores[-2]
    with pytest.raises(InvalidDetectorConfigError):
        fit_predict(pd.DataFrame({"x": [1, 2]}), ["x"], RobustZScoreConfig(threshold=0))


def test_robust_zscore_large_finite_mad_keeps_nonzero_score():
    scale = 1.3e308
    frame = pd.DataFrame({"x": np.array([-scale, -scale, 0.0, scale, scale])})
    result = fit_predict(frame, ["x"], RobustZScoreConfig(threshold=0.5))

    assert all(np.isfinite(result.scores))
    assert result.scores[0] == pytest.approx(1 / 1.4826)
    assert result.is_anomaly[0]
