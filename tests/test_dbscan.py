import pandas as pd
import pytest

from anomaly_lab.detectors.dbscan import DBSCANConfig, fit_predict
from anomaly_lab.schema import InvalidDetectorConfigError


def test_dbscan_scales_features_detects_two_dense_groups_and_noise_deterministically():
    frame = pd.DataFrame(
        {
            "util": [10, 11, 9, 90, 91, 89, 50],
            "vram": [10000, 10010, 9990, 40000, 40010, 39990, 25000],
        }
    )
    before = frame.copy(deep=True)
    config = DBSCANConfig(eps=0.35, min_samples=2)
    result = fit_predict(frame, ["util", "vram"], config)
    repeated = fit_predict(frame, ["util", "vram"], config)

    assert result.clusters == repeated.clusters
    assert result.clusters is not None
    assert result.clusters[-1] == -1
    assert result.is_anomaly[-1]
    assert len(set(result.clusters[:-1])) == 2
    assert result.scores is None
    assert result.metadata["scaler"] == "StandardScaler"
    pd.testing.assert_frame_equal(frame, before)


def test_dbscan_validates_configuration():
    with pytest.raises(InvalidDetectorConfigError):
        fit_predict(pd.DataFrame({"x": [1, 2]}), ["x"], DBSCANConfig(eps=0))
