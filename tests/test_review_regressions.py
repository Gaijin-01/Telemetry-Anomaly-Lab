import json

import pandas as pd
import pytest

from anomaly_lab.detectors.dbscan import DBSCANConfig
from anomaly_lab.detectors.dbscan import fit_predict as dbscan_fit
from anomaly_lab.detectors.isolation_forest import (
    IsolationForestConfig,
)
from anomaly_lab.detectors.isolation_forest import (
    fit_predict as isolation_fit,
)
from anomaly_lab.detectors.robust_zscore import (
    RobustZScoreConfig,
)
from anomaly_lab.detectors.robust_zscore import (
    fit_predict as robust_fit,
)
from anomaly_lab.schema import InvalidDetectorConfigError


@pytest.mark.parametrize(
    "config",
    [
        IsolationForestConfig(contamination="bad"),
        IsolationForestConfig(n_estimators=2.5),
        IsolationForestConfig(n_estimators="2"),
    ],
)
def test_isolation_forest_invalid_config_has_domain_error(config):
    with pytest.raises(InvalidDetectorConfigError):
        isolation_fit(pd.DataFrame({"x": [1.0, 2.0, 3.0]}), ["x"], config)


def test_dbscan_invalid_epsilon_has_domain_error():
    with pytest.raises(InvalidDetectorConfigError):
        dbscan_fit(pd.DataFrame({"x": [1.0, 2.0, 3.0]}), ["x"], DBSCANConfig(eps="bad"))


def test_robust_zscore_invalid_threshold_has_domain_error():
    with pytest.raises(InvalidDetectorConfigError):
        robust_fit(
            pd.DataFrame({"x": [1.0, 2.0, 3.0]}),
            ["x"],
            RobustZScoreConfig(threshold="bad"),
        )


def test_isolation_forest_rejects_out_of_range_random_state():
    with pytest.raises(InvalidDetectorConfigError):
        isolation_fit(
            pd.DataFrame({"x": [1.0, 2.0, 3.0]}),
            ["x"],
            IsolationForestConfig(random_state=-1),
        )


def test_datetime_index_results_are_json_serializable_and_stable():
    frame = pd.DataFrame(
        {"x": [1.0, 1.1, 1.0, 50.0]},
        index=pd.date_range("2024-01-01", periods=4, freq="h", name="timestamp"),
    )
    result = robust_fit(frame, ["x"], RobustZScoreConfig(threshold=2.0))

    encoded = json.dumps(result.to_dict(), allow_nan=False)
    decoded = json.loads(encoded)

    assert [row["index"] for row in decoded["rows"]] == [
        timestamp.isoformat() for timestamp in frame.index
    ]
    assert json.dumps(result.to_dict(), allow_nan=False) == encoded
