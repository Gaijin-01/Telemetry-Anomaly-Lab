import numpy as np
import pandas as pd
import pytest

from anomaly_lab.detectors.isolation_forest import (
    IsolationForestConfig,
    fit_predict,
)
from anomaly_lab.schema import InvalidDetectorConfigError


def test_isolation_forest_accepts_numpy_integer_random_state():
    result = fit_predict(
        pd.DataFrame({"x": [1.0, 2.0, 3.0, 4.0]}),
        ["x"],
        IsolationForestConfig(n_estimators=10, random_state=np.int64(1)),
    )
    assert result.parameters["random_state"] == 1


def test_isolation_forest_rejects_invalid_seed_values_with_domain_error():
    for seed in [-1, 2**32, "1", True]:
        with pytest.raises(InvalidDetectorConfigError):
            fit_predict(
                pd.DataFrame({"x": [1.0, 2.0, 3.0, 4.0]}),
                ["x"],
                IsolationForestConfig(n_estimators=10, random_state=seed),
            )
