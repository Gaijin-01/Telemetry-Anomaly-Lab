import json

import pandas as pd
import pytest

from anomaly_lab.detectors.dbscan import DBSCANConfig, fit_predict
from anomaly_lab.schema import FeatureSelectionError


def test_metadata_column_selector_string_is_rejected():
    frame = pd.DataFrame({"x": [1.0, 1.1, 1.2], "machine": ["a", "a", "b"]})
    with pytest.raises(FeatureSelectionError):
        fit_predict(frame, ["x"], metadata_columns="machine")


def test_metadata_selector_nonstring_values_are_rejected_domain_safely():
    frame = pd.DataFrame({"x": [1.0, 1.1, 1.2], "machine": ["a", "a", "b"]})
    with pytest.raises(FeatureSelectionError):
        fit_predict(frame, ["x"], metadata_columns=[3])


def test_nullable_metadata_values_serialize_to_json():
    frame = pd.DataFrame(
        {"x": [1.0, 1.1, 1.2], "machine": pd.Series(["a", pd.NA, "b"], dtype="string")}
    )
    result = fit_predict(
        frame,
        ["x"],
        config=DBSCANConfig(eps=0.5, min_samples=1),
        metadata_columns=["machine"],
    )
    payload = result.to_dict()
    assert json.dumps(payload, allow_nan=False)
    assert payload["metadata"]["row_metadata"][1]["machine"] is None
