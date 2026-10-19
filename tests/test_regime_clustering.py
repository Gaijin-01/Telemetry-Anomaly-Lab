import pandas as pd
import pytest

from anomaly_lab.detectors.dbscan import DBSCANConfig, fit_predict
from anomaly_lab.schema import FeatureSelectionError, NonNumericFeatureError


def test_regimes_cluster_explicit_numeric_features_and_keep_noise():
    frame = pd.DataFrame({"x": [0.0, 0.1, -0.1, 10.0, 10.1, 9.9, 5.0]})
    result = fit_predict(frame, ["x"], DBSCANConfig(eps=0.3, min_samples=2))
    labels = result.clusters
    assert labels is not None
    assert labels[:3] == (labels[0],) * 3
    assert labels[3:6] == (labels[3],) * 3
    assert labels[0] != labels[3]
    assert labels[-1] == -1
    assert result.is_anomaly[-1]
    assert result.metadata["cluster_id_semantics"] == "opaque; no semantic meaning"


def test_metadata_is_returned_but_never_part_of_feature_matrix():
    frame = pd.DataFrame(
        {"x": [0.0, 0.1, -0.1, 10.0, 10.1, 9.9], "machine": ["a", "a", "a", "b", "b", "b"]}
    )
    result = fit_predict(
        frame, ["x"], DBSCANConfig(eps=0.3, min_samples=2), metadata_columns=["machine"]
    )
    assert result.features == ("x",)
    assert result.clusters[:3] == (result.clusters[0],) * 3
    assert result.clusters[3:] == (result.clusters[3],) * 3
    assert result.clusters[0] != result.clusters[3]
    assert result.metadata["row_metadata"] == (
        {"machine": "a"},
        {"machine": "a"},
        {"machine": "a"},
        {"machine": "b"},
        {"machine": "b"},
        {"machine": "b"},
    )


def test_metadata_columns_must_be_explicit_and_valid():
    frame = pd.DataFrame({"x": [1, 2, 3], "machine": ["a", "a", "b"]})
    with pytest.raises(FeatureSelectionError):
        fit_predict(frame, ["x"], metadata_columns=["missing"])
    with pytest.raises(FeatureSelectionError):
        fit_predict(frame, ["x"], metadata_columns=["x"])
    with pytest.raises(FeatureSelectionError):
        fit_predict(frame, ["x"], metadata_columns=["machine", "machine"])
    with pytest.raises(NonNumericFeatureError):
        fit_predict(frame, ["x", "machine"])


def test_clustering_does_not_mutate_input():
    frame = pd.DataFrame({"x": [0.0, 0.1, 10.0, 10.1], "group": ["a", "a", "b", "b"]})
    before = frame.copy(deep=True)
    fit_predict(
        frame, ["x"], metadata_columns=["group"], config=DBSCANConfig(eps=0.2, min_samples=2)
    )
    pd.testing.assert_frame_equal(frame, before)
