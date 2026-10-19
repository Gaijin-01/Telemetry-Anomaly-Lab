import pandas as pd
import pytest

from anomaly_lab.features import prepare_features
from anomaly_lab.schema import (
    EmptyDataError,
    FeatureSelectionError,
    NonFiniteValueError,
    NonNumericFeatureError,
)


def test_prepares_selected_numeric_features_and_drops_constants_without_mutation():
    frame = pd.DataFrame({"run_id": ["a", "b"], "x": [1, 2], "fixed": [7, 7]})
    original = frame.copy(deep=True)

    prepared = prepare_features(frame, ["x", "fixed"])

    assert prepared.features == ("x",)
    assert prepared.constant_features == ("fixed",)
    assert prepared.values.tolist() == [[1.0], [2.0]]
    assert prepared.index.tolist() == [0, 1]
    pd.testing.assert_frame_equal(frame, original)


def test_rejects_missing_feature():
    with pytest.raises(FeatureSelectionError, match="missing"):
        prepare_features(pd.DataFrame({"x": [1]}), ["absent"])


def test_rejects_non_numeric_feature():
    with pytest.raises(NonNumericFeatureError, match="label"):
        prepare_features(pd.DataFrame({"label": ["x", "y"]}), ["label"])


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_rejects_non_finite_values_with_location(bad):
    with pytest.raises(NonFiniteValueError, match="row"):
        prepare_features(pd.DataFrame({"x": [1.0, bad]}), ["x"])


def test_rejects_empty_rows_features_and_all_constant_features():
    with pytest.raises(EmptyDataError):
        prepare_features(pd.DataFrame(columns=["x"]), ["x"])
    with pytest.raises(FeatureSelectionError):
        prepare_features(pd.DataFrame({"x": [1, 2]}), [])
    with pytest.raises(FeatureSelectionError, match="constant"):
        prepare_features(pd.DataFrame({"x": [1, 1]}), ["x"])
