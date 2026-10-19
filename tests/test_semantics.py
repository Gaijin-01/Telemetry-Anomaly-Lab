import pandas as pd
import pytest

from anomaly_lab.semantics import (
    DatasetSemantics,
    classify,
)


class TestClassifyUnorderedTabular:
    def test_plain_numeric_matrix_is_unordered_tabular(self):
        df = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [4.0, 5.0, 6.0]})
        profile = classify(df)
        assert profile.semantics == DatasetSemantics.UNORDERED_TABULAR
        assert profile.has_timestamp is False
        assert profile.has_group_key is False
        assert profile.has_pairing_key is False
        assert profile.ordering_field is None

    def test_metric_columns_detected(self):
        df = pd.DataFrame({"x": [1.0, 2.0, 3.0], "y": [4.0, 5.0, 6.0], "group": ["A", "B", "A"]})
        profile = classify(df)
        assert set(profile.metric_columns) == {"x", "y"}

    def test_label_column_detected_bool(self):
        df = pd.DataFrame({"x": [1.0, 2.0, 3.0], "anomaly": [False, True, False]})
        profile = classify(df)
        assert profile.label_column == "anomaly"

    def test_label_column_detected_binary_int(self):
        df = pd.DataFrame({"x": [1.0, 2.0, 3.0], "label": [0, 1, 0]})
        profile = classify(df)
        assert profile.label_column == "label"

    def test_constant_columns_not_metrics(self):
        df = pd.DataFrame({"x": [1.0, 1.0, 1.0], "y": [1.0, 2.0, 3.0]})
        profile = classify(df)
        assert "x" not in profile.metric_columns
        assert "y" in profile.metric_columns

    def test_group_key_detected(self):
        df = pd.DataFrame({"x": [1.0, 2.0, 3.0], "group": ["A", "B", "A"]})
        profile = classify(df)
        assert profile.has_group_key is True

    def test_empty_dataframe_raises(self):
        df = pd.DataFrame()
        with pytest.raises(Exception, match="non-empty"):
            classify(df)

    def test_non_dataframe_raises(self):
        from anomaly_lab.schema import AnomalyLabError

        with pytest.raises(AnomalyLabError):
            classify("not a dataframe")


class TestClassifyOrderedSeries:
    def test_datetime_column_with_increasing_values_is_ordered_series(self):
        df = pd.DataFrame(
            {
                "timestamp": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"]),
                "value": [1.0, 2.0, 3.0],
            }
        )
        profile = classify(df)
        assert profile.semantics == DatasetSemantics.ORDERED_SERIES
        assert profile.has_timestamp is True
        assert profile.ordering_field == "timestamp"

    def test_numeric_increasing_ordering_field(self):
        df = pd.DataFrame({"t": [1, 2, 3, 4], "value": [1.0, 2.0, 3.0, 4.0]})
        profile = classify(df)
        assert profile.semantics == DatasetSemantics.ORDERED_SERIES
        assert profile.ordering_field == "t"

    def test_timestamp_not_strictly_increasing_treated_as_unordered(self):
        df = pd.DataFrame(
            {
                "timestamp": pd.to_datetime(["2024-01-01", "2024-01-01", "2024-01-03"]),
                "value": [1.0, 2.0, 3.0],
            }
        )
        profile = classify(df)
        # Warning should be issued about non-strict ordering
        assert profile.has_timestamp is True
        assert profile.ordering_field is None


class TestClassifyPairedExperiment:
    def test_pairing_key_detected(self):
        df = pd.DataFrame(
            {
                "subject_id": [1, 1, 2, 2],
                "condition": ["A", "B", "A", "B"],
                "value": [1.0, 2.0, 3.0, 4.0],
            }
        )
        profile = classify(df)
        assert profile.has_pairing_key is True


class TestClassifyGrouped:
    def test_group_columns_detected(self):
        df = pd.DataFrame(
            {
                "group": ["X", "X", "Y", "Y"],
                "value": [1.0, 2.0, 3.0, 4.0],
            }
        )
        profile = classify(df)
        assert profile.semantics == DatasetSemantics.GROUPED
        assert profile.has_group_key is True


class TestAnomalyIntervals:
    def test_anomaly_intervals_detected(self):
        df = pd.DataFrame(
            {
                "value": [1.0, 2.0, 3.0, 4.0, 5.0],
                "anomaly": [False, True, True, False, True],
            }
        )
        profile = classify(df)
        assert profile.label_column == "anomaly"
        assert profile.anomaly_intervals == ((1, 2), (4, 4))

    def test_no_anomaly_intervals_when_no_label(self):
        df = pd.DataFrame({"value": [1.0, 2.0, 3.0]})
        profile = classify(df)
        assert profile.anomaly_intervals == ()


class TestDatasetProfile:
    def test_dataset_profile_is_frozen(self):
        df = pd.DataFrame({"x": [1.0, 2.0]})
        profile = classify(df)
        with pytest.raises(AttributeError):
            profile.semantics = DatasetSemantics.ORDERED_SERIES

    def test_profile_has_required_fields(self):
        df = pd.DataFrame({"x": [1.0, 2.0, 3.0]})
        profile = classify(df)
        assert hasattr(profile, "semantics")
        assert hasattr(profile, "has_timestamp")
        assert hasattr(profile, "has_group_key")
        assert hasattr(profile, "has_pairing_key")
        assert hasattr(profile, "metric_columns")
        assert hasattr(profile, "label_column")
        assert hasattr(profile, "anomaly_intervals")
        assert hasattr(profile, "ordering_field")
        assert hasattr(profile, "assumptions")
        assert hasattr(profile, "warnings")

    def test_assumptions_tuple_not_empty_for_ordered_series(self):
        df = pd.DataFrame(
            {
                "t": [1, 2, 3],
                "value": [1.0, 2.0, 3.0],
            }
        )
        profile = classify(df)
        assert len(profile.assumptions) > 0


class TestTimestampColumnDetection:
    @pytest.mark.parametrize(
        "col_name",
        [
            "timestamp",
            "Timestamp",
            "TIMESTAMP",
            "time",
            "datetime",
            "ts",
            "dt",
        ],
    )
    def test_timestamp_column_patterns(self, col_name):
        df = pd.DataFrame({col_name: [1, 2, 3], "value": [1.0, 2.0, 3.0]})
        profile = classify(df)
        assert profile.has_timestamp is True


class TestDatasetSemanticsEnum:
    def test_all_semantics_values_are_strings(self):
        for sem in DatasetSemantics:
            assert isinstance(sem.value, str)

    def test_semantics_enum_count(self):
        assert len(DatasetSemantics) == 6
