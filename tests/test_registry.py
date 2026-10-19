"""Tests for the Detector Applicability Registry."""

import pandas as pd
import pytest

from anomaly_lab.registry import (
    ALL_DETECTOR_CONTRACTS,
    ApplicabilityCode,
    DetectorContract,
    check_applicability,
)
from anomaly_lab.semantics import DatasetProfile, DatasetSemantics, classify


def _robust_zscore():
    return ALL_DETECTOR_CONTRACTS["robust-zscore"]


def _isolation_forest():
    return ALL_DETECTOR_CONTRACTS["isolation-forest"]


def _dbscan():
    return ALL_DETECTOR_CONTRACTS["dbscan"]


class TestCheckApplicability:
    def test_robust_zscore_applicable_to_unordered_tabular(self):
        df = pd.DataFrame({"x": [1.0, 2.0, 3.0, 4.0, 5.0]})
        profile = classify(df)
        code, reason = check_applicability(
            _robust_zscore(), profile, n_samples=5, constant_features=(), has_missing=False
        )
        assert code == ApplicabilityCode.APPLICABLE

    def test_robust_zscore_rejected_for_insufficient_samples(self):
        df = pd.DataFrame({"x": [1.0]})
        profile = classify(df)
        code, reason = check_applicability(
            _robust_zscore(), profile, n_samples=1, constant_features=(), has_missing=False
        )
        assert code == ApplicabilityCode.INSUFFICIENT_DATA
        assert "at least 3" in reason

    def test_robust_zscore_rejected_for_constant_features(self):
        df = pd.DataFrame({"x": [1.0, 1.0, 1.0], "y": [1.0, 2.0, 3.0]})
        profile = classify(df)
        code, reason = check_applicability(
            _robust_zscore(), profile, n_samples=3, constant_features=("x",), has_missing=False
        )
        assert code == ApplicabilityCode.INVALID_INPUT
        assert "constant feature" in reason.lower()

    def test_robust_zscore_rejected_for_missing_values(self):
        df = pd.DataFrame({"x": [1.0, None, 3.0]})
        profile = classify(df)
        code, reason = check_applicability(
            _robust_zscore(), profile, n_samples=3, constant_features=(), has_missing=True
        )
        assert code == ApplicabilityCode.INVALID_INPUT

    def test_isolation_forest_applicable_to_unordered_tabular(self):
        df = pd.DataFrame({"x": [1.0, 2.0, 3.0, 4.0, 5.0]})
        profile = classify(df)
        code, reason = check_applicability(
            _isolation_forest(), profile, n_samples=5, constant_features=(), has_missing=False
        )
        assert code == ApplicabilityCode.APPLICABLE

    def test_dbscan_not_applicable_to_streaming_ordered(self):
        profile = DatasetProfile(
            semantics=DatasetSemantics.STREAMING_ORDERED,
            has_timestamp=True,
            has_group_key=False,
            has_pairing_key=False,
            metric_columns=("x",),
            label_column=None,
            anomaly_intervals=(),
            ordering_field="t",
            assumptions=("Streaming data.",),
            warnings=(),
        )
        code, reason = check_applicability(
            _dbscan(), profile, n_samples=5, constant_features=(), has_missing=False
        )
        assert code == ApplicabilityCode.NOT_APPLICABLE
        assert "does not support semantics" in reason

    def test_temporal_order_required_but_not_provided(self):
        contract = DetectorContract(
            name="temporal-detector",
            family="test",
            supported_semantics=frozenset({DatasetSemantics.ORDERED_SERIES}),
            min_samples=2,
            constant_feature_behavior="skip",
            missing_value_policy="skip",
            scaling_required=False,
            temporal_order_required=True,
            stochastic=False,
            random_state_support=False,
            score_direction="higher_is_more_anomalous",
            output_type="point_anomaly",
            score_normalization=None,
        )
        # Data has no ordering field (simple numeric, no timestamp)
        profile = DatasetProfile(
            semantics=DatasetSemantics.ORDERED_SERIES,
            has_timestamp=True,
            has_group_key=False,
            has_pairing_key=False,
            metric_columns=("x",),
            label_column=None,
            anomaly_intervals=(),
            ordering_field=None,  # No ordering field despite being ORDERED_SERIES
            assumptions=(),
            warnings=(),
        )
        code, reason = check_applicability(
            contract, profile, n_samples=3, constant_features=(), has_missing=False
        )
        assert code == ApplicabilityCode.NOT_APPLICABLE
        assert "temporal ordering" in reason.lower()


class TestDetectorContracts:
    def test_all_contracts_have_unique_names(self):
        names = [c.name for c in ALL_DETECTOR_CONTRACTS.values()]
        assert len(names) == len(set(names))

    def test_robust_zscore_is_deterministic(self):
        rz = _robust_zscore()
        assert rz.stochastic is False
        assert rz.random_state_support is False
        assert rz.scaling_required is False

    def test_isolation_forest_is_stochastic(self):
        if_ = _isolation_forest()
        assert if_.stochastic is True
        assert if_.random_state_support is True
        assert if_.scaling_required is True

    def test_dbscan_is_cluster_output_type(self):
        db = _dbscan()
        assert db.output_type == "cluster/noise"
        assert db.stochastic is False

    def test_all_contracts_have_valid_score_direction(self):
        for contract in ALL_DETECTOR_CONTRACTS.values():
            assert contract.score_direction in (
                "higher_is_more_anomalous",
                "lower_is_more_anomalous",
            )

    def test_all_contracts_have_min_samples_positive(self):
        for contract in ALL_DETECTOR_CONTRACTS.values():
            assert contract.min_samples >= 1

    def test_all_contracts_have_supported_semantics(self):
        for contract in ALL_DETECTOR_CONTRACTS.values():
            assert len(contract.supported_semantics) > 0


class TestApplicabilityCode:
    def test_all_codes_are_strings(self):
        for code in ApplicabilityCode:
            assert isinstance(code.value, str)

    def test_all_expected_codes_present(self):
        expected = {"APPLICABLE", "NOT_APPLICABLE", "INSUFFICIENT_DATA", "INVALID_INPUT"}
        actual = {c.value for c in ApplicabilityCode}
        assert expected == actual


class TestDetectorContract:
    def test_contract_is_frozen(self):
        contract = _robust_zscore()
        with pytest.raises(AttributeError):
            contract.name = "changed"

    def test_contract_has_all_required_fields(self):
        contract = _robust_zscore()
        assert hasattr(contract, "name")
        assert hasattr(contract, "family")
        assert hasattr(contract, "supported_semantics")
        assert hasattr(contract, "min_samples")
        assert hasattr(contract, "constant_feature_behavior")
        assert hasattr(contract, "missing_value_policy")
        assert hasattr(contract, "scaling_required")
        assert hasattr(contract, "temporal_order_required")
        assert hasattr(contract, "stochastic")
        assert hasattr(contract, "random_state_support")
        assert hasattr(contract, "score_direction")
        assert hasattr(contract, "output_type")
        assert hasattr(contract, "score_normalization")
