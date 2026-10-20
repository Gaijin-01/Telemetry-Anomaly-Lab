"""Tests for ExperimentSpec and ExperimentResult schema."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from anomaly_lab.experiments import ExperimentResult, ExperimentSpec
from anomaly_lab.semantics import DatasetSemantics


class TestExperimentSpec:
    def test_spec_creates_minimal_experiment(self):
        spec = ExperimentSpec(
            experiment_name="test-exp",
            detector_name="robust-zscore",
            features=("x", "y"),
            dataset_id="dataset-001",
            input_hash="abc123",
        )
        assert spec.experiment_name == "test-exp"
        assert spec.detector_name == "robust-zscore"
        assert spec.features == ("x", "y")
        assert spec.dataset_id == "dataset-001"
        assert spec.input_hash == "abc123"

    def test_spec_with_all_fields(self):
        spec = ExperimentSpec(
            experiment_name="full-exp",
            detector_name="isolation-forest",
            features=("a", "b", "c"),
            dataset_id="dataset-002",
            input_hash="def456",
            parameters={"contamination": 0.1, "n_estimators": 200},
            random_seed=42,
            semantics=DatasetSemantics.UNORDERED_TABULAR,
            constant_features=("z",),
            metadata={"source": "synthetic"},
        )
        assert spec.parameters == {"contamination": 0.1, "n_estimators": 200}
        assert spec.random_seed == 42
        assert spec.semantics == DatasetSemantics.UNORDERED_TABULAR
        assert spec.constant_features == ("z",)
        assert spec.metadata == {"source": "synthetic"}

    def test_spec_rejects_empty_experiment_name(self):
        with pytest.raises(ValueError, match="experiment_name"):
            ExperimentSpec(
                experiment_name="",
                detector_name="robust-zscore",
                features=("x",),
                dataset_id="ds001",
                input_hash="hash",
            )

    def test_spec_rejects_empty_detector_name(self):
        with pytest.raises(ValueError, match="detector_name"):
            ExperimentSpec(
                experiment_name="exp",
                detector_name="",
                features=("x",),
                dataset_id="ds001",
                input_hash="hash",
            )

    def test_spec_rejects_empty_features(self):
        with pytest.raises(ValueError, match="features"):
            ExperimentSpec(
                experiment_name="exp",
                detector_name="robust-zscore",
                features=(),
                dataset_id="ds001",
                input_hash="hash",
            )

    def test_spec_rejects_duplicate_features(self):
        with pytest.raises(ValueError, match="duplicate"):
            ExperimentSpec(
                experiment_name="exp",
                detector_name="robust-zscore",
                features=("x", "x"),
                dataset_id="ds001",
                input_hash="hash",
            )

    def test_spec_rejects_empty_dataset_id(self):
        with pytest.raises(ValueError, match="dataset_id"):
            ExperimentSpec(
                experiment_name="exp",
                detector_name="robust-zscore",
                features=("x",),
                dataset_id="",
                input_hash="hash",
            )

    def test_spec_rejects_empty_input_hash(self):
        with pytest.raises(ValueError, match="input_hash"):
            ExperimentSpec(
                experiment_name="exp",
                detector_name="robust-zscore",
                features=("x",),
                dataset_id="ds001",
                input_hash="",
            )

    def test_spec_to_dict_roundtrip(self):
        spec = ExperimentSpec(
            experiment_name="roundtrip-test",
            detector_name="dbscan",
            features=("p", "q"),
            dataset_id="ds003",
            input_hash="hash789",
            parameters={"eps": 0.5, "min_samples": 5},
            random_seed=123,
            semantics=DatasetSemantics.ORDERED_SERIES,
            constant_features=("r",),
            metadata={"note": "test roundtrip"},
        )
        d = spec.to_dict()
        # Should be JSON serializable
        encoded = json.dumps(d, allow_nan=False)
        decoded = json.loads(encoded)
        assert decoded["experiment_name"] == "roundtrip-test"
        assert decoded["detector_name"] == "dbscan"
        assert decoded["features"] == ["p", "q"]

    def test_spec_is_hashable(self):
        spec = ExperimentSpec(
            experiment_name="hashable-test",
            detector_name="robust-zscore",
            features=("x",),
            dataset_id="ds001",
            input_hash="hashabc",
        )
        h = hash(spec)
        assert isinstance(h, int)


class TestExperimentResult:
    def test_result_creates_from_spec(self):
        spec = ExperimentSpec(
            experiment_name="result-test",
            detector_name="robust-zscore",
            features=("x", "y"),
            dataset_id="ds001",
            input_hash="hashxyz",
        )
        result = ExperimentResult(
            spec=spec,
            row_indices=(0, 5, 10),
            is_anomaly=(False, True, False),
            scores=(0.1, 0.9, 0.2),
        )
        assert result.spec == spec
        assert result.row_indices == (0, 5, 10)
        assert result.is_anomaly == (False, True, False)
        assert result.scores == (0.1, 0.9, 0.2)

    def test_result_rejects_mismatched_lengths(self):
        spec = ExperimentSpec(
            experiment_name="len-mismatch",
            detector_name="robust-zscore",
            features=("x",),
            dataset_id="ds001",
            input_hash="hash123",
        )
        with pytest.raises(ValueError, match="length"):
            ExperimentResult(
                spec=spec,
                row_indices=(0, 1, 2),
                is_anomaly=(False, True),  # only 2
                scores=(0.1, 0.9, 0.5),
            )

    def test_result_rejects_empty_row_indices(self):
        spec = ExperimentSpec(
            experiment_name="empty-result",
            detector_name="robust-zscore",
            features=("x",),
            dataset_id="ds001",
            input_hash="hash456",
        )
        with pytest.raises(ValueError, match="zero rows"):
            ExperimentResult(
                spec=spec,
                row_indices=(),
                is_anomaly=(),
                scores=None,
            )

    def test_result_rejects_nonfinite_scores(self):
        spec = ExperimentSpec(
            experiment_name="nonfinite-scores",
            detector_name="robust-zscore",
            features=("x",),
            dataset_id="ds001",
            input_hash="hash789",
        )
        with pytest.raises(ValueError, match="finite"):
            ExperimentResult(
                spec=spec,
                row_indices=(0, 1),
                is_anomaly=(False, True),
                scores=(float("inf"), 0.5),
            )

    def test_result_confidence_clamped_to_valid_range(self):
        spec = ExperimentSpec(
            experiment_name="bad-confidence",
            detector_name="robust-zscore",
            features=("x",),
            dataset_id="ds001",
            input_hash="hashabc",
        )
        result = ExperimentResult(
            spec=spec,
            row_indices=(0,),
            is_anomaly=(True,),
            scores=(0.9,),
            confidence=-0.1,
        )
        assert result.confidence == 0.0

    def test_result_confidence_clamped_above_range(self):
        spec = ExperimentSpec(
            experiment_name="confidence-clamp",
            detector_name="robust-zscore",
            features=("x",),
            dataset_id="ds001",
            input_hash="hashdef",
        )
        result = ExperimentResult(
            spec=spec,
            row_indices=(0,),
            is_anomaly=(True,),
            scores=(0.9,),
            confidence=1.5,  # > 1.0
        )
        # confidence should be clamped to [0, 1] in __post_init__
        assert result.confidence == 1.0

    def test_result_to_dict_roundtrip(self):
        spec = ExperimentSpec(
            experiment_name="dict-roundtrip",
            detector_name="isolation-forest",
            features=("x", "y"),
            dataset_id="ds002",
            input_hash="hash123",
            random_seed=42,
        )
        result = ExperimentResult(
            spec=spec,
            row_indices=(np.int64(0), np.int64(5)),
            is_anomaly=(False, True),
            scores=(0.1, 0.95),
            clusters=(0, 1),
            metadata={"run": 1},
        )
        d = result.to_dict()
        encoded = json.dumps(d, allow_nan=False)
        decoded = json.loads(encoded)
        assert decoded["spec"]["experiment_name"] == "dict-roundtrip"
        assert decoded["row_indices"] == [0, 5]
        assert decoded["scores"] == [0.1, 0.95]

    def test_result_hash_is_deterministic(self):
        spec1 = ExperimentSpec(
            experiment_name="deterministic-hash",
            detector_name="robust-zscore",
            features=("x",),
            dataset_id="ds001",
            input_hash="hashabc",
            random_seed=42,
        )
        result1 = ExperimentResult(
            spec=spec1,
            row_indices=(0, 1, 2),
            is_anomaly=(False, True, False),
            scores=(0.1, 0.9, 0.2),
        )
        result2 = ExperimentResult(
            spec=spec1,
            row_indices=(0, 1, 2),
            is_anomaly=(False, True, False),
            scores=(0.1, 0.9, 0.2),
        )
        # Same inputs + spec + seed = same hash
        assert hash(result1) == hash(result2)

    def test_result_hash_changes_with_different_spec(self):
        spec1 = ExperimentSpec(
            experiment_name="spec-a",
            detector_name="robust-zscore",
            features=("x",),
            dataset_id="ds001",
            input_hash="hashabc",
        )
        spec2 = ExperimentSpec(
            experiment_name="spec-b",
            detector_name="robust-zscore",
            features=("x",),
            dataset_id="ds001",
            input_hash="hashabc",
        )
        result1 = ExperimentResult(
            spec=spec1,
            row_indices=(0, 1),
            is_anomaly=(False, True),
            scores=(0.1, 0.9),
        )
        result2 = ExperimentResult(
            spec=spec2,
            row_indices=(0, 1),
            is_anomaly=(False, True),
            scores=(0.1, 0.9),
        )
        assert hash(result1) != hash(result2)
