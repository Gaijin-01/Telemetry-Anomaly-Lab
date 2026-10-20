"""Tests for ExperimentSpec and ExperimentResult schema."""

from __future__ import annotations

import json

import numpy as np
import pytest

from anomaly_lab.experiments import ExperimentResult, ExperimentSpec


class TestExperimentSpec:
    def test_spec_creates_minimal_experiment(self):
        spec = ExperimentSpec(
            experiment_id="test-exp",
            dataset_path="/data/telemetry.csv",
            dataset_hash="abc123",
            features=("x", "y"),
        )
        assert spec.experiment_id == "test-exp"
        assert spec.dataset_path == "/data/telemetry.csv"
        assert spec.dataset_hash == "abc123"
        assert spec.features == ("x", "y")
        assert spec.analysis_type == "anomaly_detection"
        assert spec.detectors == ()
        assert spec.random_seed == 42

    def test_spec_with_all_fields(self):
        spec = ExperimentSpec(
            experiment_id="full-exp",
            dataset_path="/data/baseline.csv",
            dataset_hash="def456",
            features=("a", "b", "c"),
            analysis_type="compare",
            detectors=("robust-zscore", "isolation-forest"),
            baseline_role="baseline",
            candidate_role="candidate",
            metric_direction="higher_is_better",
            thresholds=(("auc", 0.9),),
            random_seed=123,
            allowed_changed_dimensions=("version",),
            metadata=(("model", "v2"),),
            environment_metadata=(("python", "3.11"),),
            explicit_assumptions=("baseline is stable",),
        )
        assert spec.analysis_type == "compare"
        assert spec.baseline_role == "baseline"
        assert spec.candidate_role == "candidate"
        assert spec.thresholds == (("auc", 0.9),)
        assert spec.random_seed == 123
        assert spec.allowed_changed_dimensions == ("version",)

    def test_spec_rejects_empty_experiment_id(self):
        with pytest.raises(ValueError, match="experiment_id"):
            ExperimentSpec(
                experiment_id="",
                dataset_path="/data/test.csv",
                dataset_hash="abc",
                features=("x",),
            )

    def test_spec_rejects_empty_dataset_path(self):
        with pytest.raises(ValueError, match="dataset_path"):
            ExperimentSpec(
                experiment_id="exp",
                dataset_path="",
                dataset_hash="abc",
                features=("x",),
            )

    def test_spec_rejects_empty_dataset_hash(self):
        with pytest.raises(ValueError, match="dataset_hash"):
            ExperimentSpec(
                experiment_id="exp",
                dataset_path="/data/test.csv",
                dataset_hash="",
                features=("x",),
            )

    def test_spec_rejects_empty_features(self):
        with pytest.raises(ValueError, match="features"):
            ExperimentSpec(
                experiment_id="exp",
                dataset_path="/data/test.csv",
                dataset_hash="abc",
                features=(),
            )

    def test_spec_rejects_duplicate_features(self):
        with pytest.raises(ValueError, match="duplicate"):
            ExperimentSpec(
                experiment_id="exp",
                dataset_path="/data/test.csv",
                dataset_hash="abc",
                features=("x", "x"),
            )

    def test_spec_rejects_invalid_analysis_type(self):
        with pytest.raises(ValueError, match="analysis_type"):
            ExperimentSpec(
                experiment_id="exp",
                dataset_path="/data/test.csv",
                dataset_hash="abc",
                features=("x",),
                analysis_type="invalid_type",
            )

    def test_spec_rejects_compare_without_baseline(self):
        with pytest.raises(ValueError, match="baseline_role"):
            ExperimentSpec(
                experiment_id="exp",
                dataset_path="/data/test.csv",
                dataset_hash="abc",
                features=("x",),
                analysis_type="compare",
                candidate_role="candidate",
            )

    def test_spec_rejects_compare_without_candidate(self):
        with pytest.raises(ValueError, match="candidate_role"):
            ExperimentSpec(
                experiment_id="exp",
                dataset_path="/data/test.csv",
                dataset_hash="abc",
                features=("x",),
                analysis_type="compare",
                baseline_role="baseline",
            )

    def test_spec_to_dict_roundtrip(self):
        spec = ExperimentSpec(
            experiment_id="roundtrip-test",
            dataset_path="/data/test.csv",
            dataset_hash="abc123",
            features=("p", "q"),
            random_seed=42,
        )
        d = spec.to_dict()
        # Must be JSON-serializable
        encoded = json.dumps(d, allow_nan=False)
        decoded = json.loads(encoded)
        assert decoded["experiment_id"] == "roundtrip-test"
        assert decoded["features"] == ["p", "q"]
        assert decoded["dataset_hash"] == "abc123"

    def test_spec_is_hashable(self):
        spec = ExperimentSpec(
            experiment_id="hashable-test",
            dataset_path="/data/test.csv",
            dataset_hash="abc",
            features=("x",),
        )
        h = hash(spec)
        assert isinstance(h, int)

    def test_spec_frozen_immutable(self):
        spec = ExperimentSpec(
            experiment_id="frozen-test",
            dataset_path="/data/test.csv",
            dataset_hash="abc",
            features=("x",),
        )
        with pytest.raises(AttributeError):
            spec.experiment_id = "changed"


class TestExperimentResult:
    def test_result_creates_from_spec(self):
        spec = ExperimentSpec(
            experiment_id="result-test",
            dataset_path="/data/test.csv",
            dataset_hash="abc",
            features=("x", "y"),
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
            experiment_id="len-mismatch",
            dataset_path="/data/test.csv",
            dataset_hash="abc",
            features=("x",),
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
            experiment_id="empty-result",
            dataset_path="/data/test.csv",
            dataset_hash="abc",
            features=("x",),
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
            experiment_id="nonfinite-scores",
            dataset_path="/data/test.csv",
            dataset_hash="abc",
            features=("x",),
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
            experiment_id="bad-confidence",
            dataset_path="/data/test.csv",
            dataset_hash="abc",
            features=("x",),
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
            experiment_id="confidence-clamp",
            dataset_path="/data/test.csv",
            dataset_hash="abc",
            features=("x",),
        )
        result = ExperimentResult(
            spec=spec,
            row_indices=(0,),
            is_anomaly=(True,),
            scores=(0.9,),
            confidence=1.5,  # > 1.0
        )
        assert result.confidence == 1.0

    def test_result_to_dict_roundtrip(self):
        spec = ExperimentSpec(
            experiment_id="dict-roundtrip",
            dataset_path="/data/test.csv",
            dataset_hash="abc",
            features=("x", "y"),
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
        assert decoded["spec"]["experiment_id"] == "dict-roundtrip"
        assert decoded["row_indices"] == [0, 5]
        assert decoded["scores"] == [0.1, 0.95]
        assert decoded["clusters"] == [0, 1]
