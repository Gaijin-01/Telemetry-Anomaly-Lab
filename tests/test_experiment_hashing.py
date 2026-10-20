"""Tests for canonical hashing, provenance, and replay of experiments."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from anomaly_lab.experiments import ExperimentResult, ExperimentSpec
from anomaly_lab.experiments.hashing import (
    ExperimentHasher,
    canonical_experiment_hash,
    canonical_parameters_hash,
    canonical_spec_hash,
)
from anomaly_lab.experiments.provenance import ProvenanceRecord, collect_environment
from anomaly_lab.experiments.replay import ExperimentReplay


class TestCanonicalSpecHash:
    def test_same_spec_produces_same_hash(self):
        spec1 = ExperimentSpec(
            experiment_name="hash-consistency",
            detector_name="robust-zscore",
            features=("x", "y"),
            dataset_id="ds001",
            input_hash="abc123",
            random_seed=42,
        )
        spec2 = ExperimentSpec(
            experiment_name="hash-consistency",
            detector_name="robust-zscore",
            features=("x", "y"),
            dataset_id="ds001",
            input_hash="abc123",
            random_seed=42,
        )
        assert canonical_spec_hash(spec1) == canonical_spec_hash(spec2)

    def test_different_seed_produces_different_hash(self):
        spec1 = ExperimentSpec(
            experiment_name="seed-test",
            detector_name="robust-zscore",
            features=("x",),
            dataset_id="ds001",
            input_hash="abc123",
            random_seed=42,
        )
        spec2 = ExperimentSpec(
            experiment_name="seed-test",
            detector_name="robust-zscore",
            features=("x",),
            dataset_id="ds001",
            input_hash="abc123",
            random_seed=99,
        )
        assert canonical_spec_hash(spec1) != canonical_spec_hash(spec2)

    def test_different_features_produce_different_hash(self):
        spec1 = ExperimentSpec(
            experiment_name="feature-hash",
            detector_name="robust-zscore",
            features=("x", "y"),
            dataset_id="ds001",
            input_hash="abc123",
        )
        spec2 = ExperimentSpec(
            experiment_name="feature-hash",
            detector_name="robust-zscore",
            features=("x", "z"),
            dataset_id="ds001",
            input_hash="abc123",
        )
        assert canonical_spec_hash(spec1) != canonical_spec_hash(spec2)

    def test_spec_hash_is_hex_string(self):
        spec = ExperimentSpec(
            experiment_name="hex-test",
            detector_name="robust-zscore",
            features=("x",),
            dataset_id="ds001",
            input_hash="abc123",
        )
        h = canonical_spec_hash(spec)
        assert isinstance(h, str)
        assert len(h) == 64  # SHA256 hex
        assert all(c in "0123456789abcdef" for c in h)

    def test_feature_order_does_not_affect_hash(self):
        """Features are sorted canonically so order doesn't change hash."""
        spec1 = ExperimentSpec(
            experiment_name="order-test",
            detector_name="robust-zscore",
            features=("a", "b", "c"),
            dataset_id="ds001",
            input_hash="abc123",
        )
        spec2 = ExperimentSpec(
            experiment_name="order-test",
            detector_name="robust-zscore",
            features=("c", "a", "b"),
            dataset_id="ds001",
            input_hash="abc123",
        )
        # Both have same features just different order
        assert canonical_spec_hash(spec1) == canonical_spec_hash(spec2)


class TestCanonicalParametersHash:
    def test_same_parameters_produce_same_hash(self):
        params1 = {"contamination": 0.1, "n_estimators": 200}
        params2 = {"contamination": 0.1, "n_estimators": 200}
        assert canonical_parameters_hash(params1) == canonical_parameters_hash(params2)

    def test_different_parameters_produce_different_hash(self):
        params1 = {"contamination": 0.1}
        params2 = {"contamination": 0.2}
        assert canonical_parameters_hash(params1) != canonical_parameters_hash(params2)

    def test_parameters_hash_is_hex_string(self):
        h = canonical_parameters_hash({"key": "value"})
        assert isinstance(h, str)
        assert len(h) == 64

    def test_none_parameters_produces_consistent_hash(self):
        h = canonical_parameters_hash(None)
        assert isinstance(h, str)
        assert len(h) == 64


class TestCanonicalExperimentHash:
    def test_result_hash_includes_spec_and_results(self):
        spec = ExperimentSpec(
            experiment_name="result-hash-inclusion",
            detector_name="robust-zscore",
            features=("x",),
            dataset_id="ds001",
            input_hash="abc123",
            random_seed=42,
        )
        result1 = ExperimentResult(
            spec=spec,
            row_indices=(0, 1),
            is_anomaly=(False, True),
            scores=(0.1, 0.9),
        )
        result2 = ExperimentResult(
            spec=spec,
            row_indices=(0, 1),
            is_anomaly=(False, True),
            scores=(0.1, 0.9),
        )
        assert canonical_experiment_hash(result1) == canonical_experiment_hash(result2)

    def test_result_hash_changes_with_scores(self):
        spec = ExperimentSpec(
            experiment_name="score-hash",
            detector_name="robust-zscore",
            features=("x",),
            dataset_id="ds001",
            input_hash="abc123",
            random_seed=42,
        )
        result1 = ExperimentResult(
            spec=spec,
            row_indices=(0, 1),
            is_anomaly=(False, True),
            scores=(0.1, 0.9),
        )
        result2 = ExperimentResult(
            spec=spec,
            row_indices=(0, 1),
            is_anomaly=(False, True),
            scores=(0.1, 0.95),
        )
        assert canonical_experiment_hash(result1) != canonical_experiment_hash(result2)

    def test_result_hash_changes_with_is_anomaly(self):
        spec = ExperimentSpec(
            experiment_name="anomaly-hash",
            detector_name="robust-zscore",
            features=("x",),
            dataset_id="ds001",
            input_hash="abc123",
            random_seed=42,
        )
        result1 = ExperimentResult(
            spec=spec,
            row_indices=(0, 1),
            is_anomaly=(False, True),
            scores=(0.1, 0.9),
        )
        result2 = ExperimentResult(
            spec=spec,
            row_indices=(0, 1),
            is_anomaly=(True, False),  # swapped
            scores=(0.1, 0.9),
        )
        assert canonical_experiment_hash(result1) != canonical_experiment_hash(result2)


class TestExperimentHasher:
    def test_hasher_records_hash_history(self):
        spec = ExperimentSpec(
            experiment_name="history-test",
            detector_name="robust-zscore",
            features=("x",),
            dataset_id="ds001",
            input_hash="abc123",
        )
        result = ExperimentResult(
            spec=spec,
            row_indices=(0,),
            is_anomaly=(True,),
            scores=(0.9,),
        )
        hasher = ExperimentHasher()
        h = hasher.compute_and_record(result)
        assert h in hasher.hash_history
        assert hasher.hash_history[h] == "experiment"

    def test_hasher_detects_duplicate(self):
        spec = ExperimentSpec(
            experiment_name="dup-test",
            detector_name="robust-zscore",
            features=("x",),
            dataset_id="ds001",
            input_hash="abc123",
        )
        result = ExperimentResult(
            spec=spec,
            row_indices=(0, 1),
            is_anomaly=(False, True),
            scores=(0.1, 0.9),
        )
        hasher = ExperimentHasher()
        h1 = hasher.compute_and_record(result)
        h2 = hasher.compute_and_record(result)
        assert h1 == h2
        assert len(hasher.hash_history) == 1  # duplicate not re-added


class TestProvenanceRecord:
    def test_provenance_collects_environment(self):
        env = collect_environment()
        assert "python_version" in env
        assert "dependencies" in env
        assert "platform" in env

    def test_provenance_creates_record(self):
        spec = ExperimentSpec(
            experiment_name="prov-test",
            detector_name="robust-zscore",
            features=("x",),
            dataset_id="ds001",
            input_hash="abc123",
            random_seed=42,
        )
        result = ExperimentResult(
            spec=spec,
            row_indices=(0, 1),
            is_anomaly=(False, True),
            scores=(0.1, 0.9),
        )
        prov = ProvenanceRecord.from_result(result)
        assert prov.experiment_hash is not None
        assert prov.environment is not None
        assert prov.spec_snapshot is not None
        assert prov.experiment_hash == canonical_experiment_hash(result)

    def test_provenance_to_dict_roundtrip(self):
        spec = ExperimentSpec(
            experiment_name="prov-dict",
            detector_name="isolation-forest",
            features=("x", "y"),
            dataset_id="ds002",
            input_hash="hashxyz",
        )
        result = ExperimentResult(
            spec=spec,
            row_indices=(0,),
            is_anomaly=(True,),
            scores=(0.8,),
        )
        prov = ProvenanceRecord.from_result(result)
        d = prov.to_dict()
        encoded = json.dumps(d, allow_nan=False)
        decoded = json.loads(encoded)
        assert decoded["experiment_hash"] == prov.experiment_hash
        assert decoded["spec_snapshot"]["experiment_name"] == "prov-dict"


class TestExperimentReplay:
    def test_replay_restores_spec_and_result(self):
        spec = ExperimentSpec(
            experiment_name="replay-test",
            detector_name="robust-zscore",
            features=("x", "y"),
            dataset_id="ds001",
            input_hash="abc123",
            random_seed=42,
            parameters={"threshold": 3.0},
        )
        result = ExperimentResult(
            spec=spec,
            row_indices=(0, 5, 10),
            is_anomaly=(False, True, False),
            scores=(0.1, 0.95, 0.3),
        )
        prov = ProvenanceRecord.from_result(result)
        replay = ExperimentReplay(prov)
        restored_spec, restored_result = replay.restore()
        assert restored_spec.experiment_name == spec.experiment_name
        assert restored_spec.detector_name == spec.detector_name
        assert restored_spec.features == spec.features
        assert restored_result.row_indices == result.row_indices
        assert restored_result.is_anomaly == result.is_anomaly

    def test_replay_verifies_hash(self):
        spec = ExperimentSpec(
            experiment_name="verify-hash",
            detector_name="robust-zscore",
            features=("x",),
            dataset_id="ds001",
            input_hash="abc123",
        )
        result = ExperimentResult(
            spec=spec,
            row_indices=(0,),
            is_anomaly=(True,),
            scores=(0.9,),
        )
        prov = ProvenanceRecord.from_result(result)
        replay = ExperimentReplay(prov)
        assert replay.verify() is True

    def test_replay_fails_on_tampered_spec(self):
        spec = ExperimentSpec(
            experiment_name="tamper-test",
            detector_name="robust-zscore",
            features=("x",),
            dataset_id="ds001",
            input_hash="abc123",
        )
        result = ExperimentResult(
            spec=spec,
            row_indices=(0,),
            is_anomaly=(True,),
            scores=(0.9,),
        )
        prov = ProvenanceRecord.from_result(result)
        # Tamper with provenance
        prov.spec_snapshot["experiment_name"] = "tampered-name"
        replay = ExperimentReplay(prov)
        assert replay.verify() is False

    def test_replay_fails_on_tampered_scores(self):
        spec = ExperimentSpec(
            experiment_name="score-tamper",
            detector_name="robust-zscore",
            features=("x",),
            dataset_id="ds001",
            input_hash="abc123",
        )
        result = ExperimentResult(
            spec=spec,
            row_indices=(0,),
            is_anomaly=(True,),
            scores=(0.9,),
        )
        prov = ProvenanceRecord.from_result(result)
        # Tamper with scores
        prov.result_snapshot["scores"] = [0.99]
        replay = ExperimentReplay(prov)
        assert replay.verify() is False
