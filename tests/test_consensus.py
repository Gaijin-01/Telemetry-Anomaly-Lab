"""Tests for consensus.py — Detector Consensus Without False Certainty."""

import numpy as np
import pytest

from anomaly_lab.consensus import (
    ConsensusResult,
    _get_detector_family,
    _pairwise_agreement,
    _to_percentile_scores,
    compute_consensus,
)
from anomaly_lab.results import DetectionResult

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────


def make_detection_result(
    detector: str,
    row_indices: list,
    is_anomaly: list[bool],
    scores: list[float],
) -> DetectionResult:
    return DetectionResult(
        detector=detector,
        row_indices=tuple(row_indices),
        is_anomaly=tuple(is_anomaly),
        features=("x",),
        constant_features=(),
        parameters={},
        scores=tuple(scores),
    )


# ─────────────────────────────────────────────────────────────────────────────
# _to_percentile_scores
# ─────────────────────────────────────────────────────────────────────────────


class TestToPercentileScores:
    def test_empty_returns_empty(self):
        result = _to_percentile_scores(np.array([]))
        assert result.size == 0

    def test_uniform_scores_all_same_percentile(self):
        scores = np.array([0.5, 0.5, 0.5])
        result = _to_percentile_scores(scores)
        assert result.shape == (3,)
        # All same score → all get the same percentile (0 or 100 depending on implementation)
        assert result[0] == result[1] == result[2]

    def test_scores_are_percentile_ranked(self):
        scores = np.array([0.1, 0.5, 0.9])
        result = _to_percentile_scores(scores)
        assert result[0] < result[1] < result[2]
        assert 0.0 <= result[0] < result[1] < result[2] <= 100.0


# ─────────────────────────────────────────────────────────────────────────────
# _pairwise_agreement
# ─────────────────────────────────────────────────────────────────────────────


class TestPairwiseAgreement:
    def test_perfect_agreement(self):
        a = np.array([0, 0, 1, 1, 0])
        b = np.array([0, 0, 1, 1, 0])
        assert _pairwise_agreement(a, b) == 1.0

    def test_complete_disagreement(self):
        a = np.array([0, 0, 0, 0])
        b = np.array([1, 1, 1, 1])
        assert _pairwise_agreement(a, b) == 0.0

    def test_partial_agreement(self):
        a = np.array([0, 0, 1, 1, 0])
        b = np.array([0, 0, 0, 1, 1])
        # positions 0,1 match; position 3 matches; positions 2,4 differ
        # 3/5 agree
        assert _pairwise_agreement(a, b) == pytest.approx(0.6)


# ─────────────────────────────────────────────────────────────────────────────
# _get_detector_family
# ─────────────────────────────────────────────────────────────────────────────


class TestGetDetectorFamily:
    def test_isolation_forest(self):
        assert _get_detector_family("isolation-forest") == "tree_ensemble"
        assert _get_detector_family("isolation_forest") == "tree_ensemble"

    def test_robust_zscore(self):
        assert _get_detector_family("robust-zscore") == "statistical"
        assert _get_detector_family("robust_zscore") == "statistical"

    def test_dbscan(self):
        assert _get_detector_family("dbscan") == "density"

    def test_unknown(self):
        assert _get_detector_family("unknown-detector") == "unknown"


# ─────────────────────────────────────────────────────────────────────────────
# compute_consensus basic
# ─────────────────────────────────────────────────────────────────────────────


class TestComputeConsensusBasics:
    def test_empty_results_raises(self):
        with pytest.raises(ValueError, match="non-empty"):
            compute_consensus([])

    def test_mismatched_row_counts_raises(self):
        r1 = make_detection_result("det1", [0, 1, 2], [False, False, False], [0.1, 0.2, 0.3])
        r2 = make_detection_result("det2", [0, 1], [False, False], [0.1, 0.2])
        with pytest.raises(ValueError, match="same number of rows"):
            compute_consensus([r1, r2])

    def test_missing_scores_raises(self):
        r1 = DetectionResult(
            detector="det1",
            row_indices=(0, 1, 2),
            is_anomaly=(False, False, False),
            features=("x",),
            constant_features=(),
            parameters={},
            scores=None,  # type: ignore
        )
        with pytest.raises(ValueError, match="no scores"):
            compute_consensus([r1])


# ─────────────────────────────────────────────────────────────────────────────
# compute_consensus result fields
# ─────────────────────────────────────────────────────────────────────────────


class TestComputeConsensusResult:
    @pytest.fixture
    def two_detectors(self):
        """Two detectors with different anomaly patterns."""
        r1 = make_detection_result(
            "isolation-forest",
            [0, 1, 2, 3, 4],
            [False, False, True, True, False],
            [0.1, 0.2, 0.8, 0.9, 0.3],
        )
        r2 = make_detection_result(
            "robust-zscore",
            [0, 1, 2, 3, 4],
            [False, True, True, False, False],
            [0.05, 0.7, 0.85, 0.4, 0.15],
        )
        return [r1, r2]

    def test_n_detectors(self, two_detectors):
        result = compute_consensus(two_detectors)
        assert result.n_detectors == 2

    def test_consensus_count_per_row(self, two_detectors):
        result = compute_consensus(two_detectors)
        # Row 0: no anomalies from either -> 0
        # Row 1: det2 only -> 1
        # Row 2: both -> 2
        # Row 3: det1 only -> 1
        # Row 4: neither -> 0
        assert result.consensus_count == [0, 1, 2, 1, 0]

    def test_agreement_matrix(self, two_detectors):
        result = compute_consensus(two_detectors)
        # Row 2: both flag -> [1, 1]
        assert result.agreement_matrix[2] == [1, 1]

    def test_anomaly_score_percentiles(self, two_detectors):
        result = compute_consensus(two_detectors)
        assert len(result.anomaly_score_percentiles) == 5
        # Row 2 has highest combined percentile
        # Values should be between 0 and 100
        for p in result.anomaly_score_percentiles:
            assert 0.0 <= p <= 100.0

    def test_pairwise_agreements(self, two_detectors):
        result = compute_consensus(two_detectors)
        pair_key = "isolation-forest|robust-zscore"
        assert pair_key in result.pairwise_agreements
        # They agree on 3/5 rows: rows 0, 2, 4
        assert result.pairwise_agreements[pair_key] == pytest.approx(0.6)

    def test_rank_correlations(self, two_detectors):
        result = compute_consensus(two_detectors)
        pair_key = "isolation-forest|robust-zscore"
        assert pair_key in result.rank_correlations
        assert -1.0 <= result.rank_correlations[pair_key] <= 1.0

    def test_detector_families(self, two_detectors):
        result = compute_consensus(two_detectors)
        assert result.detector_families["isolation-forest"] == "tree_ensemble"
        assert result.detector_families["robust-zscore"] == "statistical"

    def test_dissenting_detectors(self, two_detectors):
        result = compute_consensus(two_detectors)
        # Row 1: only det2 flags (det1 disagrees)
        assert "isolation-forest" in result.dissenting_detectors[1]
        # Row 2: both flag -> no dissent
        assert result.dissenting_detectors[2] == []

    def test_n_anomalies_total(self, two_detectors):
        result = compute_consensus(two_detectors)
        # Sum of all consensus counts
        assert result.n_anomalies_total == sum(result.consensus_count)

    def test_to_dict(self, two_detectors):
        result = compute_consensus(two_detectors)
        d = result.to_dict()
        assert isinstance(d, dict)
        assert d["n_detectors"] == 2
        assert d["schema_version"] == "0.4"
        assert d["evidence_class"] == "CONSENSUS_SUPPORTING_EVIDENCE"
        assert "_caveat" in d
        assert "not ground truth" in d["_caveat"]


# ─────────────────────────────────────────────────────────────────────────────
# compute_consensus with score_percentile=False
# ─────────────────────────────────────────────────────────────────────────────


class TestComputeConsensusRawScores:
    def test_raw_score_mode(self):
        r1 = make_detection_result("det1", [0, 1, 2], [False, False, True], [0.0, 0.5, 1.0])
        r2 = make_detection_result("det2", [0, 1, 2], [False, True, False], [0.0, 1.0, 0.5])
        result = compute_consensus([r1, r2], score_percentile=False)
        # Raw scores averaged: row 0 -> 0.0, row 1 -> 0.75, row 2 -> 0.75
        assert len(result.anomaly_score_percentiles) == 3


# ─────────────────────────────────────────────────────────────────────────────
# ConsensusResult frozen dataclass
# ─────────────────────────────────────────────────────────────────────────────


class TestConsensusResultFrozen:
    def test_result_is_frozen(self):
        result = ConsensusResult(
            n_detectors=2,
            n_anomalies_total=3,
            agreement_matrix=[[0, 0], [0, 1]],
            anomaly_score_percentiles=[0.0, 50.0, 100.0],
            pairwise_agreements={"a|b": 0.5},
            rank_correlations={"a|b": 0.3},
            consensus_count=[0, 1, 2],
            dissenting_detectors=[["a"], []],
            detector_families={"a": "tree_ensemble"},
        )
        with pytest.raises(AttributeError):
            result.n_detectors = 3  # type: ignore

    def test_evidence_class(self):
        result = ConsensusResult(
            n_detectors=2,
            n_anomalies_total=0,
            agreement_matrix=[],
            anomaly_score_percentiles=[],
            pairwise_agreements={},
            rank_correlations={},
            consensus_count=[],
            dissenting_detectors=[],
            detector_families={},
        )
        assert result.evidence_class == "CONSENSUS_SUPPORTING_EVIDENCE"
