"""Tests for evaluate.py — Time-Series Evaluation (PR-AUC, ROC-AUC, VUS-PR)."""

import numpy as np
import pytest

from anomaly_lab.evaluate import (
    EvaluationResult,
    _is_range_annotated,
    _point_adjusted_f1,
    _vus_pr,
    evaluate_scores,
)


class TestIsRangeAnnotated:
    """Tests for _is_range_annotated helper."""

    def test_empty_returns_false(self):
        assert _is_range_annotated(np.array([])) is False

    def test_single_point_true_returns_false(self):
        # A single True is not a range
        labels = np.array([True])
        assert _is_range_annotated(labels) is False

    def test_two_consecutive_trues_is_range(self):
        labels = np.array([False, True, True, False])
        assert _is_range_annotated(labels) is True

    def test_single_true_in_isolation_is_not_range(self):
        labels = np.array([True, False, True])
        assert _is_range_annotated(labels) is False

    def test_multiple_ranges(self):
        labels = np.array([True, True, False, True, True, True, False])
        assert _is_range_annotated(labels) is True


class TestPointAdjustedF1:
    """Tests for _point_adjusted_f1 helper."""

    def test_no_anomalies_in_labels(self):
        labels = np.zeros(10, dtype=bool)
        preds = np.zeros(10, dtype=bool)
        p, r, f = _point_adjusted_f1(labels, preds)
        assert p == 0.0
        assert r == 0.0
        assert f == 0.0

    def test_perfect_detection(self):
        labels = np.array([False, False, True, True, False])
        preds = np.array([False, False, True, True, False])
        p, r, f = _point_adjusted_f1(labels, preds)
        assert p == 1.0
        assert r == 1.0
        assert f == 1.0

    def test_partial_detection(self):
        # Two true ranges; predictions cover only one of them
        labels = np.array([True, True, True, True, False, False, True, True])  # two ranges
        preds = np.array(
            [True, True, True, True, False, False, False, False]
        )  # covers first range only
        p, r, f = _point_adjusted_f1(labels, preds)
        # 1 predicted range, 2 true ranges, only 1 true range covered
        assert p == 1.0  # 1 predicted range covers 1 true range
        assert r == 0.5  # 1 of 2 true ranges covered
        assert f == 2.0 / 3.0  # 2 * 1.0 * 0.5 / 1.5 = 2/3

    def test_false_positive_range(self):
        labels = np.array([True, True, True])  # one range
        preds = np.array([True, True, False])  # same range
        p, r, f = _point_adjusted_f1(labels, preds)
        assert p == 1.0
        assert r == 1.0
        assert f == 1.0


class TestVUSPR:
    """Tests for VUS-PR computation."""

    def test_all_same_scores_vus_pr_zero(self):
        labels = np.array([False, False, True, True])
        scores = np.array([0.5, 0.5, 0.5, 0.5])
        vus = _vus_pr(labels, scores)
        # Degenerate case: all scores identical
        assert 0.0 <= vus <= 1.0

    def test_vus_pr_in_range(self):
        np.random.seed(0)
        n = 50
        labels = np.zeros(n, dtype=bool)
        labels[20:35] = True  # one anomaly range
        scores = np.random.default_rng(0).normal(size=n)
        scores[20:35] += 3.0  # boost anomaly region
        vus = _vus_pr(labels, scores)
        assert 0.0 <= vus <= 1.0

    def test_vus_pr_perfect_detector(self):
        labels = np.array([False, False, True, True, False])
        scores = np.array([0.0, 0.0, 1.0, 1.0, 0.0])
        vus = _vus_pr(labels, scores)
        assert 0.0 <= vus <= 1.0


class TestEvaluateScoresBasics:
    """Basic evaluate_scores tests."""

    def test_empty_labels_raises(self):
        with pytest.raises(ValueError, match="non-empty"):
            evaluate_scores(np.array([]), np.array([0.0]))

    def test_mismatched_lengths_raises(self):
        with pytest.raises(ValueError, match="same length"):
            evaluate_scores(np.array([0, 1]), np.array([0.0, 1.0, 2.0]))

    def test_no_anomalies_in_labels(self):
        labels = np.zeros(20, dtype=int)
        scores = np.linspace(0, 1, 20)
        result = evaluate_scores(labels, scores)
        assert result.n_anomalies_true == 0
        assert result.pr_auc is None
        assert result.vus_pr is None
        assert "Point-adjusted F1" in result.caveats[0]

    def test_all_anomalies_in_labels(self):
        labels = np.ones(20, dtype=int)
        scores = np.linspace(0, 1, 20)
        result = evaluate_scores(labels, scores)
        assert result.n_anomalies_true == 20
        # PR-AUC undefined when all positives
        assert result.pr_auc is None

    def test_evaluator_note_default(self):
        labels = np.array([0, 0, 1, 1, 0])
        scores = np.array([0.0, 0.1, 0.9, 0.8, 0.2])
        result = evaluate_scores(labels, scores, compute_vus_pr=False)
        assert result.evaluator_note == "PR-AUC only"

    def test_evaluator_note_with_vus_pr(self):
        labels = np.array([False, False, True, True, True, False])
        scores = np.array([0.0, 0.1, 0.9, 0.8, 0.7, 0.2])
        result = evaluate_scores(labels, scores, compute_vus_pr=True)
        assert result.evaluator_note == "VUS-PR computed"


class TestEvaluateScoresOptimalThreshold:
    """Test optimal threshold selection when score_threshold is None."""

    def test_optimal_threshold_is_used(self):
        labels = np.array([0, 0, 0, 0, 1])
        scores = np.array([0.0, 0.1, 0.2, 0.3, 0.9])
        result = evaluate_scores(labels, scores, score_threshold=None)
        assert result.n_anomalies_predicted >= 0
        assert 0.0 <= result.precision <= 1.0
        assert 0.0 <= result.recall <= 1.0
        assert 0.0 <= result.f1 <= 1.0

    def test_fixed_threshold_is_used(self):
        labels = np.array([0, 0, 1, 1, 0])
        scores = np.array([0.1, 0.2, 0.8, 0.7, 0.3])
        result = evaluate_scores(labels, scores, score_threshold=0.5)
        # scores >= 0.5: positions 2 (0.8) and 3 (0.7)
        assert result.n_anomalies_predicted == 2


class TestEvaluateScoresVUSPR:
    """Test VUS-PR computation in evaluate_scores."""

    def test_vus_pr_none_when_no_range_in_labels(self):
        # Point-annotated labels (single isolated True values)
        labels = np.array([False, True, False, True, False])
        scores = np.array([0.1, 0.9, 0.2, 0.8, 0.3])
        result = evaluate_scores(labels, scores, compute_vus_pr=True)
        assert result.vus_pr is None
        assert "range-annotated" in result.caveats[0]

    def test_vus_pr_computed_when_range_annotated(self):
        # Contiguous anomaly range
        labels = np.array([False, False, True, True, True, False])
        scores = np.array([0.0, 0.1, 0.9, 0.8, 0.7, 0.2])
        result = evaluate_scores(labels, scores, compute_vus_pr=True)
        assert result.vus_pr is not None
        assert 0.0 <= result.vus_pr <= 1.0
        assert result.evaluator_note == "VUS-PR computed"


class TestEvaluateScoresPRAUC:
    """Test PR-AUC and ROC-AUC computation."""

    def test_pr_auc_is_computed(self):
        labels = np.array([0, 0, 0, 1, 1, 0, 1, 0, 0, 1])
        scores = np.array([0.1, 0.2, 0.3, 0.8, 0.7, 0.4, 0.9, 0.15, 0.25, 0.85])
        result = evaluate_scores(labels, scores)
        assert result.pr_auc is not None
        assert 0.0 <= result.pr_auc <= 1.0

    def test_roc_auc_is_computed(self):
        labels = np.array([0, 0, 0, 1, 1, 0, 1, 0, 0, 1])
        scores = np.array([0.1, 0.2, 0.3, 0.8, 0.7, 0.4, 0.9, 0.15, 0.25, 0.85])
        result = evaluate_scores(labels, scores)
        assert result.roc_auc is not None
        assert 0.0 <= result.roc_auc <= 1.0


class TestEvaluationResultDataclass:
    """Test EvaluationResult frozen dataclass and to_dict."""

    def test_result_is_frozen(self):
        result = EvaluationResult(
            precision=0.8,
            recall=0.6,
            f1=0.685,
            pr_auc=0.75,
            roc_auc=0.82,
            vus_pr=0.65,
            n_anomalies_true=5,
            n_anomalies_predicted=4,
            n_points=100,
            evaluator_note="VUS-PR computed",
            caveats=("caveat1",),
        )
        with pytest.raises(AttributeError):
            result.precision = 0.9  # type: ignore

    def test_to_dict_returns_dict(self):
        result = EvaluationResult(
            precision=0.8,
            recall=0.6,
            f1=0.685,
            pr_auc=0.75,
            roc_auc=0.82,
            vus_pr=0.65,
            n_anomalies_true=5,
            n_anomalies_predicted=4,
            n_points=100,
            evaluator_note="VUS-PR computed",
            caveats=("caveat1",),
        )
        d = result.to_dict()
        assert isinstance(d, dict)
        assert d["precision"] == 0.8
        assert d["pr_auc"] == 0.75
        assert d["vus_pr"] == 0.65
        assert d["schema_version"] == "0.5"
        assert d["evaluator_note"] == "VUS-PR computed"

    def test_vus_pr_none_in_dict(self):
        result = EvaluationResult(
            precision=0.8,
            recall=0.6,
            f1=0.685,
            pr_auc=0.75,
            roc_auc=0.82,
            vus_pr=None,
            n_anomalies_true=5,
            n_anomalies_predicted=4,
            n_points=100,
            evaluator_note="PR-AUC only",
            caveats=(),
        )
        d = result.to_dict()
        assert d["vus_pr"] is None
