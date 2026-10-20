"""Time-series anomaly evaluation: PR-AUC, ROC-AUC, VUS-PR, and point-adjusted F1."""

from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.metrics import auc, precision_recall_curve, roc_curve


@dataclass(frozen=True, kw_only=True)
class EvaluationResult:
    precision: float
    recall: float
    f1: float
    pr_auc: float | None
    roc_auc: float | None
    vus_pr: float | None  # only if ground truth has range annotations
    n_anomalies_true: int
    n_anomalies_predicted: int
    n_points: int
    evaluator_note: str  # "VUS-PR computed" | "PR-AUC only"
    caveats: tuple[str, ...]
    schema_version: str = "0.5"

    def to_dict(self) -> dict[str, Any]:
        return {
            "precision": float(self.precision),
            "recall": float(self.recall),
            "f1": float(self.f1),
            "pr_auc": float(self.pr_auc) if self.pr_auc is not None else None,
            "roc_auc": float(self.roc_auc) if self.roc_auc is not None else None,
            "vus_pr": float(self.vus_pr) if self.vus_pr is not None else None,
            "n_anomalies_true": self.n_anomalies_true,
            "n_anomalies_predicted": self.n_anomalies_predicted,
            "n_points": self.n_points,
            "evaluator_note": self.evaluator_note,
            "caveats": list(self.caveats),
            "schema_version": self.schema_version,
        }


def _is_range_annotated(labels: np.ndarray) -> bool:
    """Return True if labels contain contiguous anomaly ranges (not just point anomalies).

    A range-annotated label vector has at least one run of 2+ consecutive True values.
    """
    if labels.size == 0:
        return False
    # Find runs of consecutive Trues using diff transitions
    # diff[i] = +1 means False→True (run start), -1 means True→False (run end)
    padded = np.concatenate([[False], labels.astype(bool), [False]])
    diff = np.diff(padded.astype(int))  # must cast to int for proper +1/-1 values
    starts = np.where(diff == 1)[0]
    ends = np.where(diff == -1)[0]
    if starts.size == 0 or ends.size == 0:
        return False
    run_lengths = ends - starts
    return bool(np.any(run_lengths >= 2))


def _vus_pr(labels: np.ndarray, scores: np.ndarray) -> float:
    """Compute Volume Under the PR Surface (VUS-PR) for time-series anomaly detection.

    VUS-PR integrates precision-recall over the temporal volume, rewarding detectors
    that correctly identify the full extent of anomaly ranges.

    This implementation follows:
    - Normalize scores to [0, 1]
    - Sweep threshold from max(scores) down to min(scores)
    - At each threshold, compute PR for the time series
    - Integrate PR over the normalized threshold volume

    Parameters
    ----------
    labels : np.ndarray
        Binary ground truth (0/1 or bool), may contain contiguous anomaly ranges.
    scores : np.ndarray
        Anomaly scores (higher = more anomalous).

    Returns
    -------
    float
        VUS-PR score in [0, 1].
    """
    labels = np.asarray(labels, dtype=np.float64).flatten()
    scores = np.asarray(scores, dtype=np.float64).flatten()
    n = len(labels)

    if n == 0 or scores.size == 0:
        return 0.0

    # Normalize scores to [0, 1]
    score_min, score_max = scores.min(), scores.max()
    if score_max == score_min:
        # All scores identical — VUS-PR reduces to a single point
        pred = scores >= score_max
        tp = np.sum(pred & (labels == 1))
        fp = np.sum(pred & (labels == 0))
        fn = np.sum((~pred.astype(bool)) & (labels == 1))
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        return precision * recall  # degenerate case

    normalized = (scores - score_min) / (score_max - score_min)

    # Sort by threshold descending (highest score first)
    sorted_idx = np.argsort(-normalized)
    sorted_labels = labels[sorted_idx]
    sorted_scores_norm = -normalized[sorted_idx]  # negate so sweep goes 0→1 in threshold space

    # Unique threshold values in descending order
    unique_vals = np.unique(sorted_scores_norm)
    n_unique = len(unique_vals)
    unique_thresholds = np.linspace(0, 1, max(2, n_unique + 1))

    total_volume = 0.0

    for t in unique_thresholds:
        # Predictions at threshold t
        pred = sorted_scores_norm >= t

        tp = float(np.sum(pred & (sorted_labels == 1)))
        fp = float(np.sum(pred & (sorted_labels == 0)))
        fn = float(np.sum((~pred.astype(bool)) & (sorted_labels == 1)))

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

        # PR curve integration (rectangle method)
        delta_t = 1.0 / len(unique_thresholds)
        total_volume += precision * delta_t

    return float(total_volume)


def _point_adjusted_f1(labels: np.ndarray, predictions: np.ndarray) -> tuple[float, float, float]:
    """Compute point-adjusted precision, recall, and F1.

    Point-adjusted evaluation: a ground-truth anomaly range is "covered" if any
    predicted point falls within it. This is the standard evaluation for
    time-series anomaly detection.

    Parameters
    ----------
    labels : np.ndarray
        Binary ground truth (0/1 or bool).
    predictions : np.ndarray
        Binary predictions (0/1 or bool).

    Returns
    -------
    tuple[float, float, float]
        (precision, recall, f1) using point-adjusted evaluation.
    """
    labels = np.asarray(labels, dtype=bool).flatten()
    predictions = np.asarray(predictions, dtype=bool).flatten()

    # Find anomaly ranges in ground truth
    padded = np.concatenate([[False], labels, [False]])
    diff = np.diff(padded.astype(int))
    true_starts = np.where(diff == 1)[0]
    true_ends = np.where(diff == -1)[0]

    n_true_anomaly_ranges = len(true_starts)

    # Find predicted ranges
    pred_padded = np.concatenate([[False], predictions, [False]])
    pred_diff = np.diff(pred_padded.astype(int))
    pred_starts = np.where(pred_diff == 1)[0]
    n_predicted_ranges = len(pred_starts)

    # For each ground-truth range, check if any prediction falls within it
    covered_ranges = 0
    for s, e in zip(true_starts, true_ends, strict=True):
        if np.any(predictions[s:e]):
            covered_ranges += 1

    # Point-adjusted metrics
    true_positives_adjusted = covered_ranges

    precision = true_positives_adjusted / n_predicted_ranges if n_predicted_ranges > 0 else 0.0
    recall = covered_ranges / n_true_anomaly_ranges if n_true_anomaly_ranges > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return precision, recall, f1


def evaluate_scores(
    labels: np.ndarray,
    scores: np.ndarray,
    score_threshold: float | None = None,
    compute_vus_pr: bool = False,
) -> EvaluationResult:
    """Evaluate anomaly detection scores against ground truth.

    Parameters
    ----------
    labels : np.ndarray
        Binary ground truth (0/1 or bool). May be point-based or range-annotated.
    scores : np.ndarray
        Anomaly scores (higher = more anomalous).
    score_threshold : float | None
        Fixed threshold for converting scores to binary predictions.
        If None, threshold is chosen to maximize F1 score.
    compute_vus_pr : bool
        If True and labels contain contiguous anomaly ranges, compute VUS-PR.
        If labels are point-annotated only, vus_pr will be None.

    Returns
    -------
    EvaluationResult
        Precision, recall, F1, PR-AUC, ROC-AUC, and optionally VUS-PR.
    """
    labels = np.asarray(labels, dtype=np.float64).flatten()
    scores = np.asarray(scores, dtype=np.float64).flatten()

    if labels.size == 0:
        raise ValueError("labels must be non-empty")
    if scores.size != labels.size:
        raise ValueError("scores and labels must have the same length")

    n_anomalies_true = int(np.sum(labels.astype(bool)))
    n_points = len(labels)

    caveats: list[str] = []
    evaluator_note = "PR-AUC only"

    # Binary predictions
    if score_threshold is None:
        # Find optimal threshold that maximizes F1
        if n_anomalies_true == 0:
            # No anomalies in ground truth
            predictions = np.zeros_like(scores) > 0
            best_f1 = 0.0
            best_precision = 1.0
            best_recall = 0.0
            best_thresh = float(scores.min())
        elif n_anomalies_true == n_points:
            # All points are anomalies
            predictions = np.ones_like(scores) > 0
            best_f1 = 1.0
            best_precision = 1.0
            best_recall = 1.0
            best_thresh = float(scores.min())
        else:
            # Sweep thresholds
            unique_scores = np.unique(scores)
            best_f1 = -1.0
            best_precision = 0.0
            best_recall = 0.0
            best_thresh = float(scores.min())

            for thresh in unique_scores:
                preds = scores >= thresh
                prec, rec, f1 = _point_adjusted_f1(labels.astype(bool), preds)
                if f1 > best_f1:
                    best_f1 = f1
                    best_precision = prec
                    best_recall = rec
                    best_thresh = thresh

        predictions = scores >= best_thresh
        used_threshold = best_thresh
    else:
        predictions = scores >= score_threshold
        used_threshold = score_threshold
        best_precision, best_recall, best_f1 = _point_adjusted_f1(labels.astype(bool), predictions)

    n_anomalies_predicted = int(np.sum(predictions))

    # PR curve + PR-AUC
    pr_auc: float | None = None
    if n_anomalies_true > 0 and n_anomalies_true < n_points:
        precision_curve, recall_curve, pr_thresholds = precision_recall_curve(labels, scores)
        # Sort by recall ascending to compute AUC correctly
        idx = np.argsort(recall_curve)
        pr_auc = float(auc(recall_curve[idx], precision_curve[idx]))

    # ROC curve + ROC-AUC
    roc_auc: float | None = None
    if n_anomalies_true > 0 and n_anomalies_true < n_points:
        fpr_curve, tpr_curve, _ = roc_curve(labels, scores)
        roc_auc = float(auc(fpr_curve, tpr_curve))

    # VUS-PR
    vus_pr: float | None = None
    if compute_vus_pr:
        if _is_range_annotated(labels.astype(bool)):
            vus_pr = float(_vus_pr(labels.astype(bool), scores))
            evaluator_note = "VUS-PR computed"
            caveats.append("VUS-PR requires contiguous anomaly ranges in ground truth")
        else:
            caveats.append(
                "vus_pr=None: labels appear point-annotated, not range-annotated; "
                "VUS-PR requires contiguous anomaly ranges"
            )

    caveats.append(
        "Point-adjusted F1 is NOT a full evaluation; "
        "always review PR-AUC, ROC-AUC, and VUS-PR together"
    )
    caveats.append(
        f"Threshold={used_threshold:.4f} is F1-optimal on this test set; "
        "do not transfer to production without calibration"
    )

    return EvaluationResult(
        precision=float(best_precision),
        recall=float(best_recall),
        f1=float(best_f1),
        pr_auc=pr_auc,
        roc_auc=roc_auc,
        vus_pr=vus_pr,
        n_anomalies_true=n_anomalies_true,
        n_anomalies_predicted=n_anomalies_predicted,
        n_points=n_points,
        evaluator_note=evaluator_note,
        caveats=tuple(caveats),
    )
