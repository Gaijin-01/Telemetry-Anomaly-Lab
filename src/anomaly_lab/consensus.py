"""Detector consensus without false certainty."""

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.stats import spearmanr

from anomaly_lab.results import DetectionResult


@dataclass(frozen=True, kw_only=True)
class ConsensusResult:
    n_detectors: int
    n_anomalies_total: int
    agreement_matrix: list[list[int]]  # per-row: how many detectors flag each row
    anomaly_score_percentiles: list[float]  # per-row percentile rank across detectors
    pairwise_agreements: dict[str, float]  # detector-pair -> proportion agreement
    rank_correlations: dict[str, float]  # detector-pair -> Spearman rho
    consensus_count: list[int]  # per-row: how many detectors agree
    dissenting_detectors: list[list[str]]  # per-row: which detectors disagree
    detector_families: dict[str, str]  # detector -> family
    schema_version: str = "0.5"
    evidence_class: str = "CONSENSUS_SUPPORTING_EVIDENCE"

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_detectors": self.n_detectors,
            "n_anomalies_total": self.n_anomalies_total,
            "agreement_matrix": self.agreement_matrix,
            "anomaly_score_percentiles": [float(p) for p in self.anomaly_score_percentiles],
            "pairwise_agreements": self.pairwise_agreements,
            "rank_correlations": {k: float(v) for k, v in self.rank_correlations.items()},
            "consensus_count": self.consensus_count,
            "dissenting_detectors": self.dissenting_detectors,
            "detector_families": self.detector_families,
            "schema_version": self.schema_version,
            "evidence_class": self.evidence_class,
            "_caveat": (
                "consensus is statistical agreement, not ground truth; "
                "3 similar methods count as 1 independent source"
            ),
        }


# ─────────────────────────────────────────────────────────────────────────────
# Detector family classification
# ─────────────────────────────────────────────────────────────────────────────

_DETECTOR_FAMILY_MAP: dict[str, str] = {
    "isolation-forest": "tree_ensemble",
    "isolation_forest": "tree_ensemble",
    "robust-zscore": "statistical",
    "robust_zscore": "statistical",
    "dbscan": "density",
    "dbscan-detector": "density",
}


def _get_detector_family(detector_name: str) -> str:
    """Return the method family for a detector."""
    return _DETECTOR_FAMILY_MAP.get(detector_name, "unknown")


# ─────────────────────────────────────────────────────────────────────────────
# Percentile normalization
# ─────────────────────────────────────────────────────────────────────────────


def _to_percentile_scores(scores: np.ndarray) -> np.ndarray:
    """Convert raw scores to percentile ranks within [0, 100].

    Uses percentileofscore so ties get the same rank.
    """
    n = len(scores)
    if n == 0:
        return np.array([])
    percentiles = np.zeros(n, dtype=np.float64)
    sorted_scores = np.sort(scores)
    for i in range(n):
        # percentile rank: what fraction of values are <= this score
        percentiles[i] = float(np.searchsorted(sorted_scores, scores[i])) / n * 100.0
    return percentiles


# ─────────────────────────────────────────────────────────────────────────────
# Pairwise agreement
# ─────────────────────────────────────────────────────────────────────────────


def _pairwise_agreement(labels_a: np.ndarray, labels_b: np.ndarray) -> float:
    """Proportion of rows where both detectors agree on anomaly/no-anomaly."""
    agreements = np.sum(labels_a == labels_b)
    return float(agreements) / len(labels_a)


# ─────────────────────────────────────────────────────────────────────────────
# Main consensus function
# ─────────────────────────────────────────────────────────────────────────────


def compute_consensus(
    results: list[DetectionResult],
    score_percentile: bool = True,
) -> ConsensusResult:
    """Compute consensus statistics across multiple detector results.

    Parameters
    ----------
    results : list[DetectionResult]
        List of detector results for the same dataset.
    score_percentile : bool
        If True, normalize scores to percentile ranks across each detector's score
        distribution before computing per-row averages.

    Returns
    -------
    ConsensusResult
        Consensus statistics including per-row agreement counts, pairwise
        agreements, rank correlations, and dissenting detectors.

    Caveats
    -------
    - Consensus is statistical agreement, NOT ground truth.
    - 3 similar methods count as 1 independent source.
    - Families (tree_ensemble, density, statistical) are not independent.
    """
    if not results:
        raise ValueError("results list must be non-empty")

    n_detectors = len(results)
    n_rows = len(results[0].row_indices)

    # Validate all results have the same number of rows
    for r in results:
        if len(r.row_indices) != n_rows:
            raise ValueError(
                f"all results must have the same number of rows; "
                f"got {len(r.row_indices)} vs expected {n_rows}"
            )

    # Build score matrix and anomaly matrix
    _score_list: list[np.ndarray] = []
    _anomaly_list: list[np.ndarray] = []

    for r in results:
        if r.scores is None:
            raise ValueError(f"detector {r.detector} has no scores; consensus requires scores")
        _score_list.append(np.array(r.scores, dtype=np.float64))
        _anomaly_list.append(np.array(r.is_anomaly, dtype=int))

    score_matrix: np.ndarray = np.stack(_score_list, axis=0)  # shape: (n_detectors, n_rows)
    anomaly_matrix: np.ndarray = np.stack(_anomaly_list, axis=0)  # shape: (n_detectors, n_rows)

    # Per-row: how many detectors flag each row as anomaly
    agreement_matrix: list[list[int]] = []
    consensus_count: list[int] = []
    for row_idx in range(n_rows):
        row_anomalies = anomaly_matrix[:, row_idx]
        n_flag = int(np.sum(row_anomalies))
        agreement_matrix.append([int(v) for v in row_anomalies])
        consensus_count.append(n_flag)

    # Per-row percentile scores
    anomaly_score_percentiles: list[float] = []
    for row_idx in range(n_rows):
        if score_percentile:
            row_scores = score_matrix[:, row_idx]
            row_percentiles = _to_percentile_scores(row_scores)
            avg_percentile = float(np.mean(row_percentiles))
        else:
            avg_percentile = float(np.mean(score_matrix[:, row_idx]))
        anomaly_score_percentiles.append(avg_percentile)

    # Pairwise agreements and rank correlations
    pairwise_agreements: dict[str, float] = {}
    rank_correlations: dict[str, float] = {}
    detector_names = [r.detector for r in results]

    for i in range(n_detectors):
        for j in range(i + 1, n_detectors):
            pair_key = f"{detector_names[i]}|{detector_names[j]}"
            # Agreement
            pairwise_agreements[pair_key] = _pairwise_agreement(
                anomaly_matrix[i], anomaly_matrix[j]
            )
            # Spearman rank correlation on scores
            rho, _ = spearmanr(score_matrix[i], score_matrix[j])
            rank_correlations[pair_key] = float(rho) if np.isfinite(rho) else 0.0

    # Dissent: a detector's vote conflicts with the majority outcome.
    # - minority (n_flag < n_detectors/2): those who DID flag are dissenting
    # - tie (n_flag == n_detectors/2): ALL detectors are dissenting (no majority)
    # - majority (n_flag > n_detectors/2): those who did NOT flag are dissenting
    dissenting_detectors: list[list[str]] = []
    half = n_detectors // 2
    for row_idx in range(n_rows):
        row_anomalies = anomaly_matrix[:, row_idx]
        n_flag = int(np.sum(row_anomalies))
        dissent: list[str] = []
        for det_idx, flagged in enumerate(row_anomalies):
            if n_flag < half:
                # Minority: only those who flagged disagree with the majority
                if flagged:
                    dissent.append(detector_names[det_idx])
            elif n_flag == half:
                # Tie: everyone is in the minority
                dissent.append(detector_names[det_idx])
            else:
                # Majority: those who didn't flag disagree with the majority
                if not flagged:
                    dissent.append(detector_names[det_idx])
        dissenting_detectors.append(dissent)

    # Detector families
    detector_families = {r.detector: _get_detector_family(r.detector) for r in results}

    n_anomalies_total = sum(consensus_count)

    return ConsensusResult(
        n_detectors=n_detectors,
        n_anomalies_total=n_anomalies_total,
        agreement_matrix=agreement_matrix,
        anomaly_score_percentiles=anomaly_score_percentiles,
        pairwise_agreements=pairwise_agreements,
        rank_correlations=rank_correlations,
        consensus_count=consensus_count,
        dissenting_detectors=dissenting_detectors,
        detector_families=detector_families,
    )
