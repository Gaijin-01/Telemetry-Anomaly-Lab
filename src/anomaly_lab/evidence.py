"""Evidence taxonomy: semantic labels for anomaly-detection and comparison results.

An anomaly detector produces statistical evidence, not a diagnosis.
This module translates raw DetectionResult / ComparisonResult into EvidenceRecord
instances that carry an evidence_class label and clear assumptions / limitations.

Evidence classes NEVER encode causal claims, root-cause diagnoses, or regression
verdicts — those require additional domain knowledge outside the scope of any
statistical detector.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Any

from anomaly_lab.results import DetectionResult


class EvidenceClass(Enum):
    POINT_ANOMALY = "POINT_ANOMALY"
    LOCAL_OUTLIER = "LOCAL_OUTLIER"
    GLOBAL_OUTLIER = "GLOBAL_OUTLIER"
    DENSITY_NOISE = "DENSITY_NOISE"
    REGIME_SHIFT = "REGIME_SHIFT"
    DISTRIBUTION_SHIFT = "DISTRIBUTION_SHIFT"
    VARIANCE_COLLAPSE = "VARIANCE_COLLAPSE"
    COMPARISON_DIFFERENCE = "COMPARISON_DIFFERENCE"
    COMPARISON_EQUIVALENT = "COMPARISON_EQUIVALENT"
    INCONCLUSIVE = "INCONCLUSIVE"
    NOT_APPLICABLE = "NOT_APPLICABLE"


@dataclass(frozen=True, kw_only=True)
class EvidenceRecord:
    """Structured evidence produced by the anomaly-lab reasoning layer."""

    evidence_class: str
    confidence: float | None  # 0.0-1.0, None if not applicable
    detector: str | None
    score: float | None
    threshold: float | None
    supporting_metrics: dict[str, float]
    assumptions: tuple[str, ...]
    limitations: tuple[str, ...]
    interpretation: str  # human-readable; distinguishes from regression/diagnosis

    def to_dict(self) -> dict[str, Any]:
        return {
            "evidence_class": self.evidence_class,
            "confidence": self.confidence,
            "detector": self.detector,
            "score": self.score,
            "threshold": self.threshold,
            "supporting_metrics": self.supporting_metrics,
            "assumptions": list(self.assumptions),
            "limitations": list(self.limitations),
            "interpretation": self.interpretation,
        }


def classify_detection_evidence(result: DetectionResult) -> EvidenceRecord:
    """Classify a DetectionResult into an EvidenceRecord.

    Rules
    -----
    - DBSCAN noise label (-1)  -> DENSITY_NOISE
    - IsolationForest score > threshold -> POINT_ANOMALY
    - Robust Z-score > threshold
        - if most other points are also flagged -> GLOBAL_OUTLIER (mass anomaly)
        - otherwise -> LOCAL_OUTLIER
    - Single extreme outlier -> POINT_ANOMALY
    - Multiple detectors agreeing -> higher confidence, still NOT ground truth

    The returned evidence_class is a statistical observation, never a claim of
    regression, root cause, or causal relationship.
    """
    detector = result.detector
    scores = result.scores
    is_anomaly = result.is_anomaly
    anomaly_count = sum(bool(a) for a in is_anomaly)
    total = len(is_anomaly)

    # ------------------------------------------------------------------
    # DBSCAN density noise
    # ------------------------------------------------------------------
    if detector == "dbscan":
        if result.clusters is None:
            return _inconclusive(detector, scores, "dbscan clusters not available")
        noise_count = sum(1 for c in result.clusters if c == -1)
        if noise_count == 0:
            return EvidenceRecord(
                evidence_class=EvidenceClass.NOT_APPLICABLE.value,
                confidence=1.0,
                detector=detector,
                score=None,
                threshold=None,
                supporting_metrics={},
                assumptions=("clustering_ran_successfully",),
                limitations=("cluster_ids_have_no_semantic_ordering",),
                interpretation="No density-based noise points detected.",
            )
        top_score = max((s for s in (scores or []) if s is not None), default=None)
        return EvidenceRecord(
            evidence_class=EvidenceClass.DENSITY_NOISE.value,
            confidence=0.8,
            detector=detector,
            score=top_score,
            threshold=None,
            supporting_metrics={
                "noise_fraction": noise_count / total,
                "total_points": total,
            },
            assumptions=(
                "eps_and_min_samples_were_appropriate_for_data_density",
                "StandardScaler_normalization_was_meaningful",
            ),
            limitations=(
                "DBSCAN_cluster_id_has_no_semantic_meaning",
                "noise_label_is_not_a_regression_or_root_cause_claim",
            ),
            interpretation=(
                f"{noise_count} point(s) labelled as density-based noise by DBSCAN. "
                "This is statistical evidence of low local density, not a diagnosis."
            ),
        )

    # ------------------------------------------------------------------
    # IsolationForest
    # ------------------------------------------------------------------
    if detector == "isolation-forest":
        if scores is None or anomaly_count == 0:
            return _inconclusive(detector, scores, "no anomalies detected")
        valid_scores = [(i, s) for i, s in enumerate(scores) if s is not None]
        top_idx, top_score = max(valid_scores, key=lambda pair: pair[1])
        threshold = _isolation_forest_threshold(result)
        confidence = min(0.95, 0.6 + 0.1 * anomaly_count)
        return EvidenceRecord(
            evidence_class=EvidenceClass.POINT_ANOMALY.value,
            confidence=confidence,
            detector=detector,
            score=top_score,
            threshold=threshold,
            supporting_metrics={
                "anomaly_count": float(anomaly_count),
                "total_points": float(total),
                "top_score": float(top_score),
            },
            assumptions=(
                "contamination_parameter_was_reasonable",
                "StandardScaler_normalization_was_meaningful",
                "anomalous_points_are_structurally_different_from_normals",
            ),
            limitations=(
                "anomaly_label_is_not_regression_or_root_cause",
                "high_score_may_reflect_both_anomaly_and_noise",
            ),
            interpretation=(
                f"IsolationForest flagged {anomaly_count} point(s) as structurally unusual. "
                "This is point-level statistical evidence only."
            ),
        )

    # ------------------------------------------------------------------
    # Robust Z-score
    # ------------------------------------------------------------------
    if detector == "robust-zscore":
        if scores is None or anomaly_count == 0:
            return _inconclusive(detector, scores, "no robust z-score exceedances")
        valid_scores = [(i, s) for i, s in enumerate(scores) if s is not None]
        top_idx, top_score = max(valid_scores, key=lambda pair: pair[1])
        threshold = result.parameters.get("threshold", 3.5)
        # Determine if local or global: flag fraction determines scope
        flag_fraction = anomaly_count / total
        if flag_fraction > 0.5:
            evidence_class = EvidenceClass.GLOBAL_OUTLIER.value
            interpretation = (
                f"Robust Z-score flagged {anomaly_count}/{total} points ({flag_fraction:.0%}). "
                "The majority of points are anomalous; this suggests a global distribution "
                "shift or that the threshold is miscalibrated for this data. "
                "This is NOT a comparative difference claim."
            )
        elif anomaly_count == 1:
            evidence_class = EvidenceClass.POINT_ANOMALY.value
            interpretation = (
                f"Single extreme outlier at index {top_idx} flagged by robust Z-score. "
                "Point-level statistical anomaly only; this is not a comparative difference claim."
            )
        else:
            evidence_class = EvidenceClass.LOCAL_OUTLIER.value
            interpretation = (
                f"{anomaly_count} points flagged by robust Z-score as local outliers "
                "(different from their immediate neighbours). "
                "This is not a comparative difference claim."
            )
        confidence = min(0.95, 0.7 + 0.05 * anomaly_count)
        return EvidenceRecord(
            evidence_class=evidence_class,
            confidence=confidence,
            detector=detector,
            score=top_score,
            threshold=threshold,
            supporting_metrics={
                "anomaly_count": anomaly_count,
                "total_points": total,
                "flag_fraction": flag_fraction,
                "top_score": top_score,
            },
            assumptions=(
                "data_follows_a_symmetric_heavy_tailed_distribution",
                "median_and_MAD_are_appropriate_central_and_scale_estimators",
            ),
            limitations=(
                "robust_zscore_is_not_a_regression_or_root_cause_claim",
                "z_score_exceedance_is_not_structural_equivalence",
            ),
            interpretation=interpretation,
        )

    # ------------------------------------------------------------------
    # Unknown detector — inconclusive
    # ------------------------------------------------------------------
    return _inconclusive(detector, scores, f"unknown detector {detector!r}")


def classify_comparison_evidence(result) -> EvidenceRecord:
    """Classify a ComparisonResult into an EvidenceRecord.

    Rules
    -----
    - REGRESSION        -> COMPARISON_DIFFERENCE
    - IMPROVEMENT        -> COMPARISON_DIFFERENCE
    - EQUIVALENT_WITHIN_MARGIN -> COMPARISON_EQUIVALENT
    - INSUFFICIENT_SAMPLE -> INCONCLUSIVE
    - NOT_COMPARABLE    -> NOT_APPLICABLE
    """
    status = result.status

    if status == "NOT_COMPARABLE":
        return EvidenceRecord(
            evidence_class=EvidenceClass.NOT_APPLICABLE.value,
            confidence=None,
            detector=None,
            score=None,
            threshold=None,
            supporting_metrics={},
            assumptions=(),
            limitations=("metadata_mismatch_precludes_comparison",),
            interpretation="Baseline and candidate are not declared comparable.",
        )

    if status == "INSUFFICIENT_SAMPLE":
        return EvidenceRecord(
            evidence_class=EvidenceClass.INCONCLUSIVE.value,
            confidence=0.0,
            detector=None,
            score=None,
            threshold=None,
            supporting_metrics={
                "warning_count": len(result.warnings),
            },
            assumptions=(),
            limitations=(
                "insufficient_samples_preclude_statistical_conclusion",
                "no_regression_or_equivalence_can_be_concluded",
            ),
            interpretation="Insufficient data to draw a statistical conclusion. "
            "Do NOT claim equivalence or difference.",
        )

    if status == "REGRESSION":
        # Map REGRESSION to COMPARISON_DIFFERENCE (not "regression" as a claim)
        return EvidenceRecord(
            evidence_class=EvidenceClass.COMPARISON_DIFFERENCE.value,
            confidence=0.9,
            detector=None,
            score=None,
            threshold=None,
            supporting_metrics={
                "delta": result.metrics[0].delta if result.metrics else None,
                "relative_change": result.metrics[0].relative_change if result.metrics else None,
            },
            assumptions=(
                "baseline_and_candidate_share_identical_run_metadata",
                "samples_are_independent_and_exchangeable",
            ),
            limitations=(
                "statistical_difference_is_not_a_causal_claim",
                "regression_label_is_metric_direction_only_not_root_cause",
                "causality_not_established",
            ),
            interpretation="A statistical difference was detected between baseline and candidate "
            "on at least one metric. This is comparative statistical evidence only; "
            "it does not establish causality or a root cause.",
        )

    # NO_REGRESSION — not the same as equivalence
    if status == "NO_REGRESSION":
        return EvidenceRecord(
            evidence_class=EvidenceClass.COMPARISON_DIFFERENCE.value,
            confidence=0.6,
            detector=None,
            score=None,
            threshold=None,
            supporting_metrics={
                "metric_count": len(result.metrics),
            },
            assumptions=(
                "baseline_and_candidate_share_identical_run_metadata",
                "samples_are_independent_and_exchangeable",
            ),
            limitations=(
                "no_significant_difference_does_not_imply_equivalence",
                "TOST_equivalence_test_required_for_equivalence_claim",
                "causality_not_established",
            ),
            interpretation="No regression detected. "
            "Non-significant difference is NOT evidence of equivalence; "
            "use TOST equivalence testing to claim equivalence.",
        )

    return _inconclusive(None, None, f"unknown comparison status {status!r}")


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _inconclusive(detector: str | None, scores, reason: str) -> EvidenceRecord:
    top_score = max((s for s in (scores or []) if s is not None), default=None)
    return EvidenceRecord(
        evidence_class=EvidenceClass.INCONCLUSIVE.value,
        confidence=0.0,
        detector=detector,
        score=top_score,
        threshold=None,
        supporting_metrics={},
        assumptions=(),
        limitations=(f"inconclusive: {reason}",),
        interpretation=f"Analysis was inconclusive: {reason}.",
    )


def _isolation_forest_threshold(result: DetectionResult) -> float:
    """Derive approximate threshold from contamination parameter."""
    cont = result.parameters.get("contamination", "auto")
    if cont == "auto":
        # sklearn uses ~0.5 * np.log(n) for auto; return sentinel
        return -0.5  # signals "auto / unknown"
    return float(cont)
