"""Regime / Anomaly / Regression Separation (Phase J).

Safe classification layer that NEVER emits causal or regression claims.

Priority order
-------------
1. Regime detected        -> REGIME_SHIFT
2. Comparison available   -> COMPARISON_DIFFERENCE / COMPARISON_EQUIVALENT
3. Detection available    -> POINT_ANOMALY / DENSITY_NOISE
4. Otherwise             -> INCONCLUSIVE

Key rules enforced
-----------------
- N=1 extreme score  -> POINT_ANOMALY (not regression)
- Multiple runs in different regimes -> DIFFERENT_OPERATING_REGIME (not equivalence)
- DBSCAN cluster cohesion -> density noise (not equivalence)
- Change-point detected -> REGIME_SHIFT (not root cause)
- TOST within margin -> COMPARISON_EQUIVALENT (not "same")
- Paired bootstrap CI beyond regression margin -> COMPARISON_DIFFERENCE (not "caused by X")
- Insufficient data -> INCONCLUSIVE (never "proven")
"""

from dataclasses import dataclass
from typing import Any

from anomaly_lab.evidence import EvidenceClass, EvidenceRecord, classify_comparison_evidence


@dataclass(frozen=True)
class RegimeAnalysisResult:
    """Result of a regime / operating-mode analysis.

    Attributes
    ----------
    regime_detected:
        True if more than one distinct operating regime was identified.
    regime_count:
        Number of distinct regimes (1 means uniform / no regime shift).
    regime_boundaries:
        List of row indices where a regime transition was detected.
    dominant_regime_fraction:
        Fraction of rows belonging to the most common regime.
    """

    regime_detected: bool
    regime_count: int
    regime_boundaries: tuple[int, ...]
    dominant_regime_fraction: float
    metadata: dict[str, Any] | None = None


def classify_observation(
    detection_results: list[Any] | None,
    comparison_result: Any,
    regime_result: RegimeAnalysisResult | None,
    context: dict | None = None,
) -> EvidenceRecord:
    """Safe classifier that never emits causal/regression/root-cause claims.

    Priority
    -------
    1. If regime detected -> REGIME_SHIFT
    2. If comparison available -> COMPARISON_DIFFERENCE / COMPARISON_EQUIVALENT
    3. If detection available -> POINT_ANOMALY / DENSITY_NOISE
    4. Otherwise -> INCONCLUSIVE

    Parameters
    ----------
    detection_results:
        List of DetectionResult from one or more detectors.
    comparison_result:
        ComparisonResult from compare_runs, or None.
    regime_result:
        RegimeAnalysisResult from regime analysis, or None.
    context:
        Optional dict with additional metadata (dataset semantics, etc.).

    Returns
    -------
    EvidenceRecord
        Structured evidence with an evidence_class label and caveats.
    """
    # ------------------------------------------------------------------
    # 1. Regime analysis — highest priority
    # ------------------------------------------------------------------
    if regime_result is not None and regime_result.regime_detected:
        boundary_count = len(regime_result.regime_boundaries)
        return EvidenceRecord(
            evidence_class=EvidenceClass.REGIME_SHIFT.value,
            confidence=0.85,
            detector=None,
            score=None,
            threshold=None,
            supporting_metrics={
                "regime_count": float(regime_result.regime_count),
                "boundary_count": float(boundary_count),
                "dominant_fraction": regime_result.dominant_regime_fraction,
            },
            assumptions=(
                "DBSCAN_clustering_and_eps_were_appropriate",
                "regimes_were_stable_within_each_segment",
            ),
            limitations=(
                "regime_shift_is_not_a_root_cause_claim",
                "regime_detection_does_not_identify_which_metric_caused_the_shift",
                "cluster_id_has_no_semantic_ordering",
            ),
            interpretation=(
                f"Regime analysis detected {regime_result.regime_count} distinct operating "
                f"regime(s) with {boundary_count} transition boundary(ies). "
                "This is statistical evidence of distributional change, not a root cause "
                "or causal event."
            ),
        )

    # ------------------------------------------------------------------
    # 2. Comparison result — second priority
    # ------------------------------------------------------------------
    if comparison_result is not None:
        return classify_comparison_evidence(comparison_result)

    # ------------------------------------------------------------------
    # 3. Detection results — third priority
    # ------------------------------------------------------------------
    if detection_results:
        # Aggregate evidence across multiple detectors if present
        if len(detection_results) == 1:
            from anomaly_lab.evidence import classify_detection_evidence

            return classify_detection_evidence(detection_results[0])

        # Multiple detectors — synthesize higher-confidence evidence
        from anomaly_lab.evidence import classify_detection_evidence

        individual = [classify_detection_evidence(r) for r in detection_results]
        anomaly_classes = {
            EvidenceClass.POINT_ANOMALY.value,
            EvidenceClass.LOCAL_OUTLIER.value,
            EvidenceClass.GLOBAL_OUTLIER.value,
            EvidenceClass.DENSITY_NOISE.value,
        }
        agreeing = [e for e in individual if e.evidence_class in anomaly_classes]
        if agreeing:
            # Average confidence across agreeing detectors, capped at 0.95
            avg_conf = sum(e.confidence or 0.0 for e in agreeing) / len(agreeing)
            combined_conf = min(0.95, avg_conf + 0.05)
            detectors = ", ".join(sorted(set(e.detector or "unknown" for e in agreeing)))
            return EvidenceRecord(
                evidence_class=EvidenceClass.POINT_ANOMALY.value,
                confidence=combined_conf,
                detector=detectors,
                score=None,
                threshold=None,
                supporting_metrics={
                    "agreeing_detector_count": float(len(agreeing)),
                    "total_detector_count": float(len(detection_results)),
                    "avg_confidence": avg_conf,
                },
                assumptions=(
                    "all_detectors_were_appropriate_for_data_characteristics",
                    "agreement_amplifies_confidence_but_not_certainty",
                ),
                limitations=(
                    "detector_consensus_is_not_ground_truth",
                    "all_output_classes_remain_statistical_evidence_only",
                    "anomaly_consensus_does_not_imply_regression_or_root_cause",
                ),
                interpretation=(
                    f"{len(agreeing)} detector(s) ({detectors}) agree on an anomaly pattern. "
                    "Detector consensus raises confidence but does not establish ground truth."
                ),
            )
        # No anomaly classes — return the most informative inconclusive
        most_confident = max(individual, key=lambda e: e.confidence or 0.0)
        return most_confident

    # ------------------------------------------------------------------
    # 4. Nothing available — inconclusive
    # ------------------------------------------------------------------
    return EvidenceRecord(
        evidence_class=EvidenceClass.INCONCLUSIVE.value,
        confidence=0.0,
        detector=None,
        score=None,
        threshold=None,
        supporting_metrics={},
        assumptions=(),
        limitations=("no_analysis_results_provided",),
        interpretation="Insufficient information to produce evidence. "
        "Do NOT claim anomaly, regression, or equivalence.",
    )
