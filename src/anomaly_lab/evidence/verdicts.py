"""Verdicts: derive EvidenceVerdictResult from EvidenceRecord.

FAILED_EXECUTION_CANNOT_PASS invariant: any failed execution always yields CANNOT_PASS,
regardless of the evidence class or confidence. TOML-serialisable results; no LLM deps.
"""

from __future__ import annotations

from dataclasses import replace

from anomaly_lab.evidence.applicability import get_applicability
from anomaly_lab.evidence.taxonomy import (
    CausalityStatus,
    EvidenceApplicability,
    EvidenceClass,
    EvidenceRecord,
    EvidenceVerdict,
    EvidenceVerdictResult,
    VerdictDecision,
    VerdictReason,
)

# -----------------------------------------------------------------------
# Invariant: FAILED_EXECUTION_CANNOT_PASS
# -----------------------------------------------------------------------

# Sentinel string used to mark failed executions in DetectionResult metadata
_FAILED_EXECUTION_CANNOT_PASS = "failed_execution"


def _is_failed(record: EvidenceRecord) -> bool:
    """Return True if the record represents a failed execution.

    Checks:
        1. evidence_class is INCONCLUSIVE with a "failed" limitation
        2. A hypothetical `failed` flag in supporting_metrics
        3. applicability is explicitly False due to execution failure
    """
    # Check limitation string for execution failure signal
    for lim in record.limitations:
        if "failed" in lim.lower() or "error" in lim.lower():
            # Further require that confidence is 0 (strong failure signal)
            if record.confidence == 0.0:
                return True

    # Check supporting_metrics for explicit failure flag
    if record.supporting_metrics.get("execution_failed", False):
        return True

    # Not applicable due to failure
    app = get_applicability(record)
    if not app.applicable:
        for constraint in app.domain_constraints:
            if "execution_failed" in constraint or "failed" in constraint:
                return True

    return False


# -----------------------------------------------------------------------
# Detection verdict
# -----------------------------------------------------------------------


def compute_detection_verdict(
    record: EvidenceRecord,
    *,
    decision: VerdictDecision = VerdictDecision.ANOMALY_DETECTION,
) -> EvidenceVerdictResult:
    """Compute an EvidenceVerdictResult for a detection-side EvidenceRecord.

    Parameters
    ----------
    record
        EvidenceRecord with evidence_class set by classify_detection_evidence.
    decision
        VerdictDecision context (default ANOMALY_DETECTION).

    Returns
    -------
    EvidenceVerdictResult

    Invariant: FAILED_EXECUTION_CANNOT_PASS — a failed execution record
    always yields CANNOT_PASS regardless of evidence_class.
    """
    # FAILED_EXECUTION_CANNOT_PASS invariant
    if _is_failed(record):
        return EvidenceVerdictResult(
            verdict=EvidenceVerdict.CANNOT_PASS,
            reason=(
                "Execution failed; verdict is CANNOT_PASS regardless of evidence class. "
                "Causality has not been established."
            ),
            reason_code=VerdictReason.FAILED_EXECUTION,
            decision=decision,
            causality_status=CausalityStatus.NOT_ESTABLISHED,
        )

    # Causality is never established for detection-only evidence
    if record.causality_status == CausalityStatus.NOT_ESTABLISHED:
        if record.confidence is not None and record.confidence == 0.0:
            return EvidenceVerdictResult(
                verdict=EvidenceVerdict.INCONCLUSIVE,
                reason="Zero confidence; evidence is insufficient to reach a verdict.",
                reason_code=VerdictReason.INSUFFICIENT_EVIDENCE,
                decision=decision,
                causality_status=CausalityStatus.NOT_ESTABLISHED,
            )

    ec = record.evidence_class
    app = get_applicability(record) if record.applicability is None else record.applicability

    if not app.applicable:
        return EvidenceVerdictResult(
            verdict=EvidenceVerdict.CANNOT_PASS,
            reason=f"Evidence class {ec!r} is not applicable to {decision.value}.",
            reason_code=VerdictReason.NOT_APPLICABLE,
            decision=decision,
            causality_status=record.causality_status,
        )

    # Map evidence class to verdict
    if ec in {
        EvidenceClass.POINT_ANOMALY.value,
        EvidenceClass.LOCAL_OUTLIER.value,
        EvidenceClass.GLOBAL_OUTLIER.value,
        EvidenceClass.DENSITY_NOISE.value,
        EvidenceClass.REGIME_SHIFT.value,
        EvidenceClass.DISTRIBUTION_SHIFT.value,
        EvidenceClass.VARIANCE_COLLAPSE.value,
    }:
        # An anomaly was detected — PASS for anomaly detection decision
        return EvidenceVerdictResult(
            verdict=EvidenceVerdict.PASS,
            reason=(
                f"{ec} detected; statistical anomaly confirmed. "
                "This is evidence only — causality has not been established."
            ),
            reason_code=VerdictReason.ANOMALY_DETECTED,
            decision=decision,
            causality_status=record.causality_status,
        )

    if ec in {EvidenceClass.INCONCLUSIVE.value, EvidenceClass.INSUFFICIENT_EVIDENCE.value}:
        return EvidenceVerdictResult(
            verdict=EvidenceVerdict.INCONCLUSIVE,
            reason=f"Evidence class {ec!r} is inconclusive; cannot reach a verdict.",
            reason_code=VerdictReason.INSUFFICIENT_EVIDENCE,
            decision=decision,
            causality_status=record.causality_status,
        )

    # Fallback
    return EvidenceVerdictResult(
        verdict=EvidenceVerdict.INCONCLUSIVE,
        reason=f"Unknown evidence class {ec!r}; defaulting to inconclusive.",
        reason_code=VerdictReason.INSUFFICIENT_EVIDENCE,
        decision=decision,
        causality_status=record.causality_status,
    )


# -----------------------------------------------------------------------
# Comparison verdict
# -----------------------------------------------------------------------


def compute_comparison_verdict(
    record: EvidenceRecord,
    *,
    decision: VerdictDecision = VerdictDecision.REGRESSION_TEST,
) -> EvidenceVerdictResult:
    """Compute an EvidenceVerdictResult for a comparison-side EvidenceRecord.

    Parameters
    ----------
    record
        EvidenceRecord with evidence_class set by classify_comparison_evidence.
    decision
        VerdictDecision context (default REGRESSION_TEST).

    Returns
    -------
    EvidenceVerdictResult

    Invariant: FAILED_EXECUTION_CANNOT_PASS — a failed execution always yields
    CANNOT_PASS regardless of the evidence class or comparison status.
    """
    # FAILED_EXECUTION_CANNOT_PASS invariant
    if _is_failed(record):
        return EvidenceVerdictResult(
            verdict=EvidenceVerdict.CANNOT_PASS,
            reason=(
                "Execution failed; verdict is CANNOT_PASS regardless of comparison status. "
                "Causality has not been established."
            ),
            reason_code=VerdictReason.FAILED_EXECUTION,
            decision=decision,
            causality_status=CausalityStatus.NOT_ESTABLISHED,
        )

    ec = record.evidence_class
    app = get_applicability(record) if record.applicability is None else record.applicability

    if not app.applicable:
        return EvidenceVerdictResult(
            verdict=EvidenceVerdict.CANNOT_PASS,
            reason=(
                f"Evidence class {ec!r} is not applicable to {decision.value} "
                f"(constraints: {list(app.domain_constraints)})."
            ),
            reason_code=(
                VerdictReason.INCOMPARABLE
                if ec == EvidenceClass.INCOMPARABLE.value
                else VerdictReason.NOT_APPLICABLE
            ),
            decision=decision,
            causality_status=record.causality_status,
        )

    # Map comparison evidence class to verdict
    if ec == EvidenceClass.IMPROVEMENT.value:
        return EvidenceVerdictResult(
            verdict=EvidenceVerdict.PASS,
            reason=(
                "Improvement detected: candidate is significantly better on ≥1 metric. "
                "This is comparative statistical evidence only — "
                "causality has not been established."
            ),
            reason_code=VerdictReason.IMPROVEMENT_DETECTED,
            decision=decision,
            causality_status=record.causality_status,
        )

    if ec == EvidenceClass.REGRESSION.value:
        return EvidenceVerdictResult(
            verdict=EvidenceVerdict.FAIL,
            reason=(
                "Regression detected: candidate is significantly worse on ≥1 metric. "
                "This is comparative statistical evidence only — "
                "causality has not been established."
            ),
            reason_code=VerdictReason.REGRESSION_DETECTED,
            decision=decision,
            causality_status=record.causality_status,
        )

    if ec == EvidenceClass.NEUTRAL.value:
        return EvidenceVerdictResult(
            verdict=EvidenceVerdict.INCONCLUSIVE,
            reason=(
                "Neutral outcome: no significant difference in either direction. "
                "This is not evidence of equivalence."
            ),
            reason_code=VerdictReason.NEUTRAL_OUTCOME,
            decision=decision,
            causality_status=record.causality_status,
        )

    if ec == EvidenceClass.COMPARISON_DIFFERENCE.value:
        # Statistical difference without direction claim
        return EvidenceVerdictResult(
            verdict=EvidenceVerdict.INCONCLUSIVE,
            reason=(
                "A statistical difference was detected, but direction (improvement vs. "
                "regression) is not established. Further analysis required."
            ),
            reason_code=VerdictReason.NEUTRAL_OUTCOME,
            decision=decision,
            causality_status=record.causality_status,
        )

    if ec == EvidenceClass.COMPARISON_EQUIVALENT.value:
        return EvidenceVerdictResult(
            verdict=EvidenceVerdict.PASS,
            reason=(
                "Equivalence established within TOST margin. "
                "This is comparative statistical evidence only — "
                "causality has not been established."
            ),
            reason_code=VerdictReason.EQUIVALENCE_ESTABLISHED,
            decision=decision,
            causality_status=record.causality_status,
        )

    if ec in {
        EvidenceClass.INSUFFICIENT_EVIDENCE.value,
        EvidenceClass.INCOMPARABLE.value,
        EvidenceClass.NOT_APPLICABLE.value,
    }:
        return EvidenceVerdictResult(
            verdict=EvidenceVerdict.CANNOT_PASS,
            reason=(
                f"Evidence class {ec!r} precludes any passing verdict. "
                "Causality has not been established."
            ),
            reason_code=(
                VerdictReason.INSUFFICIENT_EVIDENCE
                if ec == EvidenceClass.INSUFFICIENT_EVIDENCE.value
                else VerdictReason.INCOMPARABLE
            ),
            decision=decision,
            causality_status=record.causality_status,
        )

    if ec == EvidenceClass.INCONCLUSIVE.value:
        return EvidenceVerdictResult(
            verdict=EvidenceVerdict.INCONCLUSIVE,
            reason="Inconclusive evidence; cannot reach a verdict.",
            reason_code=VerdictReason.INSUFFICIENT_EVIDENCE,
            decision=decision,
            causality_status=record.causality_status,
        )

    # Fallback
    return EvidenceVerdictResult(
        verdict=EvidenceVerdict.INCONCLUSIVE,
        reason=f"Unknown evidence class {ec!r}; defaulting to inconclusive.",
        reason_code=VerdictReason.INSUFFICIENT_EVIDENCE,
        decision=decision,
        causality_status=record.causality_status,
    )


# -----------------------------------------------------------------------
# Attach verdict to EvidenceRecord (pure helper, no side effects)
# -----------------------------------------------------------------------


def attach_verdict(
    record: EvidenceRecord,
    verdict: EvidenceVerdictResult,
) -> EvidenceRecord:
    """Return a new EvidenceRecord with ``verdict`` field set."""
    return replace(record, verdict=verdict)


def attach_applicability(
    record: EvidenceRecord,
    applicability: EvidenceApplicability,
) -> EvidenceRecord:
    """Return a new EvidenceRecord with ``applicability`` field set."""
    return replace(record, applicability=applicability)
