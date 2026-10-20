"""Comparability: determine whether two EvidenceRecords can be compared.

TOML-serialisable result dicts; no LLM dependencies.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from anomaly_lab.evidence.taxonomy import EvidenceClass, EvidenceRecord


@dataclass(frozen=True, kw_only=True)
class ComparabilityResult:
    """Result of a comparability check between two EvidenceRecords."""

    comparable: bool
    same_detector: bool
    same_semantics: bool
    reason: str  # empty when comparable

    def to_dict(self) -> dict[str, Any]:
        return {
            "comparable": self.comparable,
            "same_detector": self.same_detector,
            "same_semantics": self.same_semantics,
            "reason": self.reason,
        }


def check_comparability(
    rec_a: EvidenceRecord,
    rec_b: EvidenceRecord,
) -> ComparabilityResult:
    """Return a ComparabilityResult describing whether rec_a and rec_b are comparable.

    Comparability requires:
        1. Neither record has evidence_class INCOMPARABLE or NOT_APPLICABLE
        2. Both records share the same decision context (applicability.decision)
           OR at least one has no applicability set (fall back to class-level)
        3. Same detector family is preferred but not required (cross-detector
           comparison is allowed within the same evidence_class family)

    Returns a ComparabilityResult with ``comparable`` True/False and a reason.
    """
    # Rule 1: neither may be explicitly incomparable
    _incomparable_classes = {
        EvidenceClass.INCOMPARABLE.value,
        EvidenceClass.NOT_APPLICABLE.value,
    }
    for rec, name in ((rec_a, "rec_a"), (rec_b, "rec_b")):
        if rec.evidence_class in _incomparable_classes:
            return ComparabilityResult(
                comparable=False,
                same_detector=(rec_a.detector == rec_b.detector),
                same_semantics=_same_semantics(rec_a, rec_b),
                reason=f"{name} has evidence_class {rec.evidence_class!r} which is not comparable",
            )

    # Rule 2: decision context must align
    dec_a = _get_decision(rec_a)
    dec_b = _get_decision(rec_b)
    if dec_a != dec_b and dec_a is not None and dec_b is not None:
        return ComparabilityResult(
            comparable=False,
            same_detector=(rec_a.detector == rec_b.detector),
            same_semantics=False,
            reason=(
                f"Decision context mismatch: {dec_a.value} vs {dec_b.value}. "
                "Evidence records must be evaluated in the same decision context to be comparable."
            ),
        )

    # Rule 3: semantics check
    same_semantics = _same_semantics(rec_a, rec_b)

    # Same detector is strongest signal
    same_detector = rec_a.detector is not None and rec_a.detector == rec_b.detector
    if same_detector and dec_a == dec_b:
        reason = ""
    elif not same_detector and dec_a == dec_b:
        reason = (
            f"Different detectors ({rec_a.detector!r} vs {rec_b.detector!r}) "
            "but same decision context; cross-detector comparison is allowed."
        )
    else:
        reason = ""

    return ComparabilityResult(
        comparable=True,
        same_detector=same_detector,
        same_semantics=same_semantics,
        reason=reason,
    )


def is_comparable(rec_a: EvidenceRecord, rec_b: EvidenceRecord) -> bool:
    """Return True iff rec_a and rec_b are comparable.

    Convenience wrapper around ``check_comparability``.
    """
    return check_comparability(rec_a, rec_b).comparable


def _get_decision(rec: EvidenceRecord) -> Any | None:
    """Extract the VerdictDecision from a record's applicability, or None."""
    if rec.applicability is not None:
        return rec.applicability.decision
    return None


def _same_semantics(rec_a: EvidenceRecord, rec_b: EvidenceRecord) -> bool:
    """Check whether two records have compatible semantics signals.

    Two records are semantically compatible if they share the same
    evidence_class or both are detection-side / both are comparison-side.
    """
    detection_classes = {
        EvidenceClass.POINT_ANOMALY.value,
        EvidenceClass.LOCAL_OUTLIER.value,
        EvidenceClass.GLOBAL_OUTLIER.value,
        EvidenceClass.DENSITY_NOISE.value,
        EvidenceClass.REGIME_SHIFT.value,
        EvidenceClass.DISTRIBUTION_SHIFT.value,
        EvidenceClass.VARIANCE_COLLAPSE.value,
    }
    comparison_classes = {
        EvidenceClass.COMPARISON_DIFFERENCE.value,
        EvidenceClass.COMPARISON_EQUIVALENT.value,
        EvidenceClass.IMPROVEMENT.value,
        EvidenceClass.REGRESSION.value,
        EvidenceClass.NEUTRAL.value,
        EvidenceClass.INSUFFICIENT_EVIDENCE.value,
        EvidenceClass.INCOMPARABLE.value,
    }

    if rec_a.evidence_class == rec_b.evidence_class:
        return True

    both_detection = (
        rec_a.evidence_class in detection_classes and rec_b.evidence_class in detection_classes
    )
    both_comparison = (
        rec_a.evidence_class in comparison_classes and rec_b.evidence_class in comparison_classes
    )
    return both_detection or both_comparison
