"""Taxonomy: evidence classes, causality status, verdict enums, and default rules.

TOML-serialisable: all enum values are strings; default rules are a plain dict.
No LLM dependencies.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any

# -----------------------------------------------------------------------
# Evidence class taxonomy
# -----------------------------------------------------------------------


class EvidenceClass(Enum):
    """Taxonomy of evidence classes produced by the anomaly-lab reasoning layer.

    Detection-side classes (produced by anomaly detectors):
        POINT_ANOMALY       — single statistically anomalous point
        LOCAL_OUTLIER       — locally anomalous (differs from neighbours)
        GLOBAL_OUTLIER     — globally anomalous (majority of points flagged)
        DENSITY_NOISE       — DBSCAN density noise label
        REGIME_SHIFT        — regime change detected (change-point / clustering)
        DISTRIBUTION_SHIFT  — statistical distribution change (train vs. inference)
        VARIANCE_COLLAPSE   — zero/near-zero variance collapse

    Comparison-side classes (produced by statistical comparison):
        COMPARISON_DIFFERENCE — statistically significant difference detected
        COMPARISON_EQUIVALENT — within-margin equivalence (TOST)
        IMPROVEMENT          — candidate is significantly better on at least one metric
        REGRESSION          — candidate is significantly worse on at least one metric
        NEUTRAL             — no significant difference in either direction
        INSUFFICIENT_EVIDENCE — not enough data to draw any conclusion
        INCOMPARABLE        — metadata mismatch precludes meaningful comparison

    Special classes:
        INCONCLUSIVE        — detector ran but could not reach a conclusion
        NOT_APPLICABLE      — this evidence class is not applicable to the decision
    """

    # Detection
    POINT_ANOMALY = "POINT_ANOMALY"
    LOCAL_OUTLIER = "LOCAL_OUTLIER"
    GLOBAL_OUTLIER = "GLOBAL_OUTLIER"
    DENSITY_NOISE = "DENSITY_NOISE"
    REGIME_SHIFT = "REGIME_SHIFT"
    DISTRIBUTION_SHIFT = "DISTRIBUTION_SHIFT"
    VARIANCE_COLLAPSE = "VARIANCE_COLLAPSE"

    # Comparison
    COMPARISON_DIFFERENCE = "COMPARISON_DIFFERENCE"
    COMPARISON_EQUIVALENT = "COMPARISON_EQUIVALENT"
    IMPROVEMENT = "IMPROVEMENT"
    REGRESSION = "REGRESSION"
    NEUTRAL = "NEUTRAL"
    INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"
    INCOMPARABLE = "INCOMPARABLE"

    # Special
    INCONCLUSIVE = "INCONCLUSIVE"
    NOT_APPLICABLE = "NOT_APPLICABLE"


# -----------------------------------------------------------------------
# Causality status
# -----------------------------------------------------------------------


class CausalityStatus(Enum):
    """Whether a causal relationship has been explicitly established.

    Default is NOT_ESTABLISHED for all evidence. Domain knowledge is required
    to upgrade this status; statistical evidence alone never establishes causality.
    """

    NOT_ESTABLISHED = "NOT_ESTABLISHED"
    ESTABLISHED = "ESTABLISHED"
    INDETERMINATE = "INDETERMINATE"


# -----------------------------------------------------------------------
# Evidence strength
# -----------------------------------------------------------------------


class EvidenceStrength(Enum):
    """Strength of evidence produced by the reasoning layer."""

    STRONG = "STRONG"
    MODERATE = "MODERATE"
    WEAK = "WEAK"
    NONE = "NONE"


# -----------------------------------------------------------------------
# Verdict
# -----------------------------------------------------------------------


class EvidenceVerdict(Enum):
    """Verdict for a given evidence record and decision context.

    These verdicts are derived from evidence — they are not raw detector outputs.
    """

    PASS = "PASS"  # evidence supports the decision / claim
    FAIL = "FAIL"  # evidence contradicts the decision / claim
    INCONCLUSIVE = "INCONCLUSIVE"  # evidence is insufficient to reach a verdict
    CANNOT_PASS = "CANNOT_PASS"  # execution failed or evidence cannot be used to pass


# -----------------------------------------------------------------------
# Verdict decision context
# -----------------------------------------------------------------------


class VerdictDecision(Enum):
    """Decision context that an EvidenceRecord is being evaluated against."""

    ANOMALY_DETECTION = "ANOMALY_DETECTION"
    REGRESSION_TEST = "REGRESSION_TEST"
    EQUIVALENCE_TEST = "EQUIVALENCE_TEST"
    DISTRIBUTION_COMPARISON = "DISTRIBUTION_COMPARISON"


# -----------------------------------------------------------------------
# Verdict reason (machine-readable)
# -----------------------------------------------------------------------


class VerdictReason(Enum):
    """Machine-readable reason explaining why a particular verdict was reached."""

    # Execution
    EXECUTION_SUCCEEDED = "EXECUTION_SUCCEEDED"
    EXECUTION_FAILED = "EXECUTION_FAILED"
    FAILED_EXECUTION = "FAILED_EXECUTION"

    # Detection
    ANOMALY_DETECTED = "ANOMALY_DETECTED"
    NO_ANOMALY_DETECTED = "NO_ANOMALY_DETECTED"
    INSUFFICIENT_SAMPLES = "INSUFFICIENT_SAMPLES"

    # Comparison
    IMPROVEMENT_DETECTED = "IMPROVEMENT_DETECTED"
    REGRESSION_DETECTED = "REGRESSION_DETECTED"
    NEUTRAL_OUTCOME = "NEUTRAL_OUTCOME"
    EQUIVALENCE_ESTABLISHED = "EQUIVALENCE_ESTABLISHED"
    INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"
    NOT_APPLICABLE = "NOT_APPLICABLE"
    INCOMPARABLE = "INCOMPARABLE"

    # Generic
    CAUSALITY_NOT_ESTABLISHED = "CAUSALITY_NOT_ESTABLISHED"


# -----------------------------------------------------------------------
# EvidenceRecord — extended with causality, applicability, verdict
# -----------------------------------------------------------------------


@dataclass(frozen=True, kw_only=True)
class EvidenceRecord:
    """Structured evidence produced by the anomaly-lab reasoning layer.

    Extends the v0.4 EvidenceRecord with:
        causality_status  — always NOT_ESTABLISHED unless explicitly upgraded
        applicability      — EvidenceApplicability describing decision context
        verdict            — EvidenceVerdict with reason, or None if not yet computed
    """

    evidence_class: str
    confidence: float | None  # 0.0–1.0, None if not applicable
    detector: str | None
    score: float | None
    threshold: float | None
    supporting_metrics: dict[str, float]
    assumptions: tuple[str, ...]
    limitations: tuple[str, ...]
    interpretation: str

    # v0.5 additions
    causality_status: CausalityStatus = CausalityStatus.NOT_ESTABLISHED
    applicability: EvidenceApplicability | None = None
    verdict: EvidenceVerdictResult | None = None

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
            "causality_status": self.causality_status.value,
            "applicability": self.applicability.to_dict() if self.applicability else None,
            "verdict": self.verdict.to_dict() if self.verdict else None,
        }


# -----------------------------------------------------------------------
# Applicability
# -----------------------------------------------------------------------


@dataclass(frozen=True, kw_only=True)
class EvidenceApplicability:
    """Describes whether and how an EvidenceRecord applies to a decision context."""

    applicable: bool
    decision: VerdictDecision
    domain_constraints: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "applicable": self.applicable,
            "decision": self.decision.value,
            "domain_constraints": list(self.domain_constraints),
        }


@dataclass(frozen=True, kw_only=True)
class DetectionApplicability(EvidenceApplicability):
    """Applicability specific to anomaly detection evidence."""

    detector_family: str | None = None  # e.g. "statistical", "density", "isolation"


@dataclass(frozen=True, kw_only=True)
class ComparisonApplicability(EvidenceApplicability):
    """Applicability specific to comparison evidence."""

    requires_paired_structure: bool = False

    def to_dict(self) -> dict[str, Any]:
        base = super().to_dict()
        base["requires_paired_structure"] = self.requires_paired_structure
        return base


# -----------------------------------------------------------------------
# Verdict result
# -----------------------------------------------------------------------


@dataclass(frozen=True, kw_only=True)
class EvidenceVerdictResult:
    """Result of evaluating an EvidenceRecord against a decision context."""

    verdict: EvidenceVerdict
    reason: str  # human-readable
    reason_code: VerdictReason
    decision: VerdictDecision
    causality_status: CausalityStatus

    def to_dict(self) -> dict[str, Any]:
        return {
            "verdict": self.verdict.value,
            "reason": self.reason,
            "reason_code": self.reason_code.value,
            "decision": self.decision.value,
            "causality_status": self.causality_status.value,
        }


# -----------------------------------------------------------------------
# Default TOML-serialisable applicability rules
# -----------------------------------------------------------------------

# Version tag for the rules schema
APPLICABILITY_RULES_VERSION = "0.5.0"

DEFAULT_APPLICABILITY_RULES: dict[str, Any] = {
    "version": APPLICABILITY_RULES_VERSION,
    # Mapping: evidence_class -> applicability rule
    "detection_rules": {
        "POINT_ANOMALY": {
            "applicable": True,
            "decision": "ANOMALY_DETECTION",
            "domain_constraints": [],
        },
        "LOCAL_OUTLIER": {
            "applicable": True,
            "decision": "ANOMALY_DETECTION",
            "domain_constraints": ["local_context_meaningful"],
        },
        "GLOBAL_OUTLIER": {
            "applicable": True,
            "decision": "ANOMALY_DETECTION",
            "domain_constraints": ["majority_flagged_check"],
        },
        "DENSITY_NOISE": {
            "applicable": True,
            "decision": "ANOMALY_DETECTION",
            "domain_constraints": ["density_assumption_valid"],
        },
        "REGIME_SHIFT": {
            "applicable": True,
            "decision": "DISTRIBUTION_COMPARISON",
            "domain_constraints": ["temporal_order_required"],
        },
        "DISTRIBUTION_SHIFT": {
            "applicable": True,
            "decision": "DISTRIBUTION_COMPARISON",
            "domain_constraints": ["train_test_split_valid"],
        },
        "VARIANCE_COLLAPSE": {
            "applicable": True,
            "decision": "ANOMALY_DETECTION",
            "domain_constraints": ["non_trivial_variance"],
        },
        "INCONCLUSIVE": {
            "applicable": False,
            "decision": "ANOMALY_DETECTION",
            "domain_constraints": ["insufficient_evidence"],
        },
        "NOT_APPLICABLE": {
            "applicable": False,
            "decision": "ANOMALY_DETECTION",
            "domain_constraints": [],
        },
    },
    "comparison_rules": {
        "IMPROVEMENT": {
            "applicable": True,
            "decision": "REGRESSION_TEST",
            "domain_constraints": [],
        },
        "REGRESSION": {
            "applicable": True,
            "decision": "REGRESSION_TEST",
            "domain_constraints": [],
        },
        "NEUTRAL": {
            "applicable": True,
            "decision": "REGRESSION_TEST",
            "domain_constraints": ["no_significant_difference"],
        },
        "COMPARISON_DIFFERENCE": {
            "applicable": True,
            "decision": "REGRESSION_TEST",
            "domain_constraints": [],
        },
        "COMPARISON_EQUIVALENT": {
            "applicable": True,
            "decision": "EQUIVALENCE_TEST",
            "domain_constraints": ["tost_margin_valid"],
        },
        "INSUFFICIENT_EVIDENCE": {
            "applicable": False,
            "decision": "REGRESSION_TEST",
            "domain_constraints": ["insufficient_samples"],
        },
        "INCOMPARABLE": {
            "applicable": False,
            "decision": "REGRESSION_TEST",
            "domain_constraints": ["metadata_mismatch"],
        },
        "NOT_APPLICABLE": {
            "applicable": False,
            "decision": "REGRESSION_TEST",
            "domain_constraints": [],
        },
    },
}
