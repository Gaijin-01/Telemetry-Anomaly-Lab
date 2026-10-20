"""Applicability: decision-context mapping for evidence classes.

TOML-serialisable rules dict; no LLM dependencies.
"""

from __future__ import annotations

from typing import Any

from anomaly_lab.evidence.taxonomy import (
    DEFAULT_APPLICABILITY_RULES,
    DetectionApplicability,
    EvidenceApplicability,
    EvidenceRecord,
    VerdictDecision,
)


def get_applicability(record: EvidenceRecord) -> EvidenceApplicability:
    """Return the EvidenceApplicability for an EvidenceRecord.

    Uses the class-level DEFAULT_APPLICABILITY_RULES to look up the
    decision and applicable flag from the evidence_class string value.

    If the record already carries an ``applicability`` field that is not None,
    that value is returned directly (record-level override).
    """
    if record.applicability is not None:
        return record.applicability

    evidence_class = record.evidence_class
    rules = DEFAULT_APPLICABILITY_RULES

    # Try detection rules first, then comparison rules
    for section in ("detection_rules", "comparison_rules"):
        section_rules: dict[str, dict[str, Any]] = rules.get(section, {})
        if evidence_class in section_rules:
            rule = section_rules[evidence_class]
            decision = VerdictDecision(rule["decision"])
            domain_constraints = tuple(rule.get("domain_constraints", []))
            applicable = bool(rule["applicable"])

            if section == "detection_rules":
                return DetectionApplicability(
                    applicable=applicable,
                    decision=decision,
                    domain_constraints=domain_constraints,
                )
            else:
                return ComparisonApplicability(
                    applicable=applicable,
                    decision=decision,
                    domain_constraints=domain_constraints,
                )

    # Unknown class — not applicable by default
    return EvidenceApplicability(
        applicable=False,
        decision=VerdictDecision.ANOMALY_DETECTION,
        domain_constraints=("unknown_evidence_class",),
    )


# Local import to avoid circular at module level
from anomaly_lab.evidence.taxonomy import ComparisonApplicability  # noqa: E402
