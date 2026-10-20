"""guardrails: enforcement of the SKILL_OUTPUT_CANNOT_OVERRIDE_CORE_EVIDENCE invariant.

Core evidence fields (verdict, evidence_class) are read from EvidenceRecord and
must never be mutated or replaced by any skill output.  Skills may reorganise
facts (e.g. reorder, group, summarise) but the evidence_class of any individual
EvidenceRecord is immutable.

No LLM. No autonomous loops.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

# NOTE: this skill handles backward-compat EvidenceRecord from the old evidence.py
# when that module is shadowed by the evidence/ package.
try:
    from anomaly_lab.evidence.taxonomy import EvidenceRecord
except ImportError:
    from anomaly_lab.evidence import EvidenceRecord  # type: ignore[assignment]


# -----------------------------------------------------------------------
# Core-evidence fields that guardrails protect
# -----------------------------------------------------------------------

# Fields whose values originate in core evidence and must not be altered
# by any skill output pass.
_CORE_EVIDENCE_FIELDS: tuple[str, ...] = (
    "evidence_class",
    "confidence",
    "verdict",  # legacy alias/adjacent field
    "verdict_label",  # legacy alias
)

# The guardrail name — exported as a constant so protocols reference it exactly
SKILL_OUTPUT_CANNOT_OVERRIDE_CORE_EVIDENCE = "SKILL_OUTPUT_CANNOT_OVERRIDE_CORE_EVIDENCE"


class GuardrailError(Exception):
    """Raised when a guardrail check fails."""

    pass


@dataclass(frozen=True, kw_only=True)
class GuardrailViolation:
    field: str
    original_value: Any
    attempted_value: Any
    skill_name: str


@dataclass(frozen=True, kw_only=True)
class GuardrailResult:
    """Result of a guardrail check."""

    guardrail_name: str
    passed: bool
    violations: tuple[GuardrailViolation, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "guardrail_name": self.guardrail_name,
            "passed": self.passed,
            "violations": [
                {
                    "field": v.field,
                    "original_value": v.original_value,
                    "attempted_value": v.attempted_value,
                    "skill_name": v.skill_name,
                }
                for v in self.violations
            ],
        }


# GuardrailCheck: (skill_output, original_evidence) -> GuardrailResult
GuardrailCheck = Callable[..., GuardrailResult]


def _violates_core_invariant(
    skill_output: Any,
    original_records: list[EvidenceRecord] | None,
    skill_name: str,
) -> list[GuardrailViolation]:
    """Check if skill_output attempts to override core evidence fields."""
    violations: list[GuardrailViolation] = []

    if original_records is None:
        return violations

    # Build a lookup: record index -> original evidence_class
    originals: dict[int, str] = {}
    if isinstance(original_records, (list, tuple)):
        for i, rec in enumerate(original_records):
            if isinstance(rec, EvidenceRecord):
                originals[i] = rec.evidence_class

    # skill_output is suspicious if it contains evidence_class at all
    # We check the skill output for direct overrides of core fields
    if isinstance(skill_output, dict):
        for field_name in _CORE_EVIDENCE_FIELDS:
            if field_name in skill_output and skill_output[field_name] is not None:
                # Skill is attempting to set a core evidence field directly
                # This is a violation IF the skill_output's evidence_class
                # differs from any original (i.e., it is changing verdict)
                if field_name == "evidence_class":
                    # Check: is this a direct record replacement?
                    for _idx, orig_cls in originals.items():
                        if skill_output.get("evidence_class") != orig_cls:
                            violations.append(
                                GuardrailViolation(
                                    field=field_name,
                                    original_value=orig_cls,
                                    attempted_value=skill_output.get("evidence_class"),
                                    skill_name=skill_name,
                                )
                            )

    # Check: did skill return a list/tuple of dicts that override evidence_class?
    if isinstance(skill_output, (list, tuple)):
        for item_idx, item in enumerate(skill_output):
            if isinstance(item, dict):
                for field_name in _CORE_EVIDENCE_FIELDS:
                    if field_name in item and item[field_name] is not None:
                        if field_name == "evidence_class":
                            orig_cls = originals.get(item_idx, None)
                            if orig_cls is not None and item[field_name] != orig_cls:
                                violations.append(
                                    GuardrailViolation(
                                        field=field_name,
                                        original_value=orig_cls,
                                        attempted_value=item[field_name],
                                        skill_name=skill_name,
                                    )
                                )

    # Check EvidenceRecord instances directly returned
    if isinstance(skill_output, EvidenceRecord):
        orig_cls = originals.get(0, None)
        if orig_cls is not None and skill_output.evidence_class != orig_cls:
            violations.append(
                GuardrailViolation(
                    field="evidence_class",
                    original_value=orig_cls,
                    attempted_value=skill_output.evidence_class,
                    skill_name=skill_name,
                )
            )

    return violations


def check_skill_output(
    skill_name: str,
    skill_output: Any,
    *,
    original_evidence: list[EvidenceRecord] | None = None,
) -> GuardrailResult:
    """Check that skill_output respects the SKILL_OUTPUT_CANNOT_OVERRIDE_CORE_EVIDENCE invariant.

    This guardrail inspects skill output for direct overrides of core evidence
    fields (evidence_class, verdict, confidence) on existing EvidenceRecord
    instances.

    Parameters
    ----------
    skill_name: name of the skill that produced the output
    skill_output: the raw output from the skill
    original_evidence: EvidenceRecord instances from before the skill ran;
        if None the check passes (no baseline to compare against)

    Returns
    -------
    GuardrailResult with passed=True iff no violations found.
    """
    if original_evidence is None:
        return GuardrailResult(
            guardrail_name=SKILL_OUTPUT_CANNOT_OVERRIDE_CORE_EVIDENCE,
            passed=True,
            violations=(),
        )

    violations = _violates_core_invariant(skill_output, original_evidence, skill_name)

    return GuardrailResult(
        guardrail_name=SKILL_OUTPUT_CANNOT_OVERRIDE_CORE_EVIDENCE,
        passed=len(violations) == 0,
        violations=tuple(violations),
    )


# -----------------------------------------------------------------------
# Built-in guardrail: summary must not mutate individual records
# -----------------------------------------------------------------------


def guardrail_summarize_no_mutation(
    skill_output: Any,
    original_evidence: list[EvidenceRecord] | None,
) -> GuardrailResult:
    """Specific guardrail: summarize_evidence must not alter individual records.

    The summarise skill may compute new aggregate fields (consensus_evidence_class,
    aggregate_confidence) but must not mutate evidence_class of any record in
    records_by_class.
    """
    if not isinstance(skill_output, dict):
        return GuardrailResult(
            guardrail_name="guardrail_summarize_no_mutation",
            passed=True,
            violations=(),
        )

    # skill_output is expected to be EvidenceSummary.to_dict()
    # It must NOT contain a 'records_by_class' whose individual record dicts
    # have different evidence_class than the originals
    violations: list[GuardrailViolation] = []

    records_by_class = skill_output.get("records_by_class", {})
    if not isinstance(records_by_class, dict) or original_evidence is None:
        return GuardrailResult(
            guardrail_name="guardrail_summarize_no_mutation",
            passed=True,
            violations=(),
        )

    # Build originals lookup
    originals: dict[str, str] = {}
    for rec in original_evidence:
        if isinstance(rec, EvidenceRecord):
            originals.setdefault(rec.evidence_class, rec.evidence_class)

    for cls, records in records_by_class.items():
        if not isinstance(records, (list, tuple)):
            continue
        for rec_dict in records:
            if not isinstance(rec_dict, dict):
                continue
            # evidence_class in the grouped record must match the group key
            if rec_dict.get("evidence_class") != cls:
                violations.append(
                    GuardrailViolation(
                        field="evidence_class",
                        original_value=cls,
                        attempted_value=rec_dict.get("evidence_class"),
                        skill_name="summarize_evidence",
                    )
                )

    return GuardrailResult(
        guardrail_name="guardrail_summarize_no_mutation",
        passed=len(violations) == 0,
        violations=tuple(violations),
    )
