"""summarize_evidence skill: reduce a list of EvidenceRecord objects to a
consensus summary.

SKILL_OUTPUT_CANNOT_OVERRIDE_CORE_EVIDENCE:
  - May reorder, group, filter EvidenceRecords
  - May compute aggregate confidence and consensus evidence_class (by majority vote)
  - Must NOT alter any individual EvidenceRecord.evidence_class or verdict

No LLM. No autonomous loops.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from anomaly_lab.evidence.taxonomy import EvidenceRecord


@dataclass(frozen=True, kw_only=True)
class EvidenceSummary:
    total_records: int
    evidence_class_counts: dict[str, int]
    aggregate_confidence: float | None  # mean of non-None confidences
    consensus_evidence_class: str | None  # majority vote, None if tie
    records_by_class: dict[str, list[dict[str, Any]]]
    warnings: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_records": self.total_records,
            "evidence_class_counts": dict(self.evidence_class_counts),
            "aggregate_confidence": self.aggregate_confidence,
            "consensus_evidence_class": self.consensus_evidence_class,
            "records_by_class": {k: list(v) for k, v in self.records_by_class.items()},
            "warnings": list(self.warnings),
        }


def summarize_evidence(records: list[EvidenceRecord]) -> EvidenceSummary:
    """Summarise a collection of EvidenceRecords without modifying them.

    Rules
    -----
    - evidence_class_counts: raw tally per class
    - aggregate_confidence: arithmetic mean of non-None confidences
    - consensus_evidence_class: majority vote; None on tie or empty
    - records_by_class: groups records by evidence_class (to_dict of each)
    - No individual record is altered
    """
    warnings: list[str] = []

    if not records:
        return EvidenceSummary(
            total_records=0,
            evidence_class_counts={},
            aggregate_confidence=None,
            consensus_evidence_class=None,
            records_by_class={},
            warnings=("no_records_to_summarize",),
        )

    # Raw counts — immutable operation
    class_counts: dict[str, int] = {}
    confidences: list[float] = []
    by_class: dict[str, list[dict[str, Any]]] = {}

    for rec in records:
        cls = rec.evidence_class
        class_counts[cls] = class_counts.get(cls, 0) + 1
        if rec.confidence is not None:
            confidences.append(rec.confidence)
        by_class.setdefault(cls, []).append(rec.to_dict())

    # Majority-vote consensus
    if class_counts:
        max_count = max(class_counts.values())
        winners = [k for k, v in class_counts.items() if v == max_count]
        consensus = winners[0] if len(winners) == 1 else None
    else:
        consensus = None

    agg_conf = sum(confidences) / len(confidences) if confidences else None

    if consensus is None and len(records) > 1:
        warnings.append("consensus_tied_or_ambiguous")

    return EvidenceSummary(
        total_records=len(records),
        evidence_class_counts=class_counts,
        aggregate_confidence=agg_conf,
        consensus_evidence_class=consensus,
        records_by_class=by_class,
        warnings=tuple(warnings),
    )
