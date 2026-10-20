"""export_report skill: serialise evidence summary + metadata to a structured
dict or JSON-serialisable representation.

No LLM. No autonomous loops.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from anomaly_lab.skills.summarize_evidence import EvidenceSummary


@dataclass(frozen=True, kw_only=True)
class ReportMetadata:
    dataset_name: str | None
    feature: str | None
    analyst_notes: str | None
    tool_version: str
    generated_at: str  # ISO-8601 string, set by caller

    def to_dict(self) -> dict[str, Any]:
        return {
            "dataset_name": self.dataset_name,
            "feature": self.feature,
            "analyst_notes": self.analyst_notes,
            "tool_version": self.tool_version,
            "generated_at": self.generated_at,
        }


@dataclass(frozen=True, kw_only=True)
class ExportedReport:
    metadata: ReportMetadata
    evidence_summary: dict[str, Any]
    skill_chain: tuple[str, ...]  # ordered list of skills that ran

    def to_dict(self) -> dict[str, Any]:
        return {
            "metadata": self.metadata.to_dict(),
            "evidence_summary": self.evidence_summary,
            "skill_chain": list(self.skill_chain),
        }


def export_report(
    evidence_summary: EvidenceSummary,
    *,
    metadata: ReportMetadata | None = None,
    skill_chain: tuple[str, ...] = (),
) -> ExportedReport:
    """Package an EvidenceSummary into an ExportedReport.

    Does not write to disk — caller handles I/O.
    SKILL_OUTPUT_CANNOT_OVERRIDE_CORE_EVIDENCE:
      evidence_summary is serialised as-is; no field is mutated.
    """
    meta = metadata or ReportMetadata(
        dataset_name=None,
        feature=None,
        analyst_notes=None,
        tool_version="0.5.0",
        generated_at="",
    )

    return ExportedReport(
        metadata=meta,
        evidence_summary=evidence_summary.to_dict(),
        skill_chain=skill_chain,
    )
