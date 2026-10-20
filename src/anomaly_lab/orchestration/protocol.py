"""protocol: types that define the orchestration contract.

A WorkflowStep is an immutable description of one step in an analysis pipeline.
An OrchestrationProtocol is a tuple of WorkflowSteps representing a complete run.

No LLM. No autonomous loops.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class StepType(Enum):
    SKILL = "SKILL"
    GUARDRAIL_CHECK = "GUARDRAIL_CHECK"
    DECISION = "DECISION"  # human-in-the-loop gate


@dataclass(frozen=True, kw_only=True)
class WorkflowStep:
    step_id: str
    step_type: StepType
    skill_name: str | None = None  # None for non-skill steps
    input_keys: tuple[str, ...] = ()
    output_key: str | None = None
    guardrails: tuple[str, ...] = ()  # guardrail names to run after this step
    description: str = ""
    continue_on_failure: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "step_id": self.step_id,
            "step_type": self.step_type.value,
            "skill_name": self.skill_name,
            "input_keys": list(self.input_keys),
            "output_key": self.output_key,
            "guardrails": list(self.guardrails),
            "description": self.description,
            "continue_on_failure": self.continue_on_failure,
        }


# Sentinel for a step that has not yet produced output
_NOT_RUN = object()


@dataclass
class OrchestrationProtocol:
    """An immutable, ordered sequence of WorkflowSteps.

    Attributes
    ----------
    name: human-readable protocol name
    version: protocol version string
    steps: tuple of WorkflowStep in execution order
    """

    name: str
    version: str
    steps: tuple[WorkflowStep, ...] = field(default_factory=tuple)

    def step_ids(self) -> tuple[str, ...]:
        return tuple(s.step_id for s in self.steps)

    def step(self, step_id: str) -> WorkflowStep | None:
        for s in self.steps:
            if s.step_id == step_id:
                return s
        return None

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "version": self.version,
            "steps": [s.to_dict() for s in self.steps],
        }


# -----------------------------------------------------------------------
# Canonical protocols
# -----------------------------------------------------------------------

PROTOCOL_DETECT = OrchestrationProtocol(
    name="detect",
    version="0.5",
    steps=(
        WorkflowStep(
            step_id="inspect",
            step_type=StepType.SKILL,
            skill_name="inspect_data",
            input_keys=("df",),
            output_key="inspection",
            guardrails=(),
            description="Inspect input DataFrame structure",
        ),
        WorkflowStep(
            step_id="plan",
            step_type=StepType.SKILL,
            skill_name="select_analysis_plan",
            input_keys=("inspection", "intent"),
            output_key="plan",
            guardrails=(),
            description="Select detectors based on inspection",
        ),
        WorkflowStep(
            step_id="run",
            step_type=StepType.SKILL,
            skill_name="run_experiment",
            input_keys=("plan", "df"),
            output_key="experiment_results",
            guardrails=("SKILL_OUTPUT_CANNOT_OVERRIDE_CORE_EVIDENCE",),
            description="Execute detector(s) on data",
        ),
        WorkflowStep(
            step_id="summarise",
            step_type=StepType.SKILL,
            skill_name="summarize_evidence",
            input_keys=("experiment_results",),
            output_key="evidence_summary",
            guardrails=("SKILL_OUTPUT_CANNOT_OVERRIDE_CORE_EVIDENCE",),
            description="Aggregate evidence records",
        ),
        WorkflowStep(
            step_id="export",
            step_type=StepType.SKILL,
            skill_name="export_report",
            input_keys=("evidence_summary",),
            output_key="report",
            guardrails=(),
            description="Serialise report",
        ),
    ),
)

PROTOCOL_COMPARE = OrchestrationProtocol(
    name="compare",
    version="0.5",
    steps=(
        WorkflowStep(
            step_id="inspect_baseline",
            step_type=StepType.SKILL,
            skill_name="inspect_data",
            input_keys=("df_baseline",),
            output_key="inspection_baseline",
            guardrails=(),
            description="Inspect baseline DataFrame",
        ),
        WorkflowStep(
            step_id="inspect_candidate",
            step_type=StepType.SKILL,
            skill_name="inspect_data",
            input_keys=("df_candidate",),
            output_key="inspection_candidate",
            guardrails=(),
            description="Inspect candidate DataFrame",
        ),
        WorkflowStep(
            step_id="validate_comparison",
            step_type=StepType.SKILL,
            skill_name="validate_comparison",
            input_keys=("baseline", "candidate"),
            output_key="validation_result",
            guardrails=(),
            description="Validate baseline/candidate compatibility",
        ),
        WorkflowStep(
            step_id="summarise",
            step_type=StepType.SKILL,
            skill_name="summarize_evidence",
            input_keys=("experiment_results",),
            output_key="evidence_summary",
            guardrails=("SKILL_OUTPUT_CANNOT_OVERRIDE_CORE_EVIDENCE",),
            description="Aggregate evidence records",
        ),
        WorkflowStep(
            step_id="export",
            step_type=StepType.SKILL,
            skill_name="export_report",
            input_keys=("evidence_summary",),
            output_key="report",
            guardrails=(),
            description="Serialise report",
        ),
    ),
)
