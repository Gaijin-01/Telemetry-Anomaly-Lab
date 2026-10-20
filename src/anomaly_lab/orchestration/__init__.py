"""Orchestration: protocol definitions, action primitives, and guardrails.

No LLM. No autonomous loops.
"""

from anomaly_lab.orchestration.actions import Action, ActionRegistry, ActionResult, act
from anomaly_lab.orchestration.guardrails import (
    SKILL_OUTPUT_CANNOT_OVERRIDE_CORE_EVIDENCE,
    GuardrailCheck,
    GuardrailError,
    GuardrailResult,
    check_skill_output,
)
from anomaly_lab.orchestration.protocol import OrchestrationProtocol, StepType, WorkflowStep

__all__ = [
    "OrchestrationProtocol",
    "StepType",
    "WorkflowStep",
    "Action",
    "ActionResult",
    "ActionRegistry",
    "act",
    "SKILL_OUTPUT_CANNOT_OVERRIDE_CORE_EVIDENCE",
    "GuardrailError",
    "GuardrailResult",
    "GuardrailCheck",
    "check_skill_output",
]
