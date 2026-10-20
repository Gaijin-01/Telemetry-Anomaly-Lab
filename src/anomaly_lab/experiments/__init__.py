"""Experiment specification, results, hashing, provenance, and replay."""

from anomaly_lab.experiments.spec import ExperimentSpec
from anomaly_lab.experiments.result import ExperimentResult
from anomaly_lab.experiments.hashing import (
    ExperimentHasher,
    canonical_experiment_hash,
    canonical_parameters_hash,
    canonical_spec_hash,
)
from anomaly_lab.experiments.provenance import ProvenanceRecord, collect_environment
from anomaly_lab.experiments.replay import ExperimentReplay

# Planner and runner (v0.5 applicability planner and experiment runner)
from anomaly_lab.experiments.planner import (
    plan as plan,
    ExperimentPlan as ExperimentPlan,
    PlannedStep as PlannedStep,
    ApplicabilityRecord as ApplicabilityRecord,
    ApplicabilityCode as ApplicabilityCode,
    ComponentKind as ComponentKind,
)
from anomaly_lab.experiments.runner import (
    run as run,
    run_from_path as run_from_path,
    PlanExecutionResult as PlanExecutionResult,
    StepResult as StepResult,
    StepStatus as StepStatus,
)

__all__ = [
    # Existing
    "ExperimentSpec",
    "ExperimentResult",
    "ExperimentHasher",
    "canonical_experiment_hash",
    "canonical_parameters_hash",
    "canonical_spec_hash",
    "ProvenanceRecord",
    "collect_environment",
    "ExperimentReplay",
    # New v0.5 planner and runner
    "plan",
    "ExperimentPlan",
    "PlannedStep",
    "ApplicabilityRecord",
    "ApplicabilityCode",
    "ComponentKind",
    "run",
    "run_from_path",
    "PlanExecutionResult",
    "StepResult",
    "StepStatus",
]
