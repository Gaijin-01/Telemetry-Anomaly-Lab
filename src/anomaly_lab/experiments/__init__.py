"""Experiment specification, results, hashing, provenance, and replay."""

from anomaly_lab.experiments.hashing import (
    ExperimentHasher,
    canonical_experiment_hash,
    canonical_parameters_hash,
    canonical_spec_hash,
)
from anomaly_lab.experiments.planner import (
    ApplicabilityCode as ApplicabilityCode,
)
from anomaly_lab.experiments.planner import (
    ApplicabilityRecord as ApplicabilityRecord,
)
from anomaly_lab.experiments.planner import (
    ComponentKind as ComponentKind,
)
from anomaly_lab.experiments.planner import (
    ExperimentPlan as ExperimentPlan,
)
from anomaly_lab.experiments.planner import (
    PlannedStep as PlannedStep,
)

# Planner and runner (v0.5 applicability planner and experiment runner)
from anomaly_lab.experiments.planner import (
    plan as plan,
)
from anomaly_lab.experiments.provenance import ProvenanceRecord, collect_environment
from anomaly_lab.experiments.replay import ExperimentReplay
from anomaly_lab.experiments.result import ExperimentResult
from anomaly_lab.experiments.runner import (
    PlanExecutionResult as PlanExecutionResult,
)
from anomaly_lab.experiments.runner import (
    StepResult as StepResult,
)
from anomaly_lab.experiments.runner import (
    StepStatus as StepStatus,
)
from anomaly_lab.experiments.runner import (
    run as run,
)
from anomaly_lab.experiments.runner import (
    run_from_path as run_from_path,
)
from anomaly_lab.experiments.spec import ExperimentSpec

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
