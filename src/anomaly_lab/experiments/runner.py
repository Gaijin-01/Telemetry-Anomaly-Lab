"""Experiment runner for anomaly-lab.

Executes an applicability plan by orchestrating the existing deterministic
components (detectors, change_points, regimes, consensus).  The runner does
NOT reimplement detector logic — it delegates to the existing module-level
functions in anomaly_lab.detectors.*, anomaly_lab.change_points, etc.

Output is TOML only, per project requirement.
"""

from __future__ import annotations

import io
import json
import sys
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from anomaly_lab.experiments.planner import ExperimentPlan, ComponentKind, PlannedStep
from anomaly_lab.ingest import read_csv
from anomaly_lab.semantics import classify
from anomaly_lab import consensus as consensus_module
from anomaly_lab import change_points as change_points_module
from anomaly_lab import regimes as regimes_module
from anomaly_lab.compare import MetricRule, compare_runs, RunMetadata
from anomaly_lab.results import DetectionResult

# ------------------------------------------------------------------ #
# Result types
# ------------------------------------------------------------------ #


class StepStatus(Enum):
    SUCCESS = "SUCCESS"
    SKIPPED = "SKIPPED"
    FAILED = "FAILED"


@dataclass(frozen=True, kw_only=True)
class StepResult:
    """Result of a single executed step within a plan."""

    kind: ComponentKind
    name: str
    status: StepStatus
    # Serialisable payload (or error message on failure)
    payload: dict[str, Any]
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind.value,
            "name": self.name,
            "status": self.status.value,
            "payload": self.payload,
            "error": self.error,
        }


@dataclass(frozen=True, kw_only=True)
class PlanExecutionResult:
    """Complete result of running an experiment plan.

    This is the result type returned by the experiment runner. It captures
    the outcome of executing all planned steps (detectors, change-points,
    regimes, consensus) against a dataset.

    Attributes
    ----------
    plan_summary : str
        Human-readable summary from the plan.
    step_results : tuple[StepResult, ...]
        Per-step results in execution order.
    consensus_result : StepResult | None
        Consensus step result if applicable and executed, else None.
    total_steps : int
        Total number of planned steps.
    successful_steps : int
        Number of steps that returned SUCCESS.
    schema_version : str
        Output format version ("0.5").
    """

    plan_summary: str
    step_results: tuple[StepResult, ...]
    consensus_result: StepResult | None
    total_steps: int
    successful_steps: int
    schema_version: str = "0.5"

    @property
    def all_succeeded(self) -> bool:
        return self.successful_steps == self.total_steps

    @property
    def any_failed(self) -> bool:
        return any(s.status == StepStatus.FAILED for s in self.step_results)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "plan_summary": self.plan_summary,
            "total_steps": self.total_steps,
            "successful_steps": self.successful_steps,
            "all_succeeded": self.all_succeeded,
            "step_results": [s.to_dict() for s in self.step_results],
            "consensus_result": (
                self.consensus_result.to_dict() if self.consensus_result else None
            ),
        }

    def to_toml(self) -> str:
        """Serialize PlanExecutionResult to a TOML-formatted string."""
        dict_repr = _make_toml_compatible(self.to_dict())
        return _dict_to_toml(dict_repr, indent="  ")


# ------------------------------------------------------------------ #
# TOML helpers
# ------------------------------------------------------------------ #


def _make_toml_compatible(obj: Any) -> Any:
    """Walk a dict and replace non-TOML types."""
    if isinstance(obj, dict):
        return {str(k): _make_toml_compatible(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_make_toml_compatible(item) for item in obj]
    if isinstance(obj, float) and not np.isfinite(obj):
        return None
    if isinstance(obj, (np.integer, np.floating)):
        return float(obj) if isinstance(obj, np.floating) else int(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, Enum):
        return obj.value
    return obj


# ------------------------------------------------------------------ #
# Runner
# ------------------------------------------------------------------ #


def run(
    plan: ExperimentPlan,
    frame: pd.DataFrame,
    output_path: Path | None = None,
    *,
    change_points_params: dict | None = None,
    regimes_params: dict | None = None,
    consensus_params: dict | None = None,
) -> PlanExecutionResult:
    """Execute an experiment plan against a dataset.

    Parameters
    ----------
    plan : ExperimentPlan
        Pre-computed applicability plan (from planner.plan).
    frame : pd.DataFrame
        Telemetry dataset to analyse.
    output_path : Path | None
        If provided, write TOML result to this path.
    change_points_params : dict | None
        Override parameters for the change-points step.
        Supported keys: feature (str), min_segment (int), threshold (float),
        order_by (str | None).
    regimes_params : dict | None
        Override parameters for the regimes step.
        Supported keys: feature (str), penalty (float | None), method (str),
        model (str), ordering_field (str | None).
    consensus_params : dict | None
        Override parameters for the consensus step.
        Supported keys: score_percentile (bool).

    Returns
    -------
    PlanExecutionResult
        Typed result containing per-step payloads and a TOML-compatible dict.

    Raises
    ------
    AnomalyLabError
        On unexpected internal errors (not on individual step failures,
        which are captured in StepResult payloads).
    """
    step_results: list[StepResult] = []
    consensus_result: StepResult | None = None
    successful_steps = 0

    detection_results: list[DetectionResult] = []

    for step in plan.steps:
        result = _execute_step(
            step,
            frame,
            plan.features,
            change_points_params=change_points_params,
            regimes_params=regimes_params,
            consensus_params=consensus_params,
            detection_results=detection_results,
        )
        step_results.append(result)
        if result.status == StepStatus.SUCCESS:
            successful_steps += 1
            if step.kind == ComponentKind.DETECTOR and result.payload.get("detector"):
                detection_results.append(_dict_to_detection_result(result.payload))

        if step.kind == ComponentKind.CONSENSUS:
            consensus_result = result

    plan_result = PlanExecutionResult(
        plan_summary=plan.summary,
        step_results=tuple(step_results),
        consensus_result=consensus_result,
        total_steps=len(plan.steps),
        successful_steps=successful_steps,
    )

    if output_path is not None:
        _write_toml(plan_result, output_path)

    return plan_result


def run_from_path(
    plan: ExperimentPlan,
    input_path: str | Path,
    output_path: Path | None = None,
    *,
    change_points_params: dict | None = None,
    regimes_params: dict | None = None,
    consensus_params: dict | None = None,
) -> PlanExecutionResult:
    """Convenience wrapper that loads the CSV before calling run."""
    frame = read_csv(input_path)
    return run(
        plan,
        frame,
        output_path,
        change_points_params=change_points_params,
        regimes_params=regimes_params,
        consensus_params=consensus_params,
    )


def _execute_step(
    step: PlannedStep,
    frame: pd.DataFrame,
    features: tuple[str, ...],
    *,
    change_points_params: dict | None,
    regimes_params: dict | None,
    consensus_params: dict | None,
    detection_results: list[DetectionResult],
) -> StepResult:
    """Execute a single PlannedStep and return a StepResult."""

    if step.kind == ComponentKind.DETECTOR:
        return _run_detector(step, frame, features)
    elif step.kind == ComponentKind.CHANGE_POINTS:
        return _run_change_points(step, frame, features, change_points_params or {})
    elif step.kind == ComponentKind.REGIMES:
        return _run_regimes(step, frame, features, regimes_params or {})
    elif step.kind == ComponentKind.CONSENSUS:
        return _run_consensus(step, detection_results, consensus_params or {})
    else:
        return StepResult(
            kind=step.kind,
            name=step.name,
            status=StepStatus.SKIPPED,
            payload={},
            error=f"Unknown component kind: {step.kind}",
        )


def _run_detector(
    step: PlannedStep, frame: pd.DataFrame, features: tuple[str, ...]
) -> StepResult:
    """Run a single detector and return a StepResult."""
    from anomaly_lab.detectors import (
        isolation_forest as iso_detector,
        dbscan as dbscan_detector,
        robust_zscore as robust_detector,
    )

    detector_name = step.name
    try:
        if detector_name == "isolation-forest":
            from anomaly_lab.detectors.isolation_forest import IsolationForestConfig

            cfg = step.parameters or {}
            config = IsolationForestConfig(
                contamination=cfg.get("contamination", "auto"),
                n_estimators=cfg.get("n_estimators", 200),
                random_state=cfg.get("random_state", 42),
            )
            result = iso_detector.fit_predict(frame, list(features), config)
        elif detector_name == "dbscan":
            from anomaly_lab.detectors.dbscan import DBSCANConfig

            cfg = step.parameters or {}
            config = DBSCANConfig(
                eps=cfg.get("eps", 0.5),
                min_samples=cfg.get("min_samples", 5),
            )
            result = dbscan_detector.fit_predict(
                frame, list(features), config, metadata_columns=cfg.get("metadata_columns", [])
            )
        elif detector_name == "robust-zscore":
            from anomaly_lab.detectors.robust_zscore import RobustZScoreConfig

            cfg = step.parameters or {}
            config = RobustZScoreConfig(threshold=cfg.get("threshold", 3.5))
            result = robust_detector.fit_predict(frame, list(features), config)
        else:
            raise ValueError(f"Unknown detector: {detector_name!r}")

        return StepResult(
            kind=ComponentKind.DETECTOR,
            name=detector_name,
            status=StepStatus.SUCCESS,
            payload=result.to_dict(),
        )
    except Exception as exc:
        return StepResult(
            kind=ComponentKind.DETECTOR,
            name=detector_name,
            status=StepStatus.FAILED,
            payload={},
            error=f"Detector {detector_name!r} failed: {exc}",
        )


def _run_change_points(
    step: PlannedStep,
    frame: pd.DataFrame,
    features: tuple[str, ...],
    params: dict,
) -> StepResult:
    """Run change-point detection and return a StepResult."""
    try:
        feature = params.get("feature", features[0] if features else None)
        if feature is None:
            raise ValueError("change-points requires at least one feature column")
        min_segment = int(params.get("min_segment", 2))
        threshold = float(params.get("threshold", 0.0))
        order_by = params.get("order_by")

        result = change_points_module.detect_change_points(
            frame,
            feature,
            min_segment=min_segment,
            threshold=threshold,
            order_by=order_by,
        )
        return StepResult(
            kind=ComponentKind.CHANGE_POINTS,
            name="change-points",
            status=StepStatus.SUCCESS,
            payload=result.to_dict(),
        )
    except Exception as exc:
        return StepResult(
            kind=ComponentKind.CHANGE_POINTS,
            name="change-points",
            status=StepStatus.FAILED,
            payload={},
            error=f"Change-points failed: {exc}",
        )


def _run_regimes(
    step: PlannedStep,
    frame: pd.DataFrame,
    features: tuple[str, ...],
    params: dict,
) -> StepResult:
    """Run regime detection and return a StepResult."""
    try:
        feature = params.get("feature", features[0] if features else None)
        if feature is None:
            raise ValueError("regimes requires at least one feature column")
        result = regimes_module.detect_regimes(
            frame,
            feature,
            ordering_field=params.get("ordering_field"),
            penalty=params.get("penalty"),
            method=params.get("method", "pelt"),
            model=params.get("model", "l2"),
        )
        return StepResult(
            kind=ComponentKind.REGIMES,
            name="regimes",
            status=StepStatus.SUCCESS,
            payload=result.to_dict(),
        )
    except Exception as exc:
        return StepResult(
            kind=ComponentKind.REGIMES,
            name="regimes",
            status=StepStatus.FAILED,
            payload={},
            error=f"Regimes failed: {exc}",
        )


def _run_consensus(
    step: PlannedStep,
    detection_results: list[DetectionResult],
    params: dict,
) -> StepResult:
    """Run consensus computation and return a StepResult."""
    try:
        if len(detection_results) < 2:
            return StepResult(
                kind=ComponentKind.CONSENSUS,
                name="consensus",
                status=StepStatus.SKIPPED,
                payload={},
                error="Consensus requires at least 2 detection results; fewer were produced",
            )
        score_percentile = params.get("score_percentile", True)
        result = consensus_module.compute_consensus(
            detection_results, score_percentile=score_percentile
        )
        return StepResult(
            kind=ComponentKind.CONSENSUS,
            name="consensus",
            status=StepStatus.SUCCESS,
            payload=result.to_dict(),
        )
    except Exception as exc:
        return StepResult(
            kind=ComponentKind.CONSENSUS,
            name="consensus",
            status=StepStatus.FAILED,
            payload={},
            error=f"Consensus failed: {exc}",
        )


def _dict_to_detection_result(d: dict[str, Any]) -> DetectionResult:
    """Reconstruct a DetectionResult from its to_dict() representation."""
    return DetectionResult(
        detector=d["detector"],
        row_indices=tuple(r["index"] for r in d["rows"]),
        is_anomaly=tuple(r["is_anomaly"] for r in d["rows"]),
        features=tuple(d["features"]),
        constant_features=tuple(d.get("constant_features", [])),
        parameters=d.get("parameters", {}),
        scores=tuple(r["score"] for r in d["rows"]) if d.get("scores") else None,
        clusters=tuple(r["cluster"] for r in d["rows"]) if "clusters" in d else None,
        metadata=d.get("metadata"),
        schema_version=d.get("schema_version", "0.4"),
        dataset_semantics=d.get("dataset_semantics"),
        evidence_class=d.get("evidence_class", ""),
        confidence=d.get("confidence"),
        applicability=d.get("applicability", "APPLICABLE"),
    )


def _write_toml(result: PlanExecutionResult, path: Path) -> None:
    """Write PlanExecutionResult to a TOML file."""
    toml_str = result.to_toml()
    path.write_text(toml_str, encoding="utf-8")


def _dict_to_toml(obj: Any, indent: str = "  ") -> str:
    """Simple dict-to-TOML serializer for PlanExecutionResult structure."""
    lines: list[str] = []
    _write_value(lines, obj, "", indent)
    return "\n".join(lines) + "\n"


def _write_value(lines: list[str], obj: Any, prefix: str, indent: str) -> None:
    """Recursively write a Python value as TOML."""
    if obj is None:
        lines.append(f"{prefix}= null")
    elif isinstance(obj, bool):
        lines.append(f"{prefix}= {'true' if obj else 'false'}")
    elif isinstance(obj, (int, float, str)):
        if isinstance(obj, float):
            lines.append(f"{prefix}= {obj}")
        else:
            lines.append(f"{prefix}= {obj}")
    elif isinstance(obj, dict):
        if prefix:
            lines.append(f"{prefix}= {{")
            for k, v in obj.items():
                _write_value(lines, v, f"{indent}{k}", indent)
            lines.append(f"{indent}}}")
        else:
            for k, v in obj.items():
                _write_value(lines, v, str(k), indent)
    elif isinstance(obj, (list, tuple)):
        if not obj:
            lines.append(f"{prefix}= []")
            return
        # Check if it's a list of dicts (table array)
        if all(isinstance(item, dict) for item in obj):
            for i, item in enumerate(obj):
                lines.append(f"[[{prefix}]]")
                for k, v in item.items():
                    _write_value(lines, v, f"{indent}{k}", indent)
        else:
            # Inline array
            items = []
            for item in obj:
                if item is None:
                    items.append("null")
                elif isinstance(item, bool):
                    items.append("true" if item else "false")
                elif isinstance(item, (int, float, str)):
                    items.append(repr(item))
                else:
                    items.append(str(item))
            lines.append(f"{prefix}= [{', '.join(items)}]")
    else:
        lines.append(f"{prefix}= {repr(obj)}")
