"""Experiment runner for anomaly-lab.

Executes an applicability plan by orchestrating the existing deterministic
components (detectors, change_points, regimes, consensus).  The runner does
NOT reimplement detector logic — it delegates to the existing module-level
functions in anomaly_lab.detectors.*, anomaly_lab.change_points, etc.

Output is TOML only, per project requirement.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from anomaly_lab import change_points as change_points_module
from anomaly_lab import consensus as consensus_module
from anomaly_lab import regimes as regimes_module
from anomaly_lab.experiments.planner import ComponentKind, ExperimentPlan, PlannedStep
from anomaly_lab.ingest import read_csv
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
        return _dict_to_toml(dict_repr)


# ------------------------------------------------------------------ #
# TOML helpers
# ------------------------------------------------------------------ #


def _make_toml_compatible(obj: Any) -> Any:
    """Walk a dict and replace non-TOML types with TOML-compatible equivalents."""
    if isinstance(obj, dict):
        return {str(k): _make_toml_compatible(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        # Convert list/tuple of 2-tuples (key-value pairs) to dict
        if obj and all(isinstance(it, (list, tuple)) and len(it) == 2 for it in obj):
            return {str(k): _make_toml_compatible(v) for k, v in obj}
        return [_make_toml_compatible(item) for item in obj]
    if isinstance(obj, float) and not np.isfinite(obj):
        return None
    if isinstance(obj, (np.integer, np.floating)):
        return float(obj) if isinstance(obj, np.floating) else int(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return [_make_toml_compatible(item) for item in obj.tolist()]
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


def _run_detector(step: PlannedStep, frame: pd.DataFrame, features: tuple[str, ...]) -> StepResult:
    """Run a single detector and return a StepResult."""
    from anomaly_lab.detectors import (
        dbscan as dbscan_detector,
    )
    from anomaly_lab.detectors import (
        isolation_forest as iso_detector,
    )
    from anomaly_lab.detectors import (
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
        # Filter to only detectors that produce scores (DBSCAN does not)
        scoring_results = [r for r in detection_results if r.scores is not None]
        if len(scoring_results) < 2:
            return StepResult(
                kind=ComponentKind.CONSENSUS,
                name="consensus",
                status=StepStatus.SKIPPED,
                payload={},
                error=(
                    f"Consensus requires at least 2 scoring detectors; "
                    f"got {len(scoring_results)} (detectors without scores are excluded)"
                ),
            )
        score_percentile = params.get("score_percentile", True)
        result = consensus_module.compute_consensus(
            scoring_results, score_percentile=score_percentile
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
    # Scores are embedded in rows (from DetectionResult.to_dict())
    scores = tuple(r["score"] for r in d["rows"]) if d["rows"] else None
    # Only set scores if at least one row has a non-None score
    if scores and any(s is not None for s in scores):
        scores = scores
    else:
        scores = None

    return DetectionResult(
        detector=d["detector"],
        row_indices=tuple(r["index"] for r in d["rows"]),
        is_anomaly=tuple(r["is_anomaly"] for r in d["rows"]),
        features=tuple(d["features"]),
        constant_features=tuple(d.get("constant_features", [])),
        parameters=d.get("parameters", {}),
        scores=scores,
        clusters=tuple(r["cluster"] for r in d["rows"]) if "clusters" in d else None,
        metadata=d.get("metadata"),
        schema_version=d.get("schema_version", "0.4"),
        dataset_semantics=d.get("dataset_semantics"),
        evidence_class=d.get("evidence_class", ""),
        confidence=d.get("confidence"),
        applicability=d.get("applicability", "APPLICABLE"),
    )


def _toml_escape(s):
    """Escape string content for TOML value context. Returns raw escaped (no outer quotes)."""
    if not isinstance(s, str):
        s = str(s)
    # Escape special characters
    s = s.replace("\\", "\\\\")
    s = s.replace('"', '\\"')
    s = s.replace("\n", "\\n")
    s = s.replace("\r", "\\r")
    s = s.replace("\t", "\\t")
    return s


def _fmt_val(v):
    """Format a scalar value as TOML literal. Returns None to signal 'skip this key'."""
    if v is None:
        return None
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, int):
        return str(v)
    if isinstance(v, float):
        if not np.isfinite(v):
            return None
        return repr(v)
    if isinstance(v, str):
        return f'"{_toml_escape(v)}"'
    if isinstance(v, (list, tuple)) and v and all(_is_scalar(x) for x in v):
        items = ", ".join(_fmt_val(x) for x in v)
        return f"[{items}]"
    return repr(v)


def _is_scalar(v):
    return v is None or isinstance(v, (bool, int, float, str))


def _dict_to_toml(obj, _indent="  "):
    """Serialize dict to valid TOML 1.0 using only stdlib."""
    out = []

    def emit_table(d, prefix=""):
        # Separate scalar keys, inline-table-able dicts, and real sub-tables
        scalars = {}
        subtables = {}
        inline = {}
        for k, v in d.items():
            if _is_scalar(v):
                scalars[k] = v
            elif isinstance(v, dict) and v:
                if all(_is_scalar(sv) for sv in v.values()):
                    inline[k] = v
                else:
                    subtables[k] = v
            else:
                scalars[k] = v

        # Emit scalar keys
        for k, v in scalars.items():
            fv = _fmt_val(v)
            if fv is not None:
                out.append(f"{_indent}{_toml_escape(k)} = {fv}")

        # Emit inline dicts
        for k, v in inline.items():

            def _make_pairs(d):
                result = []
                for sk, sv in d.items():
                    fv = _fmt_val(sv)
                    if fv is not None:
                        result.append(f'"{_toml_escape(sk)}" = {fv}')
                return result

            pairs = ", ".join(_make_pairs(v))
            out.append(f"{_toml_escape(k)} = {{{pairs}}}")

        # Emit subtables (nested [key] sections)
        for k, v in subtables.items():
            section = f"[{prefix}{_toml_escape(k)}]" if prefix else f"[{_toml_escape(k)}]"
            out.append(section)
            emit_table(v, prefix=f"{prefix}{k}.")

    def emit_aot(key, items):
        """Emit array-of-tables for a list of dicts."""
        for item in items:
            out.append(f"[[{key}]]")
            emit_aot_dict(item, f"{key}.")

    def emit_aot_dict(d, prefix=""):
        """Emit key=value for a dict inside AoT context."""
        # Separate inline dicts and subtables
        scalars = {}
        inline_dicts = {}
        subtables = {}
        list_of_dicts = {}
        for k, v in d.items():
            if _is_scalar(v):
                scalars[k] = v
            elif isinstance(v, dict) and v:
                if all(_is_scalar(sv) for sv in v.values()):
                    inline_dicts[k] = v
                else:
                    subtables[k] = v
            elif isinstance(v, list) and v and all(isinstance(x, dict) for x in v):
                # Use inline array of inline tables for dicts-as-values (not AoT)
                list_of_dicts[k] = v
            elif isinstance(v, list) and v and all(_is_scalar(x) for x in v):
                scalars[k] = v
            else:
                scalars[k] = v
        # Scalars at ROOT level of AoT (no indent), then blank line, then tables
        for k, v in scalars.items():
            fv = _fmt_val(v)
            if fv is not None:
                out.append(f"{_toml_escape(k)} = {fv}")
        if scalars:
            out.append("")
        for k, v in list_of_dicts.items():
            # Inline array of inline tables: rows = [{key = val, ...}, ...]
            entries = []
            for item in v:

                def _make_fields(d):
                    result = []
                    for ik, iv in d.items():
                        fv = _fmt_val(iv)
                        if fv is not None:
                            result.append(f"{_toml_escape(ik)} = {fv}")
                    return result

                fields = ", ".join(_make_fields(item))
                entries.append(f"{{{fields}}}")
            sep = ", "
            out.append(f"{_indent}{_toml_escape(k)} = [{sep.join(entries)}]")
        for k, v in inline_dicts.items():

            def _make_pairs(d):
                result = []
                for sk, sv in d.items():
                    fv = _fmt_val(sv)
                    if fv is not None:
                        result.append(f'"{_toml_escape(sk)}" = {fv}')
                return result

            pairs = ", ".join(_make_pairs(v))
            out.append(f"{_indent}{_toml_escape(k)} = {{{pairs}}}")
        for k, v in subtables.items():
            out.append(f"[{prefix}{_toml_escape(k)}]")
            emit_aot_dict(v, f"{prefix}{k}.")

    # Top-level dispatch
    for k, v in obj.items():
        if _is_scalar(v):
            fv = _fmt_val(v)
            if fv is not None:
                out.append(f"{_toml_escape(k)} = {fv}")
        elif isinstance(v, dict) and v:
            if all(_is_scalar(sv) for sv in v.values()):

                def _make_pairs_toplevel(d):
                    result = []
                    for sk, sv in d.items():
                        fv = _fmt_val(sv)
                        if fv is not None:
                            result.append(f'"{_toml_escape(sk)}" = {fv}')
                    return result

                pairs = ", ".join(_make_pairs_toplevel(v))
                out.append(f"{_toml_escape(k)} = {{{pairs}}}")
            else:
                out.append(f"[{_toml_escape(k)}]")
                emit_table(v, f"{k}.")
        elif isinstance(v, list) and v:
            if all(_is_scalar(x) for x in v):
                items = ", ".join(str(fv) for x in v for fv in [_fmt_val(x)] if fv is not None)
                out.append(f"{_toml_escape(k)} = [{items}]")
            elif all(isinstance(x, dict) for x in v):
                emit_aot(k, v)
            else:
                for x in v:
                    fv = _fmt_val(x)
                    if fv is not None:
                        out.append(f"{_indent}{fv}")
        else:
            fv = _fmt_val(v)
            if fv is not None:
                out.append(f"{_toml_escape(k)} = {fv}")
    return "\n".join(out)


def _toml_string_value(s):
    return _toml_escape(s)


def _write_toml(result: PlanExecutionResult, path: Path) -> None:
    """Write plan execution result to a TOML file."""
    d = result.to_dict()
    mc = _make_toml_compatible(d)
    path.write_text(_dict_to_toml(mc))
