"""Applicability planner for anomaly-lab experiments.

Given a dataset and feature selection, the planner determines which analysis
components are applicable and returns a structured execution plan.  The plan
is consumed by the experiment runner — planner never executes analysis itself.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING

import pandas as pd

from anomaly_lab.registry import (
    ALL_DETECTOR_CONTRACTS,
    ApplicabilityCode,
    DetectorContract,
    check_applicability,
)
from anomaly_lab.semantics import DatasetProfile, classify

if TYPE_CHECKING:
    pass


# ------------------------------------------------------------------ #
# Component kinds
# ------------------------------------------------------------------ #


class ComponentKind(Enum):
    DETECTOR = "detector"
    CHANGE_POINTS = "change_points"
    REGIMES = "regimes"
    CONSENSUS = "consensus"


# ------------------------------------------------------------------ #
# Applicability record
# ------------------------------------------------------------------ #


@dataclass(frozen=True)
class ApplicabilityRecord:
    kind: ComponentKind
    name: str  # e.g. "isolation-forest", "robust-zscore"
    code: ApplicabilityCode
    reason: str
    contract: DetectorContract | None = None  # None for non-detector components


# ------------------------------------------------------------------ #
# Planned step
# ------------------------------------------------------------------ #


@dataclass(frozen=True)
class PlannedStep:
    """A single executable step within an experiment plan."""

    kind: ComponentKind
    name: str
    # Human-readable reason for inclusion
    rationale: str
    # Default/configured parameters for this step (not validated here)
    parameters: dict = field(default_factory=dict)


# ------------------------------------------------------------------ #
# Experiment plan
# ------------------------------------------------------------------ #


@dataclass(frozen=True)
class ExperimentPlan:
    """Complete applicability-driven experiment plan for one dataset."""

    # Dataset profile used to make applicability decisions
    profile: DatasetProfile

    # All applicability records (applicable + not applicable)
    applicability_records: tuple[ApplicabilityRecord, ...]

    # Steps to execute, in recommended order
    steps: tuple[PlannedStep, ...]

    # Detector names that passed applicability checks
    applicable_detectors: tuple[str, ...] = field(default_factory=lambda: ())

    # Feature columns the plan was built for
    features: tuple[str, ...] = field(default_factory=lambda: ())

    # Number of rows in the input dataset
    n_samples: int = 0

    # Whether consensus is viable (2+ applicable detectors)
    consensus_recommended: bool = False

    # Summary string for logging / UI
    summary: str = ""

    def to_dict(self) -> dict:
        return {
            "profile": {
                "semantics": self.profile.semantics.value,
                "has_timestamp": self.profile.has_timestamp,
                "has_group_key": self.profile.has_group_key,
                "has_pairing_key": self.profile.has_pairing_key,
                "metric_columns": list(self.profile.metric_columns),
                "ordering_field": self.profile.ordering_field,
                "n_samples": self.n_samples,
            },
            "applicable_detectors": list(self.applicable_detectors),
            "consensus_recommended": self.consensus_recommended,
            "steps": [
                {
                    "kind": s.kind.value,
                    "name": s.name,
                    "rationale": s.rationale,
                    "parameters": s.parameters,
                }
                for s in self.steps
            ],
            "applicability_records": [
                {
                    "kind": r.kind.value,
                    "name": r.name,
                    "code": r.code.value,
                    "reason": r.reason,
                }
                for r in self.applicability_records
            ],
            "summary": self.summary,
        }


# ------------------------------------------------------------------ #
# Planning logic
# ------------------------------------------------------------------ #


def plan(
    frame: pd.DataFrame,
    features: list[str] | tuple[str, ...],
    *,
    include_detectors: bool = True,
    include_change_points: bool = True,
    include_regimes: bool = False,
    detector_configs: dict | None = None,
) -> ExperimentPlan:
    """Determine which anomaly-lab components are applicable to a dataset.

    Parameters
    ----------
    frame : pd.DataFrame
        Telemetry dataset to plan analysis for.
    features : list[str] | tuple[str, ...]
        Feature columns to analyse.
    include_detectors : bool
        If True (default), check all registered detectors for applicability.
    include_change_points : bool
        If True (default), include change-point scanning when data is ordered
        or has a timestamp and has at least 10 rows.
    include_regimes : bool
        If True, include regime detection when data has at least 10 rows.
        Default is False because Pelt is computationally expensive.
    detector_configs : dict | None
        Optional extra parameters passed to detector step planning,
        e.g. {"isolation-forest": {"contamination": 0.1}}.

    Returns
    -------
    ExperimentPlan
        Frozen plan containing applicability decisions and ordered steps.

    Raises
    ------
    AnomalyLabError
        If the dataset is empty or features are invalid.
    """
    from anomaly_lab.schema import AnomalyLabError

    if frame.empty:
        raise AnomalyLabError("plan requires a non-empty DataFrame")
    if not features:
        raise AnomalyLabError("plan requires at least one feature")

    n_samples = len(frame)
    profile = classify(frame)

    # Compute constant features from prepare_features (imported lazily)
    try:
        from anomaly_lab.features import prepare_features

        prepared = prepare_features(frame, features)
        constant_features = prepared.constant_features
    except Exception:
        # If feature preparation fails, defer to empty tuple; the detector
        # step itself will raise a more specific error when executed.
        constant_features = ()

    has_missing = bool(frame.isnull().any().any())

    # --- Build applicability records for detectors ---
    records: list[ApplicabilityRecord] = []
    applicable_detector_names: list[str] = []

    if include_detectors:
        for name, contract in ALL_DETECTOR_CONTRACTS.items():
            code, reason = check_applicability(
                contract,
                profile,
                n_samples,
                constant_features,
                has_missing,
            )
            records.append(
                ApplicabilityRecord(
                    kind=ComponentKind.DETECTOR,
                    name=name,
                    code=code,
                    reason=reason,
                    contract=contract,
                )
            )
            if code == ApplicabilityCode.APPLICABLE:
                applicable_detector_names.append(name)

    # --- Change-points applicability ---
    if include_change_points:
        cp_reason = _change_points_applicable_reason(profile, n_samples)
        cp_code = (
            ApplicabilityCode.APPLICABLE if cp_reason is None else ApplicabilityCode.NOT_APPLICABLE
        )
        records.append(
            ApplicabilityRecord(
                kind=ComponentKind.CHANGE_POINTS,
                name="change-points",
                code=cp_code,
                reason=cp_reason or "change-points scan is applicable",
            )
        )

    # --- Regimes applicability ---
    if include_regimes:
        reg_reason = _regimes_applicable_reason(n_samples)
        reg_code = (
            ApplicabilityCode.APPLICABLE if reg_reason is None else ApplicabilityCode.NOT_APPLICABLE
        )
        records.append(
            ApplicabilityRecord(
                kind=ComponentKind.REGIMES,
                name="regimes",
                code=reg_code,
                reason=reg_reason or "regime detection is applicable",
            )
        )

    # --- Build ordered step list ---
    steps: list[PlannedStep] = []
    detector_configs = detector_configs or {}

    # 1. Always run applicable detectors first
    for name in applicable_detector_names:
        steps.append(
            PlannedStep(
                kind=ComponentKind.DETECTOR,
                name=name,
                rationale=f"Detector {name!r} passed applicability checks",
                parameters=detector_configs.get(name, {}),
            )
        )

    # 2. Change-points scan (if applicable)
    if include_change_points and (profile.ordering_field is not None or profile.has_timestamp):
        steps.append(
            PlannedStep(
                kind=ComponentKind.CHANGE_POINTS,
                name="change-points",
                rationale="Dataset has temporal ordering; change-point scan is applicable",
                parameters={},
            )
        )

    # 3. Regimes (if applicable and requested)
    if include_regimes and n_samples >= 10:
        steps.append(
            PlannedStep(
                kind=ComponentKind.REGIMES,
                name="regimes",
                rationale="Dataset has sufficient rows for regime detection",
                parameters={},
            )
        )

    # 4. Consensus (if 2+ detectors passed)
    consensus_recommended = len(applicable_detector_names) >= 2
    if consensus_recommended:
        steps.append(
            PlannedStep(
                kind=ComponentKind.CONSENSUS,
                name="consensus",
                rationale=(
                    f"{len(applicable_detector_names)} detectors passed; "
                    "consensus computation is recommended"
                ),
                parameters={},
            )
        )

    # --- Build summary ---
    summary_parts = [
        f"samples={n_samples}",
        f"semantics={profile.semantics.value}",
        f"applicable_detectors={len(applicable_detector_names)}/{len(ALL_DETECTOR_CONTRACTS)}",
    ]
    if consensus_recommended:
        summary_parts.append("consensus=recommended")
    if profile.ordering_field:
        summary_parts.append(f"ordered_by={profile.ordering_field}")
    summary = "; ".join(summary_parts)

    return ExperimentPlan(
        profile=profile,
        applicability_records=tuple(records),
        steps=tuple(steps),
        applicable_detectors=tuple(applicable_detector_names),
        features=tuple(features),
        n_samples=n_samples,
        consensus_recommended=consensus_recommended,
        summary=summary,
    )


def _change_points_applicable_reason(profile: DatasetProfile, n_samples: int) -> str | None:
    """Return None if applicable, or an English reason if not."""
    if n_samples < 10:
        return (
            f"Dataset has only {n_samples} rows; "
            "change-point detection requires at least 10 rows for min_segment=2"
        )
    if profile.ordering_field is None and not profile.has_timestamp:
        return (
            "No ordering field or timestamp detected; "
            "change-point scan requires a strictly increasing order column"
        )
    return None


def _regimes_applicable_reason(n_samples: int) -> str | None:
    """Return None if applicable, or an English reason if not."""
    if n_samples < 10:
        return (
            f"Dataset has only {n_samples} rows; "
            "regime detection requires at least 10 rows (ruptures min series length)"
        )
    return None
