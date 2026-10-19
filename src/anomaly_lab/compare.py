from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Literal

import pandas as pd

from anomaly_lab.schema import AnomalyLabError


@dataclass(frozen=True)
class RunMetadata:
    """Explicit run identity used only for comparability, never as a metric."""

    model: str
    version: str
    quantization: str
    concurrency: str
    allow_difference: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        for field, value in asdict(self).items():
            if field == "allow_difference":
                continue  # validated below
            if not isinstance(value, str) or not value.strip():
                raise AnomalyLabError("run metadata values must be non-empty strings")
        known = frozenset(k for k in asdict(self) if k != "allow_difference")
        for dim in self.allow_difference:
            if not isinstance(dim, str) or not dim.strip():
                raise AnomalyLabError("allow_difference values must be non-empty strings")
            if dim not in known:
                raise AnomalyLabError(f"allow_difference references unknown dimension: {dim!r}")


@dataclass(frozen=True)
class MetricRule:
    """Threshold is a relative-change proportion: 0.10 means 10%."""

    name: str
    direction: Literal["higher", "lower"]
    threshold: float

    def __post_init__(self) -> None:
        if (
            not isinstance(self.name, str)
            or not self.name.strip()
            or self.direction not in ("higher", "lower")
            or isinstance(self.threshold, bool)
            or not isinstance(self.threshold, (int, float))
            or not math.isfinite(self.threshold)
            or self.threshold < 0
        ):
            raise AnomalyLabError(
                "metric requires a name, direction higher/lower, and finite "
                "non-negative relative threshold"
            )


@dataclass(frozen=True)
class MetricComparison:
    name: str
    baseline: float
    candidate: float
    delta: float
    relative_change: float | None
    direction: str
    threshold: float
    threshold_kind: str
    status: str


@dataclass(frozen=True)
class ComparisonResult:
    status: str
    metrics: tuple[MetricComparison, ...]
    mismatches: tuple[str, ...]
    warnings: tuple[str, ...]
    causality: str = "NOT ESTABLISHED"

    def to_dict(self) -> dict:
        return asdict(self)


def _finite_mean(series: pd.Series, feature: str, label: str) -> float:
    # Scale before summing so finite values near float limits do not overflow.
    values = [float(value) for value in series]
    scale = max(abs(value) for value in values)
    mean = 0.0 if scale == 0 else math.fsum(value / scale for value in values) / len(values) * scale
    if not math.isfinite(mean):
        raise AnomalyLabError(
            f"computed mean for feature {feature!r} in {label} is not finite; rescale input values"
        )
    return mean


def _finite_difference(after: float, before: float, feature: str) -> float:
    difference = after - before
    if not math.isfinite(difference):
        raise AnomalyLabError(
            f"computed delta for feature {feature!r} is not finite; rescale input values"
        )
    return difference


def compare_runs(
    baseline: pd.DataFrame,
    candidate: pd.DataFrame,
    baseline_metadata: RunMetadata,
    candidate_metadata: RunMetadata,
    features: list[str],
    rules: list[MetricRule],
    min_samples: int = 2,
) -> ComparisonResult:
    if not isinstance(baseline, pd.DataFrame) or not isinstance(candidate, pd.DataFrame):
        raise AnomalyLabError("baseline and candidate must be DataFrames")
    if not isinstance(baseline_metadata, RunMetadata) or not isinstance(
        candidate_metadata, RunMetadata
    ):
        raise AnomalyLabError("baseline and candidate metadata must be RunMetadata")
    if (
        not isinstance(features, list)
        or not features
        or any(not isinstance(f, str) or not f.strip() for f in features)
    ):
        raise AnomalyLabError("features must be a non-empty list of names")
    if not isinstance(rules, list) or any(not isinstance(rule, MetricRule) for rule in rules):
        raise AnomalyLabError("rules must be a list of MetricRule values")
    if isinstance(min_samples, bool) or not isinstance(min_samples, int) or min_samples < 1:
        raise AnomalyLabError("min_samples must be a positive integer")
    mismatches = tuple(
        key
        for key in asdict(baseline_metadata)
        if getattr(baseline_metadata, key) != getattr(candidate_metadata, key)
        and key not in baseline_metadata.allow_difference
    )
    if mismatches:
        return ComparisonResult("NOT_COMPARABLE", (), mismatches, ())
    if (
        len(set(features)) != len(features)
        or len(rules) != len(features)
        or set(features) != {rule.name for rule in rules}
    ):
        raise AnomalyLabError("features must have exactly one metric rule each")
    missing = (set(features) - set(baseline.columns)) | (set(features) - set(candidate.columns))
    if missing:
        raise AnomalyLabError(f"missing comparison feature(s): {', '.join(sorted(missing))}")
    numeric: dict[tuple[str, str], pd.Series] = {}
    for label, frame in (("baseline", baseline), ("candidate", candidate)):
        if frame.empty:
            raise AnomalyLabError(f"feature data in {label} is empty")
        for name in features:
            series = frame[name]
            if not pd.api.types.is_numeric_dtype(series.dtype):
                raise AnomalyLabError(f"feature {name!r} in {label} must be numeric")
            if series.empty or not bool(series.map(math.isfinite).all()):
                raise AnomalyLabError(f"feature {name!r} in {label} must be non-empty and finite")
            numeric[(label, name)] = series
    rules_by_name = {rule.name: rule for rule in rules}
    if len(baseline) < min_samples or len(candidate) < min_samples:
        metrics = []
        for name in features:
            before = _finite_mean(numeric[("baseline", name)], name, "baseline")
            after = _finite_mean(numeric[("candidate", name)], name, "candidate")
            delta = _finite_difference(after, before, name)
            relative_change = None if before == 0 else delta / abs(before)
            if relative_change is not None and not math.isfinite(relative_change):
                raise AnomalyLabError(
                    f"computed relative change for feature {name!r} is not finite; "
                    "rescale input values"
                )
            metrics.append(
                MetricComparison(
                    name,
                    before,
                    after,
                    delta,
                    relative_change,
                    rules_by_name[name].direction,
                    rules_by_name[name].threshold,
                    "relative_change_proportion",
                    "INSUFFICIENT_SAMPLE",
                )
            )
        return ComparisonResult(
            "INSUFFICIENT_SAMPLE",
            tuple(metrics),
            (),
            (f"Small sample: each run requires at least {min_samples} rows.",),
        )
    metrics = []
    for name in features:
        rule = rules_by_name[name]
        before = _finite_mean(numeric[("baseline", name)], name, "baseline")
        after = _finite_mean(numeric[("candidate", name)], name, "candidate")
        delta = _finite_difference(after, before, name)
        relative_change = None if before == 0 else delta / abs(before)
        if relative_change is not None and not math.isfinite(relative_change):
            raise AnomalyLabError(
                f"computed relative change for feature {name!r} is not finite; rescale input values"
            )
        regression = relative_change is not None and (
            relative_change < -rule.threshold
            if rule.direction == "higher"
            else relative_change > rule.threshold
        )
        metrics.append(
            MetricComparison(
                name,
                before,
                after,
                delta,
                relative_change,
                rule.direction,
                rule.threshold,
                "relative_change_proportion",
                "REGRESSION" if regression else "NO_REGRESSION",
            )
        )
    status = (
        "REGRESSION"
        if any(metric.status == "REGRESSION" for metric in metrics)
        else "NO_REGRESSION"
    )
    return ComparisonResult(status, tuple(metrics), (), ())
