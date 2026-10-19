from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd

from anomaly_lab.schema import (
    EmptyDataError,
    FeatureSelectionError,
    InvalidDetectorConfigError,
    NonFiniteValueError,
    NonNumericFeatureError,
)


@dataclass(frozen=True)
class ChangePointCandidate:
    split_position: int
    left_median: float
    right_median: float
    median_shift: float


@dataclass(frozen=True)
class ChangePointResult:
    feature: str
    candidates: tuple[ChangePointCandidate, ...]
    min_segment: int
    threshold: float
    order_by: str | None
    retrospective: bool = True
    interpretation: str = "run boundary candidate; root cause is not established"

    def to_dict(self) -> dict[str, Any]:
        return {
            "feature": self.feature,
            "candidates": [asdict(candidate) for candidate in self.candidates],
            "min_segment": self.min_segment,
            "threshold": self.threshold,
            "order_by": self.order_by,
            "retrospective": self.retrospective,
            "interpretation": self.interpretation,
        }


def detect_change_points(
    frame: pd.DataFrame,
    feature: str,
    *,
    min_segment: int,
    threshold: float,
    order_by: str | None = None,
) -> ChangePointResult:
    """Scan all eligible splits using absolute two-sided median shift, retrospectively."""
    if (
        isinstance(min_segment, (bool, np.bool_))
        or not isinstance(min_segment, (int, np.integer))
        or min_segment < 2
    ):
        raise InvalidDetectorConfigError("min_segment must be an integer of at least 2")
    if (
        isinstance(threshold, (bool, np.bool_))
        or not isinstance(threshold, (int, float, np.integer, np.floating))
        or not np.isfinite(threshold)
        or threshold < 0
    ):
        raise InvalidDetectorConfigError("threshold must be a finite non-negative number")
    if not isinstance(frame, pd.DataFrame):
        raise FeatureSelectionError("input must be a pandas DataFrame")
    if frame.empty:
        raise EmptyDataError("input dataset has zero rows")
    if order_by is not None:
        if order_by not in frame.columns:
            raise FeatureSelectionError(f"missing order column: {order_by}")
        order = frame[order_by]
        invalid_order = (
            bool(order.isna().any())
            or bool(order.duplicated().any())
            or not bool(order.is_monotonic_increasing)
        )
        if invalid_order:
            raise FeatureSelectionError("order column must be unique and strictly increasing")
    if feature not in frame.columns:
        raise FeatureSelectionError(f"missing feature: {feature}")
    series = frame[feature]
    if not pd.api.types.is_numeric_dtype(series.dtype) or pd.api.types.is_bool_dtype(series.dtype):
        raise NonNumericFeatureError(f"feature {feature!r} must have numeric non-boolean dtype")
    values = series.to_numpy(dtype=np.float64, copy=True)
    if not np.isfinite(values).all():
        raise NonFiniteValueError(f"feature {feature!r} contains non-finite values")
    candidates = []
    for split in range(min_segment, len(values) - min_segment + 1):
        left = float(np.median(values[:split]))
        right = float(np.median(values[split:]))
        shift = abs(right - left)
        if not all(np.isfinite(v) for v in (left, right, shift)):
            raise NonFiniteValueError("median change-point statistic is non-finite")
        if shift >= threshold:
            candidates.append(ChangePointCandidate(split, left, right, shift))
    candidates.sort(key=lambda item: (-item.median_shift, item.split_position))
    return ChangePointResult(
        feature, tuple(candidates), int(min_segment), float(threshold), order_by
    )
