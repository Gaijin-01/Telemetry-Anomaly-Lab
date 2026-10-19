from dataclasses import dataclass
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
    """Retrospective median-shift candidate at a given split boundary."""

    split_position: int  # row index where the right segment begins
    left_median: float  # median of the left segment
    right_median: float  # median of the right segment
    median_shift: float  # |right_median - left_median|
    left_size: int  # number of rows in left segment
    right_size: int  # number of rows in right segment
    support: int  # min(left_size, right_size) — balanced splits preferred

    # raw_sort_key records the original shift before ranking tie-breaking.
    # It is not serialized; use median_shift for the observed shift magnitude.
    raw_sort_key: float = 0.0


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
            "candidates": [
                {
                    "split_position": c.split_position,
                    "left_median": float(c.left_median),
                    "right_median": float(c.right_median),
                    "median_shift": float(c.median_shift),
                    "left_size": c.left_size,
                    "right_size": c.right_size,
                    "support": c.support,
                }
                for c in self.candidates
            ],
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
    """Scan all eligible splits using absolute two-sided median shift, retrospectively.

    Candidates are ranked by:
      1. median_shift (descending) — larger shifts are stronger evidence
      2. support (descending)       — more balanced splits win ties
      3. split_position (ascending) — earlier boundary preferred when both tie

    A tiny edge split must not win a tie over a strongly supported central
    boundary merely because both have the same raw median shift.
    """
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
    total = len(values)
    for split in range(min_segment, total - min_segment + 1):
        left = values[:split]
        right = values[split:]
        left_median = float(np.median(left))
        right_median = float(np.median(right))
        shift = abs(right_median - left_median)
        if not all(np.isfinite(v) for v in (left_median, right_median, shift)):
            raise NonFiniteValueError("median change-point statistic is non-finite")
        if shift >= threshold:
            left_size = split
            right_size = total - split
            support = min(left_size, right_size)
            candidates.append(
                ChangePointCandidate(
                    split,
                    left_median,
                    right_median,
                    shift,
                    left_size,
                    right_size,
                    support,
                    raw_sort_key=shift,
                )
            )
    # Primary: larger shift. Tie-break: larger support. Further tie: earlier position.
    candidates.sort(key=lambda item: (-item.median_shift, -item.support, item.split_position))
    return ChangePointResult(
        feature, tuple(candidates), int(min_segment), float(threshold), order_by
    )
