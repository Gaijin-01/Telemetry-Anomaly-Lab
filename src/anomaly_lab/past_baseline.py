from dataclasses import dataclass
from typing import Literal

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
class PastBaselineResult:
    method: str
    baseline: pd.DataFrame
    residual: pd.DataFrame
    order_by: str | None


def compute_past_baseline(
    frame: pd.DataFrame,
    features: list[str] | tuple[str, ...],
    *,
    method: Literal["ewma", "rolling"],
    alpha: float | None = None,
    window: int | None = None,
    order_by: str | None = None,
) -> PastBaselineResult:
    """Compare each ordered row with a baseline fitted strictly to earlier rows.

    Without ``order_by``, the caller-provided row order is authoritative. When
    supplied, the column must contain unique, non-null, strictly increasing timestamps.
    """
    if method == "ewma":
        if (
            isinstance(alpha, (bool, np.bool_))
            or not isinstance(alpha, (int, float, np.integer, np.floating))
            or not np.isfinite(alpha)
            or not 0 < alpha <= 1
            or window is not None
        ):
            raise InvalidDetectorConfigError("EWMA requires finite alpha in (0, 1] and no window")
        alpha = float(alpha)
    elif method == "rolling":
        if (
            isinstance(window, (bool, np.bool_))
            or not isinstance(window, (int, np.integer))
            or window < 1
            or alpha is not None
        ):
            raise InvalidDetectorConfigError(
                "rolling requires a positive integer window and no alpha"
            )
        window = int(window)
    else:
        raise InvalidDetectorConfigError("method must be 'ewma' or 'rolling'")
    if not isinstance(frame, pd.DataFrame):
        raise FeatureSelectionError("input must be a pandas DataFrame")
    if frame.empty:
        raise EmptyDataError("input dataset has zero rows")
    selected = tuple(features)
    if not selected or len(set(selected)) != len(selected):
        raise FeatureSelectionError("select distinct numeric features")
    if order_by is not None:
        if order_by not in frame.columns:
            raise FeatureSelectionError(f"missing order column: {order_by}")
        order = frame[order_by]
        if order.isna().any() or order.duplicated().any() or not order.is_monotonic_increasing:
            raise FeatureSelectionError("order column must be unique and strictly increasing")
    missing = [feature for feature in selected if feature not in frame.columns]
    if missing:
        raise FeatureSelectionError(f"missing feature(s): {', '.join(missing)}")
    values = {}
    for feature in selected:
        series = frame[feature]
        if not pd.api.types.is_numeric_dtype(series.dtype) or pd.api.types.is_bool_dtype(
            series.dtype
        ):
            raise NonNumericFeatureError(f"feature {feature!r} must have numeric non-boolean dtype")
        array = series.to_numpy(dtype=np.float64, copy=True)
        if not np.isfinite(array).all():
            raise NonFiniteValueError(f"feature {feature!r} contains non-finite values")
        values[feature] = array
    baseline = pd.DataFrame(index=frame.index, columns=selected, dtype=float)
    for feature, array in values.items():
        estimates = np.full(len(array), np.nan)
        if method == "ewma" and len(array):
            state = array[0]
            for i in range(1, len(array)):
                estimates[i] = state
                with np.errstate(over="ignore", invalid="ignore"):
                    state = alpha * array[i] + (1 - alpha) * state
                if not np.isfinite(state):
                    raise NonFiniteValueError(f"EWMA baseline for {feature!r} is non-finite")
        elif method == "rolling":
            for i in range(window, len(array)):
                with np.errstate(over="ignore", invalid="ignore"):
                    estimates[i] = np.mean(array[i - window : i])
                if not np.isfinite(estimates[i]):
                    raise NonFiniteValueError(f"rolling baseline for {feature!r} is non-finite")
        baseline[feature] = estimates
    with np.errstate(over="ignore", invalid="ignore"):
        residual_values = {
            feature: values[feature] - baseline[feature].to_numpy() for feature in selected
        }
    if any(
        not np.isfinite(residual_values[feature][np.isfinite(baseline[feature].to_numpy())]).all()
        for feature in selected
    ):
        raise NonFiniteValueError("past-baseline residual contains non-finite values")
    residual = pd.DataFrame(residual_values, index=frame.index)
    return PastBaselineResult(method, baseline, residual, order_by)
