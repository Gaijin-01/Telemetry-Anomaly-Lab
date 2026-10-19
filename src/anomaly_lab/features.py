from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from anomaly_lab.schema import (
    EmptyDataError,
    FeatureSelectionError,
    NonFiniteValueError,
    NonNumericFeatureError,
)


@dataclass(frozen=True)
class PreparedFeatures:
    values: np.ndarray
    features: tuple[str, ...]
    constant_features: tuple[str, ...]
    index: pd.Index


def prepare_features(
    frame: pd.DataFrame, features: list[str] | tuple[str, ...]
) -> PreparedFeatures:
    if not isinstance(frame, pd.DataFrame):
        raise FeatureSelectionError("input must be a pandas DataFrame")
    if frame.empty:
        raise EmptyDataError("input dataset has zero rows")
    selected = tuple(features)
    if not selected:
        raise FeatureSelectionError("select at least one feature")
    if len(set(selected)) != len(selected):
        raise FeatureSelectionError("feature selection contains duplicates")
    missing = [name for name in selected if name not in frame.columns]
    if missing:
        raise FeatureSelectionError(f"missing feature(s): {', '.join(missing)}")
    for name in selected:
        series = frame[name]
        if not pd.api.types.is_numeric_dtype(series.dtype) or pd.api.types.is_bool_dtype(
            series.dtype
        ):
            raise NonNumericFeatureError(f"feature {name!r} must have numeric non-boolean dtype")
        values = series.to_numpy(dtype=np.float64, copy=True)
        invalid = np.flatnonzero(~np.isfinite(values))
        if invalid.size:
            locations = [(int(pos), _json_safe_index(frame.index[pos])) for pos in invalid[:10]]
            raise NonFiniteValueError(
                f"feature {name!r} has non-finite values at row positions/indexes {locations}"
            )
    matrix = frame.loc[:, list(selected)].to_numpy(dtype=np.float64, copy=True)
    constant_mask = np.all(matrix == matrix[0], axis=0)
    constant = tuple(
        name for name, is_constant in zip(selected, constant_mask, strict=True) if is_constant
    )
    active = tuple(
        name for name, is_constant in zip(selected, constant_mask, strict=True) if not is_constant
    )
    if not active:
        raise FeatureSelectionError(
            "all selected features are constant; no informative feature remains"
        )
    active_matrix = matrix[:, ~constant_mask].copy()
    active_matrix.setflags(write=False)
    return PreparedFeatures(active_matrix, active, constant, frame.index.copy())


def _json_safe_index(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    return value
