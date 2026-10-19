"""Dataset semantics classification for telemetry data."""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING

import pandas as pd

from anomaly_lab.schema import AnomalyLabError

if TYPE_CHECKING:
    pass


class DatasetSemantics(Enum):
    UNORDERED_TABULAR = "unordered_tabular"
    ORDERED_SERIES = "ordered_series"
    PAIRED_EXPERIMENT = "paired_experiment"
    INDEPENDENT_EXPERIMENT = "independent_experiment"
    GROUPED = "grouped"
    STREAMING_ORDERED = "streaming_ordered"


@dataclass(frozen=True, kw_only=True)
class DatasetProfile:
    semantics: DatasetSemantics
    has_timestamp: bool
    has_group_key: bool
    has_pairing_key: bool
    metric_columns: tuple[str, ...]
    label_column: str | None
    anomaly_intervals: tuple[tuple[int, int], ...]
    ordering_field: str | None
    assumptions: tuple[str, ...]
    warnings: tuple[str, ...]


# -------------------------------------------------------------------
# Timestamp column detection
# -------------------------------------------------------------------

_TIMESTAMP_PATTERNS = (
    re.compile(r"timestamp", re.IGNORECASE),
    re.compile(r"time$", re.IGNORECASE),
    re.compile(r"^time$", re.IGNORECASE),
    re.compile(r"date$", re.IGNORECASE),
    re.compile(r"^dt\b", re.IGNORECASE),
    re.compile(r"datetime", re.IGNORECASE),
    re.compile(r"^ts$", re.IGNORECASE),
    re.compile(r"^t\b", re.IGNORECASE),
)


def _looks_like_timestamp_column(name: str) -> bool:
    return any(p.match(name) for p in _TIMESTAMP_PATTERNS)


def _detect_timestamp_columns(frame: pd.DataFrame) -> list[str]:
    candidates = []
    for col in frame.columns:
        if _looks_like_timestamp_column(col):
            candidates.append(col)
    return candidates


def _is_datetime_series(series: pd.Series) -> bool:
    return pd.api.types.is_datetime64_any_dtype(series.dtype) or pd.api.types.is_timedelta64_dtype(
        series.dtype
    )


# -------------------------------------------------------------------
# Group / key column detection
# -------------------------------------------------------------------

_LABEL_COLUMN_NAMES = frozenset(
    {"anomaly", "label", "is_anomaly", "is_outlier", "outlier", "flag", "target", "case"}
)


def _looks_like_label_column(series: pd.Series, name: str) -> bool:
    name_lower = name.lower().strip()
    if name_lower in _LABEL_COLUMN_NAMES:
        return True
    dtype = series.dtype
    if pd.api.types.is_bool_dtype(dtype):
        return True
    if pd.api.types.is_object_dtype(dtype):
        unique = series.dropna().unique()
        if len(unique) <= 2:
            str_vals = set(str(v).lower() for v in unique)
            if str_vals <= {"0", "1", "true", "false", "yes", "no", "anomaly", "normal"}:
                return True
    if pd.api.types.is_numeric_dtype(dtype):
        unique = series.dropna().unique()
        if len(unique) <= 2 and set(unique).issubset({0, 1}):
            return True
    return False


def _detect_group_columns(frame: pd.DataFrame) -> list[str]:
    groups = []
    for col in frame.columns:
        series = frame[col]
        if pd.api.types.is_numeric_dtype(series.dtype):
            continue
        if _looks_like_label_column(series, col):
            continue
        if series.nunique() < len(frame) and series.nunique() > 1:
            groups.append(col)
    return groups


# -------------------------------------------------------------------
# Pairing key detection
# -------------------------------------------------------------------

_PAIRING_PATTERNS = (
    re.compile(r"pair", re.IGNORECASE),
    re.compile(r"subject", re.IGNORECASE),
    re.compile(r"patient", re.IGNORECASE),
    re.compile(r"sample", re.IGNORECASE),
    re.compile(r"run_id", re.IGNORECASE),
    re.compile(r"experiment", re.IGNORECASE),
)


def _looks_like_pairing_key(name: str, n_unique: int, n_rows: int) -> bool:
    if any(p.search(name) for p in _PAIRING_PATTERNS):
        return n_unique > 1 and n_unique < n_rows
    return False


def _detect_pairing_keys(frame: pd.DataFrame) -> list[str]:
    pairing_cols = []
    n_rows = len(frame)
    for col in frame.columns:
        series = frame[col]
        n_unique = series.nunique()
        if _looks_like_pairing_key(col, n_unique, n_rows):
            pairing_cols.append(col)
    return pairing_cols


# -------------------------------------------------------------------
# Metric column detection
# -------------------------------------------------------------------


def _detect_metric_columns(frame: pd.DataFrame) -> list[str]:
    metrics = []
    for col in frame.columns:
        series = frame[col]
        is_numeric = pd.api.types.is_numeric_dtype(series.dtype)
        if is_numeric and not _looks_like_label_column(series, col):
            if series.nunique() > 1:
                metrics.append(col)
    return metrics


# -------------------------------------------------------------------
# Label column detection
# -------------------------------------------------------------------


def _detect_label_column(frame: pd.DataFrame) -> str | None:
    for col in frame.columns:
        series = frame[col]
        if _looks_like_label_column(series, col):
            return col
    return None


# -------------------------------------------------------------------
# Anomaly intervals
# -------------------------------------------------------------------


def _detect_anomaly_intervals(
    frame: pd.DataFrame, label_col: str | None
) -> tuple[tuple[int, int], ...]:
    if label_col is None:
        return ()
    series = frame[label_col]
    intervals = []
    in_interval = False
    start = 0
    for i, value in enumerate(series):
        is_anomaly = bool(value) if pd.notna(value) else False
        if is_anomaly and not in_interval:
            start = i
            in_interval = True
        elif not is_anomaly and in_interval:
            intervals.append((start, i - 1))
            in_interval = False
    if in_interval:
        intervals.append((start, len(series) - 1))
    return tuple(intervals)


# -------------------------------------------------------------------
# Strictly increasing check for ordering
# -------------------------------------------------------------------


def _detect_ordering_field(frame: pd.DataFrame, timestamp_cols: list[str]) -> str | None:
    for col in timestamp_cols:
        series = frame[col]
        try:
            if pd.api.types.is_datetime64_any_dtype(series.dtype):
                diffs = series.diff().dropna()
                if (diffs > pd.Timedelta(0)).all():
                    return col
            elif pd.api.types.is_numeric_dtype(series.dtype):
                diffs = series.diff().dropna()
                if (diffs > 0).all():
                    return col
        except Exception:
            continue
    return None


# -------------------------------------------------------------------
# Main classification
# -------------------------------------------------------------------


def classify(frame: pd.DataFrame) -> DatasetProfile:
    if not isinstance(frame, pd.DataFrame):
        raise AnomalyLabError("classify requires a pandas DataFrame")
    if frame.empty:
        raise AnomalyLabError("classify requires a non-empty DataFrame")

    warnings: list[str] = []
    assumptions: list[str] = []

    timestamp_cols = _detect_timestamp_columns(frame)
    has_timestamp = len(timestamp_cols) > 0

    ordering_field = None
    if has_timestamp:
        ordering_field = _detect_ordering_field(frame, timestamp_cols)
        if ordering_field is None:
            warnings.append(
                f"Timestamp column(s) {timestamp_cols} detected "
                "but values are not strictly increasing; "
                "treating dataset as unordered tabular."
            )

    group_cols = _detect_group_columns(frame)
    has_group_key = len(group_cols) > 0

    pairing_cols = _detect_pairing_keys(frame)
    has_pairing_key = len(pairing_cols) > 0

    metric_cols = _detect_metric_columns(frame)
    label_col = _detect_label_column(frame)

    anomaly_intervals = _detect_anomaly_intervals(frame, label_col)

    # Determine semantics
    semantics = _infer_semantics(
        frame=frame,
        has_timestamp=has_timestamp,
        ordering_field=ordering_field,
        has_group_key=has_group_key,
        has_pairing_key=has_pairing_key,
        metric_cols=metric_cols,
        label_col=label_col,
        warnings=warnings,
    )

    assumptions = _build_assumptions(
        semantics, has_timestamp, ordering_field, has_group_key, has_pairing_key, metric_cols
    )

    return DatasetProfile(
        semantics=semantics,
        has_timestamp=has_timestamp,
        has_group_key=has_group_key,
        has_pairing_key=has_pairing_key,
        metric_columns=tuple(metric_cols),
        label_column=label_col,
        anomaly_intervals=anomaly_intervals,
        ordering_field=ordering_field,
        assumptions=tuple(assumptions),
        warnings=tuple(warnings),
    )


def _infer_semantics(
    frame: pd.DataFrame,
    has_timestamp: bool,
    ordering_field: str | None,
    has_group_key: bool,
    has_pairing_key: bool,
    metric_cols: list[str],
    label_col: str | None,
    warnings: list[str],
) -> DatasetSemantics:
    n_rows = len(frame)
    n_numeric = len(metric_cols)

    # Check for streaming/ordered series first
    if has_timestamp and ordering_field is not None:
        return DatasetSemantics.ORDERED_SERIES

    # Check for grouped data
    if has_group_key:
        return DatasetSemantics.GROUPED

    # Check for paired experiment
    if has_pairing_key:
        if label_col is not None:
            return DatasetSemantics.PAIRED_EXPERIMENT
        else:
            warnings.append(
                "Pairing key detected but no label column found; "
                "treating as independent_experiment because anomaly labels "
                "are required for paired_experiment semantics."
            )
            return DatasetSemantics.INDEPENDENT_EXPERIMENT

    # Unordered tabular is the default for plain numeric feature matrices
    if not has_timestamp:
        if n_numeric > 0 and n_rows > 1:
            return DatasetSemantics.UNORDERED_TABULAR
        elif n_rows <= 1:
            warnings.append("Dataset has only one row; semantics classification is limited.")
            return DatasetSemantics.UNORDERED_TABULAR

    # Fallback: timestamp exists but not strictly ordered
    if has_timestamp:
        warnings.append(
            "Timestamp column detected but order cannot be verified as strictly increasing; "
            "treating as unordered_tabular. For ordered series, ensure timestamps "
            "are monotonically increasing."
        )
        return DatasetSemantics.UNORDERED_TABULAR

    return DatasetSemantics.UNORDERED_TABULAR


def _build_assumptions(
    semantics: DatasetSemantics,
    has_timestamp: bool,
    ordering_field: str | None,
    has_group_key: bool,
    has_pairing_key: bool,
    metric_cols: list[str],
) -> list[str]:
    assumptions = []
    if semantics == DatasetSemantics.UNORDERED_TABULAR:
        assumptions.append("Rows are independent and identically distributed (i.i.d.)")
        assumptions.append("No temporal ordering is assumed or used")
    elif semantics == DatasetSemantics.ORDERED_SERIES:
        assumptions.append(f"Data is ordered by '{ordering_field}'")
        assumptions.append("Temporal autocorrelation may exist between consecutive rows")
    elif semantics == DatasetSemantics.PAIRED_EXPERIMENT:
        assumptions.append("Rows are paired by a shared key across conditions")
        assumptions.append("Pair-level matching is preserved")
    elif semantics == DatasetSemantics.INDEPENDENT_EXPERIMENT:
        assumptions.append("Each row represents an independent experimental run")
        assumptions.append("No pairing or grouping structure is assumed")
    elif semantics == DatasetSemantics.GROUPED:
        assumptions.append("Rows belong to groups; within-group correlation may exist")
    elif semantics == DatasetSemantics.STREAMING_ORDERED:
        assumptions.append(
            "Data arrives in temporal order; only past data is available at each point"
        )
    return assumptions
