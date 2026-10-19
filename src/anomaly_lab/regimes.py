from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from anomaly_lab.schema import RegimeDetectionError


@dataclass(frozen=True, kw_only=True)
class RegimeSegment:
    start_index: int
    end_index: int  # exclusive
    start_timestamp: str | None
    end_timestamp: str | None
    median_values: dict[str, float]
    mean_values: dict[str, float]
    n_rows: int
    segment_id: int


@dataclass(frozen=True, kw_only=True)
class RegimeAnalysisResult:
    feature: str
    breakpoints: tuple[int, ...]
    segments: tuple[RegimeSegment, ...]
    n_regimes: int
    ordering_field: str | None
    penalty: float | None
    method: str  # "pelt" | "bottomup" | "window"
    model: str  # "l2" | "l1" | "rbf"
    interpretation: str = "REGIME_SHIFT candidates; root cause is not established"
    assumptions: tuple[str, ...] = ()
    limitations: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "feature": self.feature,
            "breakpoints": list(self.breakpoints),
            "segments": [
                {
                    "start_index": seg.start_index,
                    "end_index": seg.end_index,
                    "start_timestamp": seg.start_timestamp,
                    "end_timestamp": seg.end_timestamp,
                    "median_values": seg.median_values,
                    "mean_values": seg.mean_values,
                    "n_rows": seg.n_rows,
                    "segment_id": seg.segment_id,
                }
                for seg in self.segments
            ],
            "n_regimes": self.n_regimes,
            "ordering_field": self.ordering_field,
            "penalty": self.penalty,
            "method": self.method,
            "model": self.model,
            "interpretation": self.interpretation,
            "assumptions": list(self.assumptions),
            "limitations": list(self.limitations),
        }


def detect_regimes(
    frame: pd.DataFrame,
    feature: str,
    ordering_field: str | None = None,
    penalty: float | None = 1.0,
    method: str = "pelt",
    model: str = "l2",
) -> RegimeAnalysisResult:
    """Detect regime changes in a time series using change-point detection.

    Parameters
    ----------
    frame : pd.DataFrame
        Input telemetry data.
    feature : str
        Numeric feature column to analyze.
    ordering_field : str | None
        If provided, values must be strictly increasing; data is sorted by this field.
    penalty : float | None
        Penalty for adding a change point (higher = fewer breakpoints). None uses ruptures default.
    method : str
        Segmentation method: "pelt" (default), "bottomup", or "window".
    model : str
        Cost function model: "l2" (default), "l1", or "rbf".

    Returns
    -------
    RegimeAnalysisResult
        Detected breakpoints and regime segment statistics.

    Raises
    ------
    RegimeDetectionError
        When preconditions fail (insufficient data, unordered series, missing column).
    """
    _MIN_SERIES_LENGTH = 10

    if feature not in frame.columns:
        raise RegimeDetectionError(f"feature column {feature!r} not found in data")

    series = frame[feature].to_numpy()

    if series.shape[0] < _MIN_SERIES_LENGTH:
        raise RegimeDetectionError(
            f"INSUFFICIENT_DATA: series has {series.shape[0]} rows; "
            f"at least {_MIN_SERIES_LENGTH} are required for change-point detection"
        )

    # Validate finite values
    if not np.isfinite(series).all():
        raise RegimeDetectionError(
            f"feature {feature!r} contains non-finite values; clean data before analysis"
        )

    # Handle ordering
    timestamps_for_segments: list[str | None] = [None] * len(frame)

    working_frame = frame
    if ordering_field is not None:
        if ordering_field not in working_frame.columns:
            raise RegimeDetectionError(f"ordering_field {ordering_field!r} not found in data")
        ordering_values = working_frame[ordering_field].to_numpy()
        if not np.all(np.diff(ordering_values) > 0):
            raise RegimeDetectionError(
                f"ORDERING_VIOLATION: ordering_field {ordering_field!r} must have strictly "
                f"increasing values; detected non-increasing order"
            )
        sort_idx = np.argsort(ordering_values)
        working_frame = working_frame.iloc[sort_idx].reset_index(drop=True)
        series = working_frame[feature].to_numpy()
        if ordering_field in working_frame.columns:
            ts_series = working_frame[ordering_field]
            timestamps_for_segments = [
                str(ts_series.iloc[i]) if pd.notna(ts_series.iloc[i]) else None
                for i in range(len(ts_series))
            ]

    try:
        import ruptures
    except ImportError:
        raise RegimeDetectionError(
            "pip install '.[regimes]' to use change-point detection"
        ) from None

    # Validate method and model
    valid_methods = {"pelt", "bottomup", "window"}
    valid_models = {"l2", "l1", "rbf"}
    if method not in valid_methods:
        raise RegimeDetectionError(f"method {method!r} not in {valid_methods}")
    if model not in valid_models:
        raise RegimeDetectionError(f"model {model!r} not in {valid_models}")

    # Map model to ruptures constant
    bkpt_model = {"l2": "l2", "l1": "l1", "rbf": "rbf"}[model]

    # Build algorithm
    if method == "pelt":
        algo = ruptures.Pelt(model=bkpt_model).segment(
            series, penalty=penalty if penalty is not None else "rbfptic"
        )
        # Pelt returns (breakpoints, ) where breakpoints[0] is always 0
        raw_bkpts = algo[0] if isinstance(algo, tuple) else algo
    elif method == "bottomup":
        algo = ruptures.BottomUp(model=bkpt_model).segment(
            series, n_bkps=max(2, int(series.shape[0] / _MIN_SERIES_LENGTH))
        )
        raw_bkpts = algo[0] if isinstance(algo, tuple) else algo
    elif method == "window":
        algo = ruptures.Window(width=max(3, series.shape[0] // 5), model=bkpt_model).segment(series)
        raw_bkpts = algo[0] if isinstance(algo, tuple) else algo
    else:
        raw_bkpts = [0, len(series)]

    # Normalize breakpoints: always start with 0, end with len(series)
    breakpoints = tuple(int(b) for b in raw_bkpts)
    if breakpoints[0] != 0:
        breakpoints = (0,) + breakpoints
    if breakpoints[-1] != len(series):
        breakpoints = breakpoints + (len(series),)

    # Build segments
    segments: list[RegimeSegment] = []
    for seg_id, (start, end) in enumerate(zip(breakpoints[:-1], breakpoints[1:], strict=True)):
        seg_series = series[start:end]
        n_rows = end - start

        # Median and mean per feature (single feature here)
        seg_median = float(np.median(seg_series))
        seg_mean = float(np.mean(seg_series))

        ts_len = len(timestamps_for_segments)
        start_ts = timestamps_for_segments[start] if start < ts_len else None
        end_ts = timestamps_for_segments[end - 1] if (end - 1) < ts_len else None

        segments.append(
            RegimeSegment(
                start_index=int(start),
                end_index=int(end),
                start_timestamp=start_ts,
                end_timestamp=end_ts,
                median_values={feature: seg_median},
                mean_values={feature: seg_mean},
                n_rows=n_rows,
                segment_id=seg_id,
            )
        )

    return RegimeAnalysisResult(
        feature=feature,
        breakpoints=breakpoints,
        segments=tuple(segments),
        n_regimes=len(segments),
        ordering_field=ordering_field,
        penalty=penalty,
        method=method,
        model=model,
        interpretation="REGIME_SHIFT candidates; root cause is not established",
        assumptions=(
            "stationarity within segments",
            "change points are abrupt shifts, not gradual drifts",
            f"method={method} with model={model}",
        ),
        limitations=(
            "penalty value controls sensitivity; domain knowledge needed to tune",
            f"n_regimes={len(segments)} is descriptive, not ground truth",
            "ruptures is approximate; different seeds/parameters may yield different breakpoints",
        ),
    )
