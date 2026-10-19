from dataclasses import dataclass

import numpy as np
import pandas as pd

from anomaly_lab.features import prepare_features
from anomaly_lab.results import DetectionResult
from anomaly_lab.schema import InvalidDetectorConfigError


@dataclass(frozen=True)
class RobustZScoreConfig:
    threshold: float = 3.5


def fit_predict(
    frame: pd.DataFrame,
    features: list[str] | tuple[str, ...],
    config: RobustZScoreConfig | None = None,
) -> DetectionResult:
    config = config or RobustZScoreConfig()
    if (
        isinstance(config.threshold, bool)
        or not isinstance(config.threshold, (int, float, np.floating))
        or not np.isfinite(config.threshold)
        or config.threshold <= 0
    ):
        raise InvalidDetectorConfigError("threshold must be finite and positive")
    prepared = prepare_features(frame, features)
    medians = np.median(prepared.values, axis=0)
    deviations = np.abs(prepared.values - medians)
    mads = np.median(deviations, axis=0)
    scores_by_feature = np.zeros_like(deviations, dtype=np.float64)
    positive_mad = mads > 0
    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        scores_by_feature[:, positive_mad] = (
            deviations[:, positive_mad] / mads[positive_mad]
        ) / 1.4826
    if not np.isfinite(scores_by_feature).all():
        raise InvalidDetectorConfigError("robust Z-scores must be finite")
    for column in np.flatnonzero(~positive_mad):
        scores_by_feature[:, column] = np.where(
            deviations[:, column] == 0.0, 0.0, config.threshold + 1.0
        )
    scores = np.max(scores_by_feature, axis=1)
    return DetectionResult(
        detector="robust-zscore",
        row_indices=tuple(_index_value(value) for value in prepared.index),
        is_anomaly=tuple(bool(score > config.threshold) for score in scores),
        features=prepared.features,
        constant_features=prepared.constant_features,
        parameters={"threshold": config.threshold, "mad_scale": 1.4826},
        scores=tuple(float(score) for score in scores),
        metadata={
            "center": "median",
            "scale": "1.4826 * MAD",
            "zero_mad": "deviation gets threshold+1 ordinal score; not a magnitude estimate",
            "score_semantics": "maximum absolute modified Z-score; larger is more anomalous",
        },
    )


def _index_value(value: object) -> object:
    if isinstance(value, np.generic):
        return value.item()
    return value
