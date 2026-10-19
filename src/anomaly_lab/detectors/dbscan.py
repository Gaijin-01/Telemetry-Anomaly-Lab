from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN as SKDBSCAN
from sklearn.preprocessing import StandardScaler

from anomaly_lab.features import prepare_features
from anomaly_lab.results import DetectionResult
from anomaly_lab.schema import InvalidDetectorConfigError


@dataclass(frozen=True)
class DBSCANConfig:
    eps: float = 0.5
    min_samples: int = 5


def fit_predict(
    frame: pd.DataFrame,
    features: list[str] | tuple[str, ...],
    config: DBSCANConfig | None = None,
) -> DetectionResult:
    config = config or DBSCANConfig()
    if (
        isinstance(config.eps, bool)
        or not isinstance(config.eps, (int, float, np.floating))
        or not np.isfinite(config.eps)
        or config.eps <= 0
    ):
        raise InvalidDetectorConfigError("eps must be finite and positive")
    if (
        isinstance(config.min_samples, bool)
        or not isinstance(config.min_samples, int)
        or config.min_samples < 1
    ):
        raise InvalidDetectorConfigError("min_samples must be a positive integer")
    prepared = prepare_features(frame, features)
    if config.min_samples > len(frame):
        raise InvalidDetectorConfigError("min_samples cannot exceed number of input rows")
    scaled = StandardScaler().fit_transform(prepared.values)
    labels = SKDBSCAN(eps=config.eps, min_samples=config.min_samples).fit_predict(scaled)
    return DetectionResult(
        detector="dbscan",
        row_indices=tuple(_index_value(value) for value in prepared.index),
        is_anomaly=tuple(bool(label == -1) for label in labels),
        features=prepared.features,
        constant_features=prepared.constant_features,
        parameters={"eps": config.eps, "min_samples": config.min_samples},
        clusters=tuple(int(label) for label in labels),
        metadata={"scaler": "StandardScaler", "score_semantics": None},
    )


def _index_value(value: object) -> object:
    if isinstance(value, np.generic):
        return value.item()
    return value
