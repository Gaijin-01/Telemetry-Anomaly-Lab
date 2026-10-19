from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

from anomaly_lab.features import prepare_features
from anomaly_lab.results import DetectionResult
from anomaly_lab.schema import InvalidDetectorConfigError


@dataclass(frozen=True)
class IsolationForestConfig:
    contamination: float | str = "auto"
    n_estimators: int = 200
    random_state: int = 42


def fit_predict(
    frame: pd.DataFrame,
    features: list[str] | tuple[str, ...],
    config: IsolationForestConfig | None = None,
) -> DetectionResult:
    config = config or IsolationForestConfig()
    if config.contamination != "auto" and (
        isinstance(config.contamination, bool)
        or not isinstance(config.contamination, (int, float, np.floating))
        or not np.isfinite(config.contamination)
        or not 0.0 < config.contamination <= 0.5
    ):
        raise InvalidDetectorConfigError("contamination must be 'auto' or a number in (0, 0.5]")
    if (
        isinstance(config.n_estimators, bool)
        or not isinstance(config.n_estimators, (int, np.integer))
        or config.n_estimators < 1
    ):
        raise InvalidDetectorConfigError("n_estimators must be a positive integer")
    if (
        isinstance(config.random_state, bool)
        or not isinstance(config.random_state, (int, np.integer))
        or not 0 <= config.random_state <= 2**32 - 1
    ):
        raise InvalidDetectorConfigError("random_state must be an integer in [0, 4294967295]")
    prepared = prepare_features(frame, features)
    if len(frame) < 2:
        raise InvalidDetectorConfigError("Isolation Forest requires at least two rows")
    scaled = StandardScaler().fit_transform(prepared.values)
    model = IsolationForest(
        contamination=config.contamination,
        n_estimators=config.n_estimators,
        random_state=config.random_state,
    )
    predictions = model.fit_predict(scaled)
    scores = -model.score_samples(scaled)
    return DetectionResult(
        detector="isolation-forest",
        row_indices=tuple(_index_value(value) for value in prepared.index),
        is_anomaly=tuple(bool(value == -1) for value in predictions),
        features=prepared.features,
        constant_features=prepared.constant_features,
        parameters={
            "contamination": config.contamination,
            "n_estimators": config.n_estimators,
            "random_state": config.random_state,
        },
        scores=tuple(float(score) for score in scores),
        metadata={
            "scaler": "StandardScaler",
            "score_semantics": "negative score_samples; larger is more anomalous",
        },
    )


def _index_value(value: object) -> object:
    if isinstance(value, np.generic):
        return value.item()
    return value
