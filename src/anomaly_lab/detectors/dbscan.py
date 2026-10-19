from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN as SKDBSCAN
from sklearn.preprocessing import StandardScaler

from anomaly_lab.features import prepare_features
from anomaly_lab.results import DetectionResult
from anomaly_lab.schema import FeatureSelectionError, InvalidDetectorConfigError


@dataclass(frozen=True)
class DBSCANConfig:
    eps: float = 0.5
    min_samples: int = 5


def fit_predict(
    frame: pd.DataFrame,
    features: list[str] | tuple[str, ...],
    config: DBSCANConfig | None = None,
    *,
    metadata_columns: list[str] | tuple[str, ...] = (),
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
    if isinstance(metadata_columns, str) or not isinstance(metadata_columns, (list, tuple)):
        raise FeatureSelectionError("metadata_columns must be a list or tuple of column names")
    if any(not isinstance(name, str) for name in metadata_columns):
        raise FeatureSelectionError("metadata column names must be strings")
    prepared = prepare_features(frame, features)
    selected_metadata = tuple(metadata_columns)
    if len(set(selected_metadata)) != len(selected_metadata):
        raise FeatureSelectionError("metadata selection contains duplicates")
    missing_metadata = [name for name in selected_metadata if name not in frame.columns]
    if missing_metadata:
        raise FeatureSelectionError(f"missing metadata column(s): {', '.join(missing_metadata)}")
    overlap = set(prepared.features) | set(prepared.constant_features)
    if overlap.intersection(selected_metadata):
        raise FeatureSelectionError("metadata columns must be separate from selected features")
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
        metadata={
            "scaler": "StandardScaler",
            "score_semantics": None,
            "cluster_id_semantics": "opaque; no semantic meaning",
            "row_metadata": tuple(
                {name: _metadata_value(frame.iloc[position][name]) for name in selected_metadata}
                for position in range(len(frame))
            ),
        },
    )


def _metadata_value(value: object) -> object:
    if value is None or value is pd.NA or value is pd.NaT:
        return None
    if isinstance(value, (float, np.floating)) and np.isnan(value):
        return None
    if isinstance(value, np.generic):
        return value.item()
    return value


def _index_value(value: object) -> object:
    if isinstance(value, np.generic):
        return value.item()
    return value
