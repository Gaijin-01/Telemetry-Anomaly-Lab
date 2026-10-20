from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class DetectionResult:
    detector: str
    row_indices: tuple[Any, ...]
    is_anomaly: tuple[bool, ...]
    features: tuple[str, ...]
    constant_features: tuple[str, ...]
    parameters: dict[str, Any]
    scores: tuple[float | None, ...] | None = None
    clusters: tuple[int, ...] | None = None
    metadata: dict[str, Any] | None = None
    # Phase N fields
    schema_version: str = "0.5"
    dataset_semantics: str | None = None
    evidence_class: str = ""
    confidence: float | None = None
    applicability: str = "APPLICABLE"

    def __post_init__(self) -> None:
        size = len(self.row_indices)
        if size == 0:
            raise ValueError("DetectionResult cannot represent zero rows")
        if len(self.is_anomaly) != size:
            raise ValueError("is_anomaly length must match row_indices")
        if self.scores is not None and len(self.scores) != size:
            raise ValueError("scores length must match row_indices")
        if self.clusters is not None and len(self.clusters) != size:
            raise ValueError("clusters length must match row_indices")
        if self.scores is not None and any(
            score is not None and not np.isfinite(score) for score in self.scores
        ):
            raise ValueError("scores must be finite or None")

    def to_dict(self) -> dict[str, Any]:
        return {
            "detector": self.detector,
            "features": list(self.features),
            "constant_features": list(self.constant_features),
            "parameters": _safe(self.parameters),
            "metadata": _safe(self.metadata or {}),
            "schema_version": self.schema_version,
            "dataset_semantics": self.dataset_semantics,
            "evidence_class": self.evidence_class,
            "confidence": self.confidence,
            "applicability": self.applicability,
            "rows": [
                {
                    "position": position,
                    "index": _safe(index),
                    "is_anomaly": anomaly,
                    "score": self.scores[position] if self.scores is not None else None,
                    "cluster": self.clusters[position] if self.clusters is not None else None,
                }
                for position, (index, anomaly) in enumerate(
                    zip(self.row_indices, self.is_anomaly, strict=True)
                )
            ],
        }


def _safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe(item) for item in value]
    if isinstance(value, (pd.Timestamp, np.datetime64)):
        return pd.Timestamp(value).isoformat()
    if isinstance(value, (pd.Timedelta, np.timedelta64)):
        return pd.Timedelta(value).isoformat()
    if isinstance(value, np.generic):
        return _safe(value.item())
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(f"value of type {type(value).__name__} is not JSON serializable")
