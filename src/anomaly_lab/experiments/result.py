"""Experiment result: frozen, hashable result of a single anomaly-detection experiment."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from anomaly_lab.experiments.spec import ExperimentSpec
from anomaly_lab.results import _safe


@dataclass(frozen=True, kw_only=True)
class ExperimentResult:
    """Frozen, hashable result of a single anomaly-detection experiment.

    Attributes
    ----------
    spec :
        The ExperimentSpec that generated this result.
    row_indices :
        Tuple of row indices (positions) in the original dataset.
    is_anomaly :
        Tuple of bools indicating whether each row is anomalous.
    scores :
        Anomaly scores for each row, or None if the detector does not produce scores.
    clusters :
        Cluster labels for each row, or None if not applicable.
    metadata :
        Arbitrary caller-supplied key/value store.  None is equivalent to {}.
    schema_version :
        Schema version for forward/backward compatibility.
    confidence :
        Confidence in [0, 1] for the overall detection verdict.
    applicability :
        One of "APPLICABLE", "NOT_APPLICABLE", "INSUFFICIENT_DATA", "INVALID_INPUT".
    """

    spec: ExperimentSpec
    row_indices: tuple[int, ...]
    is_anomaly: tuple[bool, ...]
    scores: tuple[float | None, ...] | None = None
    clusters: tuple[int, ...] | None = None
    metadata: dict[str, Any] | None = None
    schema_version: str = "0.5"
    confidence: float | None = None
    applicability: str = "APPLICABLE"

    def __post_init__(self) -> None:
        size = len(self.row_indices)
        if size == 0:
            raise ValueError("ExperimentResult cannot represent zero rows")
        if len(self.is_anomaly) != size:
            raise ValueError("is_anomaly length must match row_indices")
        if self.scores is not None:
            if len(self.scores) != size:
                raise ValueError("scores length must match row_indices")
            if any(score is not None and not np.isfinite(score) for score in self.scores):
                raise ValueError("scores must be finite or None")
        if self.clusters is not None and len(self.clusters) != size:
            raise ValueError("clusters length must match row_indices")
        if self.confidence is not None:
            # Clamp to valid range (frozen dataclass — use object.__setattr__)
            clamped = max(0.0, min(1.0, self.confidence))
            object.__setattr__(self, "confidence", clamped)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-serializable dict."""
        return {
            "spec": self.spec.to_dict(),
            "schema_version": self.schema_version,
            "confidence": self.confidence,
            "applicability": self.applicability,
            "row_indices": [
                int(i) if np.issubdtype(type(i), np.integer) else i for i in self.row_indices
            ],
            "is_anomaly": [bool(a) for a in self.is_anomaly],
            "scores": [float(s) if s is not None else None for s in (self.scores or ())],
            "clusters": [int(c) if c is not None else None for c in (self.clusters or ())],
            "rows": [
                {
                    "position": position,
                    "index": int(index) if np.issubdtype(type(index), np.integer) else index,
                    "is_anomaly": bool(anomaly),
                    "score": (
                        float(self.scores[position])
                        if self.scores is not None and self.scores[position] is not None
                        else None
                    ),
                    "cluster": (
                        int(self.clusters[position]) if self.clusters is not None else None
                    ),
                }
                for position, (index, anomaly) in enumerate(
                    zip(self.row_indices, self.is_anomaly, strict=True)
                )
            ],
            "metadata": _safe(self.metadata or {}),
        }

    @property
    def row_count(self) -> int:
        """Number of rows represented in this result."""
        return len(self.row_indices)
