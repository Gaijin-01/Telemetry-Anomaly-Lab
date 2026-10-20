"""Experiment specification: frozen, hashable, validatable experiment configuration."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from anomaly_lab.semantics import DatasetSemantics


@dataclass(frozen=True, kw_only=True)
class ExperimentSpec:
    """Frozen, hashable specification of a single anomaly-detection experiment.

    Attributes
    ----------
    experiment_name :
        Human-readable name for this experiment.
    detector_name :
        Name of the detector to run (e.g. "robust-zscore", "isolation-forest").
    features :
        Tuple of feature column names used in this experiment.
    dataset_id :
        Identifier for the input dataset.
    input_hash :
        SHA256 hash of the canonical input data snapshot.
    parameters :
        Detector-specific hyperparameter dict.  None is canonical for
        detectors that take no parameters.
    random_seed :
        Random seed for stochastic detectors.  None if the detector is
        deterministic or no seed was set.
    semantics :
        Dataset semantics classification at the time the experiment was run.
    constant_features :
        Feature columns that were found to be constant and excluded.
    metadata :
        Arbitrary caller-supplied key/value store.  None is equivalent to {}.
    schema_version :
        Schema version for forward/backward compatibility.
    """

    experiment_name: str
    detector_name: str
    features: tuple[str, ...]
    dataset_id: str
    input_hash: str
    parameters: dict[str, Any] | None = None
    random_seed: int | None = None
    semantics: DatasetSemantics | None = None
    constant_features: tuple[str, ...] = ()
    metadata: dict[str, Any] | None = None
    schema_version: str = "0.5"

    def __post_init__(self) -> None:
        if not self.experiment_name or not self.experiment_name.strip():
            raise ValueError("experiment_name cannot be empty")
        if not self.detector_name or not self.detector_name.strip():
            raise ValueError("detector_name cannot be empty")
        if not self.features:
            raise ValueError("features cannot be empty")
        if len(self.features) != len(set(self.features)):
            raise ValueError("features contains duplicate entries")
        if not self.dataset_id or not self.dataset_id.strip():
            raise ValueError("dataset_id cannot be empty")
        if not self.input_hash or not self.input_hash.strip():
            raise ValueError("input_hash cannot be empty")

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-serializable dict."""
        return {
            "experiment_name": self.experiment_name,
            "detector_name": self.detector_name,
            "features": list(self.features),
            "dataset_id": self.dataset_id,
            "input_hash": self.input_hash,
            "parameters": _safe(self.parameters or {}),
            "random_seed": self.random_seed,
            "semantics": self.semantics.value if self.semantics else None,
            "constant_features": list(self.constant_features),
            "metadata": _safe(self.metadata or {}),
            "schema_version": self.schema_version,
        }


def _safe(value: Any) -> Any:
    """Recursively convert numpy / pandas types to JSON-safe Python types."""
    if isinstance(value, dict):
        return {str(k): _safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return [_safe(item) for item in value.tolist()]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)
