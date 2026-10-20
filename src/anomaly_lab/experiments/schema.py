"""Experiment specification: frozen dataclass, immutable, validated.

This module defines the immutable experiment specification that governs
which detectors run, on what features, with what thresholds.
"""

from __future__ import annotations

from dataclasses import dataclass

VALID_ANALYSIS_TYPES = frozenset({"anomaly_detection", "compare", "change_point", "regime"})
VALID_METRIC_DIRECTIONS = frozenset({"higher_is_better", "lower_is_better"})


@dataclass(frozen=True, kw_only=True)
class ExperimentSpec:
    """Immutable, validated experiment specification.

    Attributes
    ----------
    experiment_id:
        Stable, human-readable identifier for the experiment.
    dataset_path:
        Path to the dataset file used for this experiment.
    dataset_hash:
        SHA256 hash of the dataset content (dataset_hash).
    features:
        Tuple of feature column names used in this experiment.
    metadata:
        Key-value pairs describing the spec.
    analysis_type:
        One of "anomaly_detection", "compare", "change_point", or "regime".
    detectors:
        Names of detectors to apply.
    baseline_role:
        Role name of the baseline dataset used in "compare" mode.
    candidate_role:
        Role name of the candidate dataset used in "compare" mode.
    metric_direction:
        One of "higher_is_better", "lower_is_better", or None.
    thresholds:
        Tuple of (metric_name, threshold_value) pairs.
    random_seed:
        Integer random seed for reproducibility.
    allowed_changed_dimensions:
        Feature dimensions that may differ between baseline and candidate.
    environment_metadata:
        Key-value pairs describing the environment at creation time.
    explicit_assumptions:
        Human-readable assumptions held by this experiment.
    """

    experiment_id: str
    dataset_path: str
    dataset_hash: str
    features: tuple[str, ...]
    metadata: tuple[tuple[str, str], ...] = ()
    analysis_type: str = "anomaly_detection"
    detectors: tuple[str, ...] = ()
    baseline_role: str | None = None
    candidate_role: str | None = None
    metric_direction: str | None = None
    thresholds: tuple[tuple[str, float], ...] = ()
    random_seed: int = 42
    allowed_changed_dimensions: tuple[str, ...] = ()
    environment_metadata: tuple[tuple[str, str], ...] = ()
    explicit_assumptions: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.experiment_id or not self.experiment_id.strip():
            raise ValueError("experiment_id must be a non-empty string")
        if not self.dataset_path or not self.dataset_path.strip():
            raise ValueError("dataset_path must be a non-empty string")
        if not self.dataset_hash or not self.dataset_hash.strip():
            raise ValueError("dataset_hash must be a non-empty string")
        if not self.features:
            raise ValueError("features cannot be empty")
        if len(self.features) != len(set(self.features)):
            raise ValueError("features contains duplicate entries")
        if self.analysis_type not in VALID_ANALYSIS_TYPES:
            raise ValueError(
                f"analysis_type must be one of {VALID_ANALYSIS_TYPES}, got {self.analysis_type!r}"
            )
        if self.analysis_type == "compare":
            if not self.baseline_role:
                raise ValueError("analysis_type='compare' requires baseline_role")
            if not self.candidate_role:
                raise ValueError(
                    "analysis_type='compare' requires both baseline_role and candidate_role"
                )
        if self.metric_direction is not None:
            if self.metric_direction not in VALID_METRIC_DIRECTIONS:
                raise ValueError(
                    f"metric_direction must be one of {VALID_METRIC_DIRECTIONS} or None, "
                    f"got {self.metric_direction!r}"
                )
        # Thresholds require non-empty features
        if self.thresholds and not self.features:
            raise ValueError("thresholds requires features to be non-empty")

    def to_dict(self) -> dict:
        """Serialize to a JSON-compatible dict."""
        return {
            "experiment_id": self.experiment_id,
            "dataset_path": self.dataset_path,
            "dataset_hash": self.dataset_hash,
            "features": list(self.features),
            "metadata": list(self.metadata),
            "analysis_type": self.analysis_type,
            "detectors": list(self.detectors),
            "baseline_role": self.baseline_role,
            "candidate_role": self.candidate_role,
            "metric_direction": self.metric_direction,
            "thresholds": list(self.thresholds),
            "random_seed": self.random_seed,
            "allowed_changed_dimensions": list(self.allowed_changed_dimensions),
            "environment_metadata": list(self.environment_metadata),
            "explicit_assumptions": list(self.explicit_assumptions),
        }
