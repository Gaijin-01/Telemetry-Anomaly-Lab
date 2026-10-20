"""Provenance tracking for experiment reproducibility."""

from __future__ import annotations

import sys
from datetime import UTC, datetime
from typing import Any

import numpy as np

from anomaly_lab.experiments.hashing import canonical_experiment_hash
from anomaly_lab.experiments.result import ExperimentResult
from anomaly_lab.results import _safe


def collect_environment() -> dict[str, Any]:
    """Collect environment information for reproducibility.

    Returns a dict with Python version, package versions, OS info, and platform.
    """
    import importlib
    import platform as _platform

    packages = ["numpy", "pandas", "scikit-learn", "scipy"]
    deps: dict[str, str | None] = {}
    for pkg in packages:
        try:
            m = importlib.import_module(pkg)
            deps[pkg] = getattr(m, "__version__", "unknown")
        except ImportError:
            deps[pkg] = None

    return {
        "python_version": f"{sys.version_info.major}.{sys.version_info.minor}."
        f"{sys.version_info.micro}",
        "platform": _platform.platform(),
        "processor": _platform.processor(),
        "dependencies": deps,
        "collected_at": datetime.now(UTC).isoformat(),
    }


class ProvenanceRecord:
    """Immutable record capturing the full provenance of an experiment run.

    Attributes
    ----------
    experiment_hash :
        SHA256 hex digest of the experiment (spec + result).
    environment :
        Snapshot of the runtime environment (Python version, packages, OS).
    spec_snapshot :
        Canonical dict representation of the ExperimentSpec.
    result_snapshot :
        Canonical dict representation of the ExperimentResult.
    collected_at :
        ISO8601 timestamp of when this record was created.
    """

    __slots__ = (
        "experiment_hash",
        "environment",
        "spec_snapshot",
        "result_snapshot",
        "collected_at",
    )

    def __init__(
        self,
        experiment_hash: str,
        environment: dict[str, Any],
        spec_snapshot: dict[str, Any],
        result_snapshot: dict[str, Any],
        collected_at: str,
    ) -> None:
        self.experiment_hash = experiment_hash
        self.environment = environment
        self.spec_snapshot = spec_snapshot
        self.result_snapshot = result_snapshot
        self.collected_at = collected_at

    @classmethod
    def from_result(cls, result: ExperimentResult) -> ProvenanceRecord:
        """Create a ProvenanceRecord from an ExperimentResult."""

        env = collect_environment()
        spec_dict = result.spec.to_dict()
        result_dict = _result_to_dict(result)

        return cls(
            experiment_hash=canonical_experiment_hash(result),
            environment=env,
            spec_snapshot=spec_dict,
            result_snapshot=result_dict,
            collected_at=datetime.now(UTC).isoformat(),
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-serializable dict."""
        return {
            "experiment_hash": self.experiment_hash,
            "environment": _safe(self.environment),
            "spec_snapshot": _safe(self.spec_snapshot),
            "result_snapshot": _safe(self.result_snapshot),
            "collected_at": self.collected_at,
        }


def _result_to_dict(result: ExperimentResult) -> dict[str, Any]:
    """Convert ExperimentResult to a dict suitable for provenance snapshot."""
    return {
        "schema_version": result.schema_version,
        "confidence": result.confidence,
        "applicability": result.applicability,
        "row_indices": [
            int(i) if isinstance(i, (int, np.integer)) else i for i in result.row_indices
        ],
        "is_anomaly": [bool(a) for a in result.is_anomaly],
        "scores": [float(s) if s is not None else None for s in (result.scores or ())],
        "clusters": [int(c) if c is not None else None for c in (result.clusters or ())],
        "metadata": _safe(result.metadata or {}),
    }
