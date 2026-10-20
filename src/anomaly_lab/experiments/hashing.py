"""Canonical hashing for experiment specs, parameters, and results."""

from __future__ import annotations

import hashlib
import json
from typing import Any

from anomaly_lab.experiments.spec import ExperimentSpec
from anomaly_lab.experiments.result import ExperimentResult


# Canonical JSON encoder: sorted keys, no whitespace, UTF-8
_canonical_encoder = json.dumps(
    {}, ensure_ascii=False, separators=(",", ":"), sort_keys=True
).strip("{}")


def _canonical_json(obj: Any) -> str:
    """Return a canonical JSON string for any JSON-serializable value."""
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def canonical_parameters_hash(parameters: dict[str, Any] | None) -> str:
    """SHA256 hex digest of a parameters dict in canonical form.

    Canonical means sorted keys, no extra whitespace.
    None is treated as {} for hashing purposes.
    """
    canonical = _canonical_json(parameters) if parameters else _canonical_json({})
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def canonical_spec_hash(spec: ExperimentSpec) -> str:
    """SHA256 hex digest of an ExperimentSpec in canonical form.

    The canonical form is derived exclusively from the spec's semantic
    fields (not its object id or memory address).
    """
    hasher = hashlib.sha256()
    # Fields in deterministic order
    hasher.update(_canonical_json(spec.experiment_name).encode("utf-8"))
    hasher.update(_canonical_json(spec.detector_name).encode("utf-8"))
    # Features are sorted canonically
    hasher.update(_canonical_json(tuple(sorted(spec.features))).encode("utf-8"))
    hasher.update(_canonical_json(spec.dataset_id).encode("utf-8"))
    hasher.update(_canonical_json(spec.input_hash).encode("utf-8"))
    hasher.update(canonical_parameters_hash(spec.parameters).encode("utf-8"))
    hasher.update(_canonical_json(spec.random_seed).encode("utf-8"))
    hasher.update(
        _canonical_json(spec.semantics.value if spec.semantics else None).encode("utf-8")
    )
    hasher.update(
        _canonical_json(tuple(sorted(spec.constant_features))).encode("utf-8")
    )
    hasher.update(_canonical_json(spec.schema_version).encode("utf-8"))
    return hasher.hexdigest()


def canonical_experiment_hash(result: ExperimentResult) -> str:
    """SHA256 hex digest of an ExperimentResult in canonical form.

    Combines the canonical spec hash with result fields (scores, is_anomaly,
    clusters) in a deterministic order so that the same result always
    produces the same hash.
    """
    hasher = hashlib.sha256()
    # Spec canonical hash
    hasher.update(canonical_spec_hash(result.spec).encode("utf-8"))
    # Result fields
    hasher.update(_canonical_json(result.row_indices).encode("utf-8"))
    hasher.update(_canonical_json(result.is_anomaly).encode("utf-8"))
    hasher.update(_canonical_json(result.scores).encode("utf-8"))
    hasher.update(_canonical_json(result.clusters).encode("utf-8"))
    hasher.update(_canonical_json(result.applicability).encode("utf-8"))
    hasher.update(_canonical_json(result.schema_version).encode("utf-8"))
    return hasher.hexdigest()


class ExperimentHasher:
    """Computes and records canonical hashes for experiments.

    Maintains a history of computed hashes and can detect duplicate
    experiment results.
    """

    def __init__(self) -> None:
        self.hash_history: dict[str, str] = {}  # hash -> category

    def compute_and_record(
        self, result: ExperimentResult, category: str = "experiment"
    ) -> str:
        """Compute the canonical experiment hash and record it.

        Returns the hash.  If the same hash was already recorded,
        it is not re-added to the history.
        """
        h = canonical_experiment_hash(result)
        if h not in self.hash_history:
            self.hash_history[h] = category
        return h

    def is_duplicate(self, result: ExperimentResult) -> bool:
        """Return True if an identical experiment result has already been recorded."""
        h = canonical_experiment_hash(result)
        return h in self.hash_history
