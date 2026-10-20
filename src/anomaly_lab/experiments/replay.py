"""Replay and verification of experiment provenance records."""

from __future__ import annotations

from typing import Any

from anomaly_lab.experiments.hashing import (
    canonical_experiment_hash,
    canonical_spec_hash,
)
from anomaly_lab.experiments.provenance import ProvenanceRecord
from anomaly_lab.experiments.spec import ExperimentSpec
from anomaly_lab.experiments.result import ExperimentResult
from anomaly_lab.semantics import DatasetSemantics


class ExperimentReplay:
    """Reconstruct and verify experiment results from provenance records.

    Usage::

        replay = ExperimentReplay(provenance_record)
        if replay.verify():
            spec, result = replay.restore()
    """

    def __init__(self, provenance: ProvenanceRecord) -> None:
        self.provenance = provenance

    def verify(self) -> bool:
        """Verify that the provenance record is internally consistent.

        Checks that the stored experiment_hash matches a freshly-computed
        hash of the restored spec and result snapshots.
        """
        try:
            spec = self._reconstruct_spec()
            result = self._reconstruct_result(spec)
            recomputed = canonical_experiment_hash(result)
            return recomputed == self.provenance.experiment_hash
        except Exception:
            return False

    def restore(self) -> tuple[ExperimentSpec, ExperimentResult]:
        """Reconstruct the ExperimentSpec and ExperimentResult from the provenance record.

        Raises
        ------
        ValueError :
            If the provenance record cannot be decoded into a valid spec/result.
        """
        spec = self._reconstruct_spec()
        result = self._reconstruct_result(spec)
        return spec, result

    def _reconstruct_spec(self) -> ExperimentSpec:
        """Reconstruct ExperimentSpec from spec_snapshot."""
        snap = self.provenance.spec_snapshot
        semantics_val = snap.get("semantics")
        semantics = DatasetSemantics(semantics_val) if semantics_val else None

        return ExperimentSpec(
            experiment_name=snap["experiment_name"],
            detector_name=snap["detector_name"],
            features=tuple(snap["features"]),
            dataset_id=snap["dataset_id"],
            input_hash=snap["input_hash"],
            parameters=snap.get("parameters"),
            random_seed=snap.get("random_seed"),
            semantics=semantics,
            constant_features=tuple(snap.get("constant_features") or []),
            metadata=snap.get("metadata"),
            schema_version=snap.get("schema_version", "0.5"),
        )

    def _reconstruct_result(self, spec: ExperimentSpec) -> ExperimentResult:
        """Reconstruct ExperimentResult from result_snapshot and spec."""
        snap = self.provenance.result_snapshot

        return ExperimentResult(
            spec=spec,
            row_indices=tuple(snap["row_indices"]),
            is_anomaly=tuple(snap["is_anomaly"]),
            scores=tuple(snap["scores"]) if snap.get("scores") else None,
            clusters=tuple(snap["clusters"]) if snap.get("clusters") else None,
            metadata=snap.get("metadata"),
            schema_version=snap.get("schema_version", "0.5"),
            confidence=snap.get("confidence"),
            applicability=snap.get("applicability", "APPLICABLE"),
        )
