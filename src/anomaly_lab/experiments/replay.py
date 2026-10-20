"""Replay and verification of experiment provenance records."""

from __future__ import annotations

from anomaly_lab.experiments.hashing import (
    canonical_experiment_hash,
)
from anomaly_lab.experiments.provenance import ProvenanceRecord
from anomaly_lab.experiments.result import ExperimentResult
from anomaly_lab.experiments.spec import ExperimentSpec


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

        return ExperimentSpec(
            experiment_id=snap["experiment_id"],
            dataset_path=snap["dataset_path"],
            dataset_hash=snap["dataset_hash"],
            features=tuple(snap["features"]),
            analysis_type=snap.get("analysis_type", "anomaly_detection"),
            detectors=tuple(snap.get("detectors") or ()),
            baseline_role=snap.get("baseline_role"),
            candidate_role=snap.get("candidate_role"),
            metric_direction=snap.get("metric_direction"),
            thresholds=tuple(snap.get("thresholds") or ()),
            random_seed=snap.get("random_seed", 42),
            allowed_changed_dimensions=tuple(snap.get("allowed_changed_dimensions") or ()),
            environment_metadata=tuple(snap.get("environment_metadata") or ()),
            explicit_assumptions=tuple(snap.get("explicit_assumptions") or ()),
            metadata=tuple(snap.get("metadata") or ()),
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
