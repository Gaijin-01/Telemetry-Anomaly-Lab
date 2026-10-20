"""run_experiment skill: execute one detector on one feature column and
return a DetectionResult.

No LLM. No autonomous loops. No retries beyond a single run.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from anomaly_lab.detectors.dbscan import DBSCANConfig
from anomaly_lab.detectors.dbscan import fit_predict as dbscan_fit_predict
from anomaly_lab.detectors.isolation_forest import (
    IsolationForestConfig,
)
from anomaly_lab.detectors.isolation_forest import (
    fit_predict as iforest_fit_predict,
)
from anomaly_lab.detectors.robust_zscore import (
    RobustZScoreConfig,
)
from anomaly_lab.detectors.robust_zscore import (
    fit_predict as rzscore_fit_predict,
)
from anomaly_lab.results import DetectionResult

_DETECTOR_FIT_PREDICT: dict[str, callable] = {
    "isolation-forest": iforest_fit_predict,
    "robust-zscore": rzscore_fit_predict,
    "dbscan": dbscan_fit_predict,
}

_DETECTOR_CONFIG: dict[str, type] = {
    "isolation-forest": IsolationForestConfig,
    "robust-zscore": RobustZScoreConfig,
    "dbscan": DBSCANConfig,
}


@dataclass(frozen=True, kw_only=True)
class ExperimentRunResult:
    detector: str
    feature: str
    detection: DetectionResult | None
    error_message: str | None
    warnings: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "detector": self.detector,
            "feature": self.feature,
            "detection": self.detection.to_dict() if self.detection else None,
            "error_message": self.error_message,
            "warnings": list(self.warnings),
        }


def run_experiment(
    detector_name: str,
    feature: str,
    df: pd.DataFrame,
    detector_params: dict[str, Any] | None = None,
) -> ExperimentRunResult:
    """Run a single detector on a single feature column.

    Parameters
    ----------
    detector_name: one of 'isolation-forest', 'robust-zscore', 'dbscan'
    feature: column name in df
    df: input DataFrame
    detector_params: optional override for detector constructor

    Returns
    -------
    ExperimentRunResult with detection or error_message, never both.

    Raises
    ------
    ValueError if detector_name is unknown.
    """
    if detector_name not in _DETECTOR_FIT_PREDICT:
        raise ValueError(f"Unknown detector: {detector_name!r}")

    fit_predict_fn = _DETECTOR_FIT_PREDICT[detector_name]
    config_cls = _DETECTOR_CONFIG[detector_name]
    params = detector_params or {}

    warnings: list[str] = []

    try:
        # Build config from params dict (pop known config fields)
        config_kwargs = {k: v for k, v in params.items() if k in _config_field_names(config_cls)}
        config = config_cls(**config_kwargs) if config_kwargs else config_cls()
        result = fit_predict_fn(df, [feature], config)
        return ExperimentRunResult(
            detector=detector_name,
            feature=feature,
            detection=result,
            error_message=None,
            warnings=tuple(warnings),
        )
    except Exception as exc:  # pragma: no cover — detectors are third-party
        return ExperimentRunResult(
            detector=detector_name,
            feature=feature,
            detection=None,
            error_message=f"{type(exc).__name__}: {exc}",
            warnings=tuple(warnings),
        )


def _config_field_names(config_cls: type) -> frozenset[str]:
    """Extract field names from a frozen dataclass config."""
    try:
        import dataclasses

        if dataclasses.is_dataclass(config_cls):
            return frozenset(f.name for f in dataclasses.fields(config_cls))
    except Exception:
        pass
    return frozenset()
