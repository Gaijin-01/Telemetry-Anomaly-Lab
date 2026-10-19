"""PyOD-based detector adapters (Phase L).

PyOD (Probability Outlier Detection) is an optional extra.
These adapters are only functional when pyod is installed.

Detector semantics
------------------
All three (ECOD, LOF, KNN) use PyOD's decision_scores_ where
larger values mean more anomalous. We return these scores directly,
consistent with the higher = more anomalous convention used
throughout anomaly-lab.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from anomaly_lab.features import prepare_features
from anomaly_lab.results import DetectionResult
from anomaly_lab.schema import AnomalyLabError, InsufficientDataError, InvalidDetectorConfigError

# pyod is imported lazily so the rest of the package works without it.
_pyod_available: bool | None = None


def _ensure_pyod() -> type:
    """Import and return pyod module. Raises ImportError if not installed."""
    global _pyod_available
    if _pyod_available is None:
        try:
            import pyod  # noqa: F401

            _pyod_available = True
        except ImportError:
            _pyod_available = False
    if not _pyod_available:
        raise ImportError(
            "pyod is not installed. Install it with: pip install pyod\n"
            "Or use the 'pyod' extra: pip install 'anomaly-lab[pyod]'"
        )
    import pyod

    return pyod


#: Upper bound for contamination across all PyOD models in this module.
#: PyOD models (ECOD, LOF, KNN) all require contamination in (0, 0.5].
_CONTAMINATION_MAX = 0.5


def _validate_contamination(value: float, name: str) -> None:
    if not isinstance(value, (int, float)):
        raise InvalidDetectorConfigError(f"{name}: contamination must be a number")
    if not (0 < value <= _CONTAMINATION_MAX):
        raise InvalidDetectorConfigError(
            f"{name}: contamination must be in (0, {_CONTAMINATION_MAX}]"
        )


@dataclass(frozen=True)
class ECODConfig:
    """ECOD (Empirical CDF-based Outlier Detection) configuration.

    ECOD is a parameter-free, highly interpretable outlier detection algorithm
    based on empirical CDF functions. It requires no hyperparameters beyond
    contamination and parallelism (n_jobs).

    Reference: Li et al. 2021, "ECOD: A Python Package for
    Empirical Cumulative Distribution Functions for Outlier Detection".
    """

    contamination: float = 0.1
    n_jobs: int = 1


def fit_predict_ecod(
    frame: pd.DataFrame,
    features: list[str] | tuple[str, ...],
    config: ECODConfig | None = None,
) -> DetectionResult:
    """ECOD (Empirical Cumulative Distribution-based Outlier Detection).

    ECOD computes tail probabilities based on empirical CDFs.
    Larger decision_scores_ = more anomalous (consistent with anomaly-lab convention).
    """
    _ensure_pyod()
    from pyod.models.ecod import ECOD as PyODECOD

    config = config or ECODConfig()
    _validate_contamination(config.contamination, "ECOD")
    if not isinstance(config.n_jobs, int) or config.n_jobs < 1:
        raise InvalidDetectorConfigError("ECOD: n_jobs must be a positive integer")

    prepared = prepare_features(frame, features)
    if prepared.values.shape[0] < 5:
        raise InsufficientDataError(
            f"ECOD requires at least 5 rows; got {prepared.values.shape[0]}"
        )

    model = PyODECOD(contamination=config.contamination, n_jobs=config.n_jobs)
    model.fit(prepared.values)

    # PyOD decision_scores_: larger = more anomalous
    raw_scores = model.decision_scores_
    if not np.all(np.isfinite(raw_scores)):
        raise AnomalyLabError(
            f"ECOD produced non-finite scores (NaN or inf); "
            f"input may contain invalid values. Input shape: {prepared.values.shape}"
        )
    # Guard against degenerate all-zero scores
    max_raw = float(np.max(raw_scores))
    scores = raw_scores if max_raw > 0 else np.zeros_like(raw_scores)

    predictions = model.predict(prepared.values)  # 0 = inlier, 1 = outlier

    return DetectionResult(
        detector="ecod",
        row_indices=tuple(_index_value(v) for v in prepared.index),
        is_anomaly=tuple(bool(p == 1) for p in predictions),
        features=prepared.features,
        constant_features=prepared.constant_features,
        parameters={"contamination": config.contamination, "n_jobs": config.n_jobs},
        scores=tuple(float(s) for s in scores),
        metadata={
            "model": "ECOD",
            "score_semantics": "larger = more anomalous (PyOD decision score)",
            "method_family": "distribution_based",
        },
    )


@dataclass(frozen=True)
class LOFConfig:
    """Local Outlier Factor (k-NN LOF) configuration."""

    contamination: float = 0.1
    n_neighbors: int = 20
    metric: str = "minkowski"
    p: int = 2


def fit_predict_lof(
    frame: pd.DataFrame,
    features: list[str] | tuple[str, ...],
    config: LOFConfig | None = None,
) -> DetectionResult:
    """LOF (Local Outlier Factor).

    LOF computes the local density deviation of a point relative to its k-nearest
    neighbours.  A value ≈ 1 means normal density; > 1 indicates outlier.
    We return higher = more anomalous (decision_scores_ are already oriented).
    """
    _ensure_pyod()
    from pyod.models.lof import LOF as PyODLOF

    config = config or LOFConfig()
    _validate_contamination(config.contamination, "LOF")
    if not isinstance(config.n_neighbors, int) or config.n_neighbors < 1:
        raise InvalidDetectorConfigError("LOF: n_neighbors must be a positive integer")

    prepared = prepare_features(frame, features)
    min_needed = config.n_neighbors + 1
    if prepared.values.shape[0] < min_needed:
        raise InsufficientDataError(
            f"LOF with n_neighbors={config.n_neighbors} requires at least "
            f"{min_needed} rows; got {prepared.values.shape[0]}"
        )

    model = PyODLOF(
        contamination=config.contamination,
        n_neighbors=config.n_neighbors,
        metric=config.metric,
        p=config.p,
    )
    model.fit(prepared.values)
    scores = model.decision_scores_
    if not np.all(np.isfinite(scores)):
        raise AnomalyLabError(
            f"LOF produced non-finite scores (NaN or inf); "
            f"input may contain invalid values. Input shape: {prepared.values.shape}"
        )
    predictions = model.predict(prepared.values)

    return DetectionResult(
        detector="lof",
        row_indices=tuple(_index_value(v) for v in prepared.index),
        is_anomaly=tuple(bool(p == 1) for p in predictions),
        features=prepared.features,
        constant_features=prepared.constant_features,
        parameters={
            "contamination": config.contamination,
            "n_neighbors": config.n_neighbors,
            "metric": config.metric,
            "p": config.p,
        },
        scores=tuple(float(s) for s in scores),
        metadata={
            "model": "LOF",
            "score_semantics": "larger = more anomalous (LOF score; ~1 = normal, >1 = outlier)",
            "method_family": "density_based",
        },
    )


@dataclass(frozen=True)
class KNNConfig:
    """k-Nearest Neighbours distance-based detector configuration."""

    contamination: float = 0.1
    n_neighbors: int = 5
    method: str = "median"  # 'median' or 'mean'
    metric: str = "minkowski"
    p: int = 2


def fit_predict_knn(
    frame: pd.DataFrame,
    features: list[str] | tuple[str, ...],
    config: KNNConfig | None = None,
) -> DetectionResult:
    """KNN (Median of k-NN distances).

    Computes the distance to the k-th nearest neighbour (or median/mean of
    all k distances) for each point. Larger distances indicate stronger
    isolation = more anomalous.  We return higher = more anomalous.
    """
    _ensure_pyod()
    from pyod.models.knn import KNN as PyODKNN

    config = config or KNNConfig()
    _validate_contamination(config.contamination, "KNN")
    if not isinstance(config.n_neighbors, int) or config.n_neighbors < 1:
        raise InvalidDetectorConfigError("KNN: n_neighbors must be a positive integer")
    if config.method not in ("median", "mean"):
        raise InvalidDetectorConfigError("KNN: method must be 'median' or 'mean'")

    prepared = prepare_features(frame, features)
    min_needed = config.n_neighbors + 1
    if prepared.values.shape[0] < min_needed:
        raise InsufficientDataError(
            f"KNN with n_neighbors={config.n_neighbors} requires at least "
            f"{min_needed} rows; got {prepared.values.shape[0]}"
        )

    model = PyODKNN(
        contamination=config.contamination,
        n_neighbors=config.n_neighbors,
        method=config.method,
        metric=config.metric,
        p=config.p,
    )
    model.fit(prepared.values)
    scores = model.decision_scores_
    if not np.all(np.isfinite(scores)):
        raise AnomalyLabError(
            f"KNN produced non-finite scores (NaN or inf); "
            f"input may contain invalid values. Input shape: {prepared.values.shape}"
        )
    predictions = model.predict(prepared.values)

    return DetectionResult(
        detector="knn",
        row_indices=tuple(_index_value(v) for v in prepared.index),
        is_anomaly=tuple(bool(p == 1) for p in predictions),
        features=prepared.features,
        constant_features=prepared.constant_features,
        parameters={
            "contamination": config.contamination,
            "n_neighbors": config.n_neighbors,
            "method": config.method,
            "metric": config.metric,
            "p": config.p,
        },
        scores=tuple(float(s) for s in scores),
        metadata={
            "model": "KNN",
            "score_semantics": "larger = more anomalous (k-NN distance; more isolated = outlier)",
            "method_family": "distance_based",
        },
    )


def _index_value(value: object) -> object:
    if isinstance(value, np.generic):
        return value.item()
    return value
