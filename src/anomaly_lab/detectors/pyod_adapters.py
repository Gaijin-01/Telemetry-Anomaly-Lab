"""PyOD-based detector adapters (Phase L).

PyOD (Probability Outlier Detection) is an optional extra.
These adapters are only functional when pyod is installed.

Detector semantics
------------------
- ECOD  : Empirical Cumulative Distribution-based Outlier Detection.
          Lower score = more anomalous (inverted before returning).
- LOF   : Local Outlier Factor (k-NN LOF).  Lower score = more anomalous (inverted).
- KNN   : Median of k-NN distances.          Lower score = more anomalous (inverted).

All three produce higher = more anomalous scores consistent with
the rest of the anomaly-lab detector interface.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from anomaly_lab.features import prepare_features
from anomaly_lab.results import DetectionResult
from anomaly_lab.schema import InvalidDetectorConfigError

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


@dataclass(frozen=True)
class ECODConfig:
    """ECOD configuration."""

    contamination: float = 0.1
    n_bins: int = 10


def fit_predict_ecod(
    frame: pd.DataFrame,
    features: list[str] | tuple[str, ...],
    config: ECODConfig | None = None,
) -> DetectionResult:
    """ECOD (Empirical Cumulative Distribution-based Outlier Detection).

    ECOD computes tail probabilities based on empirical CDFs.
    Lower raw score = more anomalous; we invert to higher = more anomalous.
    """
    _ensure_pyod()
    from pyod.models.ecod import ECOD as PyODECOD

    config = config or ECODConfig()
    if not isinstance(config.contamination, (int, float)) or not (0 < config.contamination <= 1):
        raise InvalidDetectorConfigError("contamination must be in (0, 1]")
    if not isinstance(config.n_bins, int) or config.n_bins < 1:
        raise InvalidDetectorConfigError("n_bins must be a positive integer")

    prepared = prepare_features(frame, features)

    model = PyODECOD(contamination=config.contamination, n_bins=config.n_bins)
    model.fit(prepared.values)
    # ECOD: lower raw scores are more anomalous
    raw_scores = model.decision_scores_  # shape (n_samples,)
    # Invert: higher = more anomalous
    max_raw = float(np.max(raw_scores))
    if max_raw == 0:
        scores = np.zeros_like(raw_scores)
    else:
        scores = raw_scores  # already oriented: higher in raw = less anomalous
        # Actually ECOD's decision_scores_ are already "outlier scores" where
        # larger = more anomalous. Let us verify the direction.
        # Per pyod docs: decision_scores_ are raw anomaly scores (larger = more outlying).
        # So we use them directly as our scores.
    predictions = model.predict(prepared.values)  # 0 = inlier, 1 = outlier

    return DetectionResult(
        detector="ecod",
        row_indices=tuple(_index_value(v) for v in prepared.index),
        is_anomaly=tuple(bool(p == 1) for p in predictions),
        features=prepared.features,
        constant_features=prepared.constant_features,
        parameters={"contamination": config.contamination, "n_bins": config.n_bins},
        scores=tuple(float(s) for s in scores),
        metadata={
            "model": "ECOD",
            "score_semantics": "larger = more anomalous (raw pyod decision score)",
        },
    )


@dataclass(frozen=True)
class LOFConfig:
    """Local Outlier Factor configuration."""

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
    We return higher = more anomalous.
    """
    _ensure_pyod()
    from pyod.models.lof import LOF as PyODLOF

    config = config or LOFConfig()
    if not isinstance(config.n_neighbors, int) or config.n_neighbors < 1:
        raise InvalidDetectorConfigError("n_neighbors must be a positive integer")
    if not isinstance(config.contamination, (int, float)) or not (0 < config.contamination <= 1):
        raise InvalidDetectorConfigError("contamination must be in (0, 1]")

    prepared = prepare_features(frame, features)

    model = PyODLOF(
        contamination=config.contamination,
        n_neighbors=config.n_neighbors,
        metric=config.metric,
        p=config.p,
    )
    model.fit(prepared.values)
    # LOF decision_scores_: larger = more anomalous
    scores = model.decision_scores_
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
    isolation = more anomalous.
    """
    _ensure_pyod()
    from pyod.models.knn import KNN as PyODKNN

    config = config or KNNConfig()
    if not isinstance(config.n_neighbors, int) or config.n_neighbors < 1:
        raise InvalidDetectorConfigError("n_neighbors must be a positive integer")
    if not isinstance(config.contamination, (int, float)) or not (0 < config.contamination <= 1):
        raise InvalidDetectorConfigError("contamination must be in (0, 1]")
    if config.method not in ("median", "mean"):
        raise InvalidDetectorConfigError("method must be 'median' or 'mean'")

    prepared = prepare_features(frame, features)

    model = PyODKNN(
        contamination=config.contamination,
        n_neighbors=config.n_neighbors,
        method=config.method,
        metric=config.metric,
        p=config.p,
    )
    model.fit(prepared.values)
    # KNN decision_scores_: larger = more anomalous (larger distance to neighbours)
    scores = model.decision_scores_
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
        },
    )


def _index_value(value: object) -> object:
    if isinstance(value, np.generic):
        return value.item()
    return value
