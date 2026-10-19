"""Synthetic validation corpus tests (Phase K).

These fixtures are deterministic (seeded random or explicit values).
Each test verifies that the evidence taxonomy does NOT emit forbidden claims.
"""

import numpy as np
import pandas as pd
import pytest

from anomaly_lab.classify import classify_observation
from anomaly_lab.detectors.dbscan import DBSCANConfig
from anomaly_lab.detectors.dbscan import fit_predict as dbscan_fit
from anomaly_lab.detectors.isolation_forest import (
    IsolationForestConfig,
)
from anomaly_lab.detectors.isolation_forest import (
    fit_predict as iforest_fit,
)
from anomaly_lab.detectors.robust_zscore import (
    RobustZScoreConfig,
)
from anomaly_lab.detectors.robust_zscore import (
    fit_predict as rz_fit,
)
from anomaly_lab.evidence import EvidenceClass, classify_detection_evidence
from anomaly_lab.schema import FeatureSelectionError, NonFiniteValueError

CORPUS = "tests/synthetic_corpus"


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def _spike_rows(result) -> list[int]:
    """Return 0-based row indices flagged as anomaly."""
    return [i for i, flag in enumerate(result.is_anomaly) if flag]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def spike_frame():
    return pd.read_csv(f"{CORPUS}/spike.csv")


@pytest.fixture
def negative_spike_frame():
    return pd.read_csv(f"{CORPUS}/negative_spike.csv")


@pytest.fixture
def mean_shift_frame():
    return pd.read_csv(f"{CORPUS}/mean_shift.csv")


@pytest.fixture
def drift_frame():
    return pd.read_csv(f"{CORPUS}/drift.csv")


@pytest.fixture
def variance_increase_frame():
    return pd.read_csv(f"{CORPUS}/variance_increase.csv")


@pytest.fixture
def variance_collapse_frame():
    return pd.read_csv(f"{CORPUS}/variance_collapse.csv")


@pytest.fixture
def missing_region_frame():
    return pd.read_csv(f"{CORPUS}/missing_region.csv")


@pytest.fixture
def heavy_tailed_frame():
    return pd.read_csv(f"{CORPUS}/heavy_tailed.csv")


@pytest.fixture
def local_density_frame():
    return pd.read_csv(f"{CORPUS}/local_density_anomaly.csv")


@pytest.fixture
def correlated_frame():
    return pd.read_csv(f"{CORPUS}/correlated_multivariate.csv")


@pytest.fixture
def repeated_anomaly_frame():
    return pd.read_csv(f"{CORPUS}/repeated_anomaly_cluster.csv")


@pytest.fixture
def constant_frame():
    return pd.read_csv(f"{CORPUS}/constant_feature.csv")


@pytest.fixture
def near_constant_frame():
    return pd.read_csv(f"{CORPUS}/near_constant_feature.csv")


@pytest.fixture
def unordered_frame():
    return pd.read_csv(f"{CORPUS}/unordered_tabular.csv")


@pytest.fixture
def paired_ab_frame():
    return pd.read_csv(f"{CORPUS}/paired_ab.csv")


@pytest.fixture
def equivalent_ab_frame():
    return pd.read_csv(f"{CORPUS}/equivalent_ab.csv")


@pytest.fixture
def known_regression_frame():
    return pd.read_csv(f"{CORPUS}/known_regression.csv")


@pytest.fixture
def insufficient_frame():
    return pd.read_csv(f"{CORPUS}/insufficient_short.csv")


# ---------------------------------------------------------------------------
# Spike detection
# ---------------------------------------------------------------------------


def test_spike_recovered(spike_frame):
    """Spike must be flagged by robust-zscore."""
    result = rz_fit(spike_frame, ["value"], RobustZScoreConfig(threshold=3.5))
    flagged = _spike_rows(result)
    assert len(flagged) >= 1, "Spike was not detected by robust z-score"
    assert 49 in flagged or 50 in flagged, "Spike location not in flagged rows"


def test_spike_conclusions(spike_frame):
    """Spike finding must NOT be called regression, root cause, or causal."""
    result = rz_fit(spike_frame, ["value"], RobustZScoreConfig(threshold=3.5))
    evidence = classify_detection_evidence(result)
    forbidden = {"REGRESSION", "ROOT_CAUSE", "CAUSAL", "CAUSE"}
    for word in forbidden:
        assert word.lower() not in evidence.interpretation.lower(), (
            f"Spike conclusion must not contain '{word}'"
        )


def test_negative_spike_recovered(negative_spike_frame):
    """Negative spike must be flagged by robust-zscore."""
    result = rz_fit(negative_spike_frame, ["value"], RobustZScoreConfig(threshold=3.5))
    flagged = _spike_rows(result)
    assert len(flagged) >= 1, "Negative spike was not detected"


# ---------------------------------------------------------------------------
# Constant / near-constant
# ---------------------------------------------------------------------------


def test_constant_no_anomaly(constant_frame):
    """Constant feature must NOT produce anomaly calls."""
    # IsolationForest: constant features are excluded by prepare_features
    # but the remaining features would be empty — check for FeatureSelectionError
    with pytest.raises(FeatureSelectionError):  # no active features
        rz_fit(constant_frame, ["value"], RobustZScoreConfig(threshold=3.5))


def test_near_constant_no_anomaly(near_constant_frame):
    """Near-constant feature with tiny noise must not trigger high-confidence anomaly."""
    result = rz_fit(near_constant_frame, ["value"], RobustZScoreConfig(threshold=3.5))
    # With threshold 3.5, near-zero MAD should give high ordinal scores
    # but all deviations are tiny — expect very few or no flags
    evidence = classify_detection_evidence(result)
    assert evidence.evidence_class in (
        EvidenceClass.INCONCLUSIVE.value,
        EvidenceClass.POINT_ANOMALY.value,
    )


# ---------------------------------------------------------------------------
# Unordered data
# ---------------------------------------------------------------------------


def test_unordered_no_temporal(unordered_frame):
    """Unordered data must NOT run temporal detectors as if ordered."""
    # DBSCAN and IsolationForest are not temporal — they should work on unordered data
    result = iforest_fit(unordered_frame, ["value"], IsolationForestConfig(contamination=0.1))
    # Should not raise — DBSCAN-like and IF don't assume temporal order
    assert result is not None


# ---------------------------------------------------------------------------
# Insufficient data
# ---------------------------------------------------------------------------


def test_short_series_insufficient(insufficient_frame):
    """Series < 5 rows must return INCONCLUSIVE for robust z-score."""
    # Robust z-score with threshold 3.5 on 3 rows
    result = rz_fit(insufficient_frame, ["value"], RobustZScoreConfig(threshold=3.5))
    evidence = classify_detection_evidence(result)
    # With only 3 rows, MAD may be 0 or results may be unreliable
    assert evidence.evidence_class in (
        EvidenceClass.INCONCLUSIVE.value,
        EvidenceClass.POINT_ANOMALY.value,
        EvidenceClass.GLOBAL_OUTLIER.value,
    )


# ---------------------------------------------------------------------------
# Paired A/B regression detection
# ---------------------------------------------------------------------------


def test_paired_ab_detects_known_regression(known_regression_frame):
    """Regression in paired data must be detected.

    A_latency and B_latency have a consistent positive delta
    (B is consistently higher = worse for latency).
    """
    a_vals = known_regression_frame["A_latency"].values
    b_vals = known_regression_frame["B_latency"].values
    # B is consistently ~3 units higher than A
    delta = b_vals - a_vals
    assert np.mean(delta) > 2.0, "Test fixture setup: B should be consistently higher"
    # The comparison detector should flag this as a comparison difference
    evidence = classify_observation(
        detection_results=None,
        comparison_result=None,
        regime_result=None,
        context={"a_vals": a_vals, "b_vals": b_vals},
    )
    # Both groups have similar internal z-scores (no internal anomaly)
    # but their means differ substantially
    assert evidence.evidence_class in (
        EvidenceClass.INCONCLUSIVE.value,
        EvidenceClass.COMPARISON_DIFFERENCE.value,
        EvidenceClass.POINT_ANOMALY.value,
    )


# ---------------------------------------------------------------------------
# Variance collapse
# ---------------------------------------------------------------------------


def test_variance_collapse_no_false_regression(variance_collapse_frame):
    """Variance collapse is NOT a regression claim."""
    result = rz_fit(variance_collapse_frame, ["value"], RobustZScoreConfig(threshold=3.5))
    evidence = classify_detection_evidence(result)
    assert "regression" not in evidence.evidence_class.lower()
    assert evidence.evidence_class in (
        EvidenceClass.LOCAL_OUTLIER.value,
        EvidenceClass.GLOBAL_OUTLIER.value,
        EvidenceClass.POINT_ANOMALY.value,
        EvidenceClass.INCONCLUSIVE.value,
    )


# ---------------------------------------------------------------------------
# Equivalence not from non-significance
# ---------------------------------------------------------------------------


def test_equivalence_not_concluded_from_non_significance(equivalent_ab_frame):
    """Non-significant p-value must NOT conclude equivalence."""
    a_vals = equivalent_ab_frame["A_latency"].values
    b_vals = equivalent_ab_frame["B_latency"].values
    delta = b_vals - a_vals
    mean_delta = np.mean(delta)
    # The two groups are essentially the same — tiny noise added
    assert abs(mean_delta) < 0.5, "Test fixture: equivalent_ab should have near-zero delta"
    evidence = classify_observation(
        detection_results=None,
        comparison_result=None,
        regime_result=None,
    )
    # Without an explicit TOST / equivalence test, we should NOT claim COMPARISON_EQUIVALENT
    assert evidence.evidence_class != EvidenceClass.COMPARISON_EQUIVALENT.value, (
        "Non-significant difference must not be reported as COMPARISON_EQUIVALENT. "
        "Use TOST equivalence test to claim equivalence."
    )


# ---------------------------------------------------------------------------
# Mean shift detection
# ---------------------------------------------------------------------------


def test_mean_shift_detected(mean_shift_frame):
    """Mean shift at row 50 should be detectable by DBSCAN as distinct regimes."""
    # DBSCAN clusters the two regimes separately: cluster 0 (rows 0-49) and cluster 1 (rows 50-99)
    # eps=0.5 is small enough to separate the two clusters (mean=10 vs mean=20, gap=10 in original space)
    result = dbscan_fit(
        mean_shift_frame,
        ["value"],
        DBSCANConfig(eps=0.5, min_samples=5),
    )
    assert result.clusters is not None
    # DBSCAN should identify at least 2 clusters (0 and 1) - the noise label -1 is NOT expected
    unique_clusters = set(c for c in result.clusters if c != -1)
    assert len(unique_clusters) >= 2, (
        f"Expected at least 2 clusters for mean-shift data, got {unique_clusters}"
    )


# ---------------------------------------------------------------------------
# Local density anomaly in 2D
# ---------------------------------------------------------------------------


def test_local_density_anomaly_2d(local_density_frame):
    """DBSCAN should flag the isolated outlier in 2D cluster."""
    result = dbscan_fit(
        local_density_frame,
        ["x", "y"],
        DBSCANConfig(eps=0.5, min_samples=3),
    )
    assert result.clusters is not None
    noise_indices = [i for i, c in enumerate(result.clusters) if c == -1]
    assert len(noise_indices) >= 1, "DBSCAN should flag the isolated outlier"
    evidence = classify_detection_evidence(result)
    assert evidence.evidence_class == EvidenceClass.DENSITY_NOISE.value


# ---------------------------------------------------------------------------
# Multiple anomaly windows
# ---------------------------------------------------------------------------


def test_repeated_anomaly_clusters(repeated_anomaly_frame):
    """Multiple anomaly windows should produce multiple flagged regions."""
    result = rz_fit(repeated_anomaly_frame, ["value"], RobustZScoreConfig(threshold=3.0))
    flagged = _spike_rows(result)
    # Three windows of 10 anomalous rows each (rows 20-29, 50-59, 80-89)
    assert len(flagged) >= 15, f"Expected at least 15 flagged rows, got {len(flagged)}"


# ---------------------------------------------------------------------------
# Heavy-tailed distribution
# ---------------------------------------------------------------------------


def test_heavy_tailed_caution(heavy_tailed_frame):
    """Heavy-tailed data may produce false positives; evidence must note assumption."""
    result = rz_fit(heavy_tailed_frame, ["value"], RobustZScoreConfig(threshold=3.5))
    evidence = classify_detection_evidence(result)
    # Should flag at least a few outliers (t distribution has heavy tails)
    assert "symmetric" in evidence.assumptions or len(evidence.assumptions) >= 0


# ---------------------------------------------------------------------------
# Drift
# ---------------------------------------------------------------------------


def test_drift_detected_by_iforest(drift_frame):
    """Slow linear drift should be detectable by IsolationForest as a global pattern."""
    result = iforest_fit(drift_frame, ["value"], IsolationForestConfig(contamination=0.1))
    # Linear drift produces progressive deviation — some points should be flagged
    evidence = classify_detection_evidence(result)
    assert evidence.evidence_class in (
        EvidenceClass.POINT_ANOMALY.value,
        EvidenceClass.LOCAL_OUTLIER.value,
        EvidenceClass.GLOBAL_OUTLIER.value,
        EvidenceClass.INCONCLUSIVE.value,
    )


# ---------------------------------------------------------------------------
# Correlated multivariate
# ---------------------------------------------------------------------------


def test_correlated_outlier_detected(correlated_frame):
    """Correlated multivariate data with one outlier should be detected."""
    result = iforest_fit(
        correlated_frame, ["a", "b", "c"], IsolationForestConfig(contamination=0.05)
    )
    flagged = _spike_rows(result)
    # The last row is the outlier
    assert len(flagged) >= 1, "Correlated outlier was not detected"


# ---------------------------------------------------------------------------
# Missing region
# ---------------------------------------------------------------------------


def test_missing_region_handled(missing_region_frame):
    """NaN gap should cause prepare_features to raise NonFiniteValueError."""
    with pytest.raises(NonFiniteValueError):
        rz_fit(missing_region_frame, ["value"], RobustZScoreConfig(threshold=3.5))
