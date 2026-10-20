"""Phase 2: TDD compatibility tests for PyOD adapters.

These tests verify that the ECOD, LOF, and KNN adapters work correctly
against the installed version of PyOD (no mocking).
"""

import numpy as np
import pandas as pd
import pytest

from anomaly_lab.detectors.pyod_adapters import (
    ECODConfig,
    KNNConfig,
    LOFConfig,
    fit_predict_ecod,
    fit_predict_knn,
    fit_predict_lof,
)
from anomaly_lab.schema import InvalidDetectorConfigError

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

RNG = np.random.default_rng(42)


def _normal_frame(n=100, seed=0) -> pd.DataFrame:
    """Clean tabular data with one clear outlier."""
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, 3))
    X[-1] = X[-1] * 10 + np.array([8.0, 9.0, 7.0])  # outlier
    return pd.DataFrame(X, columns=["a", "b", "c"])


def _constant_frame() -> pd.DataFrame:
    """Frame with a constant feature column."""
    rng = np.random.default_rng(42)
    return pd.DataFrame(
        {
            "x": [1.0] * 25,
            "y": list(rng.standard_normal(25)),
        }
    )


# ---------------------------------------------------------------------------
# ECOD — constructor / instantiation
# ---------------------------------------------------------------------------


class TestEcodInstantiation:
    """ECOD should be instantiable with the adapter's default config."""

    def test_ecod_instantiation_does_not_raise(self):
        """ECOD(contamination=0.1) must not raise TypeError."""
        # This is the primary defect: adapter passes n_bins which doesn't exist
        config = ECODConfig()
        # The fit_predict call is what triggers ECOD instantiation
        frame = _normal_frame(n=20, seed=0)
        result = fit_predict_ecod(frame, ["a", "b", "c"], config)
        assert result.detector == "ecod"

    def test_ecod_n_bins_is_not_a_valid_parameter(self):
        """ECOD in the installed PyOD does not accept n_bins."""
        import inspect

        from pyod.models.ecod import ECOD

        params = list(inspect.signature(ECOD.__init__).parameters)
        assert "n_bins" not in params, "ECOD no longer accepts n_bins — adapter must be updated"

    def test_ecod_contamination_upper_bound_is_05(self):
        """PyOD ECOD raises if contamination > 0.5."""
        from pyod.models.ecod import ECOD

        frame = _normal_frame(n=20, seed=0)
        # Adapter's own validation will catch this too, but this
        # test documents the actual PyOD boundary
        with pytest.raises(ValueError, match="contamination must be"):
            ECOD(contamination=0.6).fit(frame[["a"]].values)


# ---------------------------------------------------------------------------
# ECOD — fit / predict
# ---------------------------------------------------------------------------


class TestEcodFitPredict:
    """ECOD fit_predict smoke tests."""

    def test_ecod_returns_detection_result(self):
        frame = _normal_frame(n=50, seed=1)
        result = fit_predict_ecod(frame, ["a", "b", "c"])
        assert result.detector == "ecod"
        assert len(result.scores) == 50
        assert len(result.is_anomaly) == 50
        assert isinstance(result.scores, tuple)
        assert isinstance(result.is_anomaly, tuple)

    def test_ecod_outlier_gets_higher_score(self):
        """Outlier row should have higher score than normal rows."""
        frame = _normal_frame(n=50, seed=2)
        result = fit_predict_ecod(frame, ["a", "b", "c"])
        outlier_score = result.scores[-1]
        normal_scores = list(result.scores[:-1])
        assert outlier_score > np.median(normal_scores), (
            "Outlier (last row) should have higher score than median normal score"
        )

    def test_ecod_scores_are_all_finite(self):
        frame = _normal_frame(n=50, seed=3)
        result = fit_predict_ecod(frame, ["a", "b", "c"])
        assert all(np.isfinite(s) for s in result.scores)

    def test_ecod_row_indices_preserved(self):
        """Row indices must match the input frame's index."""
        frame = _normal_frame(n=30, seed=4)
        frame.index = [f"row_{i:03d}" for i in range(30)]
        result = fit_predict_ecod(frame, ["a", "b", "c"])
        assert result.row_indices == tuple(frame.index)

    def test_ecod_higher_score_means_more_anomalous(self):
        """Verify score direction: larger = more anomalous for ECOD."""
        frame = _normal_frame(n=50, seed=5)
        result = fit_predict_ecod(frame, ["a", "b", "c"])
        # is_anomaly should be True for the row with the highest score
        max_score_idx = result.scores.index(max(result.scores))
        assert result.is_anomaly[max_score_idx], "Row with max score should be flagged as anomaly"

    def test_ecod_contamination_validated_by_adapter(self):
        """Adapter should reject contamination outside (0, 0.5]."""
        frame = _normal_frame(n=20, seed=0)
        with pytest.raises(InvalidDetectorConfigError):
            fit_predict_ecod(frame, ["a", "b", "c"], ECODConfig(contamination=0.0))
        with pytest.raises(InvalidDetectorConfigError):
            fit_predict_ecod(frame, ["a", "b", "c"], ECODConfig(contamination=0.6))


# ---------------------------------------------------------------------------
# LOF — constructor / instantiation
# ---------------------------------------------------------------------------


class TestLofInstantiation:
    """LOF should be instantiable with the adapter's config."""

    def test_lof_instantiation_does_not_raise(self):
        config = LOFConfig()
        frame = _normal_frame(n=25, seed=0)
        result = fit_predict_lof(frame, ["a", "b", "c"], config)
        assert result.detector == "lof"

    def test_lof_contamination_upper_bound_is_05(self):
        """PyOD LOF raises if contamination > 0.5."""
        from pyod.models.lof import LOF

        frame = _normal_frame(n=20, seed=0)
        with pytest.raises(ValueError, match="contamination must be"):
            LOF(contamination=0.6).fit(frame[["a"]].values)


# ---------------------------------------------------------------------------
# LOF — fit / predict
# ---------------------------------------------------------------------------


class TestLofFitPredict:
    """LOF fit_predict smoke tests."""

    def test_lof_returns_detection_result(self):
        frame = _normal_frame(n=50, seed=1)
        result = fit_predict_lof(frame, ["a", "b", "c"])
        assert result.detector == "lof"
        assert len(result.scores) == 50
        assert len(result.is_anomaly) == 50

    def test_lof_outlier_gets_higher_score(self):
        frame = _normal_frame(n=50, seed=2)
        result = fit_predict_lof(frame, ["a", "b", "c"])
        outlier_score = result.scores[-1]
        normal_scores = list(result.scores[:-1])
        assert outlier_score > np.median(normal_scores)

    def test_lof_scores_are_all_finite(self):
        frame = _normal_frame(n=50, seed=3)
        result = fit_predict_lof(frame, ["a", "b", "c"])
        assert all(np.isfinite(s) for s in result.scores)

    def test_lof_row_indices_preserved(self):
        frame = _normal_frame(n=30, seed=4)
        frame.index = [f"row_{i:03d}" for i in range(30)]
        result = fit_predict_lof(frame, ["a", "b", "c"])
        assert result.row_indices == tuple(frame.index)

    def test_lof_higher_score_means_more_anomalous(self):
        frame = _normal_frame(n=50, seed=5)
        result = fit_predict_lof(frame, ["a", "b", "c"])
        max_score_idx = result.scores.index(max(result.scores))
        assert result.is_anomaly[max_score_idx]

    def test_lof_contamination_validated_by_adapter(self):
        frame = _normal_frame(n=20, seed=0)
        with pytest.raises(InvalidDetectorConfigError):
            fit_predict_lof(frame, ["a", "b", "c"], LOFConfig(contamination=0.0))
        with pytest.raises(InvalidDetectorConfigError):
            fit_predict_lof(frame, ["a", "b", "c"], LOFConfig(contamination=0.6))

    def test_lof_n_neighbors_validated(self):
        frame = _normal_frame(n=20, seed=0)
        with pytest.raises(InvalidDetectorConfigError):
            fit_predict_lof(frame, ["a", "b", "c"], LOFConfig(n_neighbors=0))


# ---------------------------------------------------------------------------
# KNN — constructor / instantiation
# ---------------------------------------------------------------------------


class TestKnnInstantiation:
    """KNN should be instantiable with the adapter's config."""

    def test_knn_instantiation_does_not_raise(self):
        config = KNNConfig()
        frame = _normal_frame(n=20, seed=0)
        result = fit_predict_knn(frame, ["a", "b", "c"], config)
        assert result.detector == "knn"

    def test_knn_contamination_upper_bound_is_05(self):
        """PyOD KNN raises if contamination > 0.5."""
        from pyod.models.knn import KNN

        frame = _normal_frame(n=20, seed=0)
        with pytest.raises(ValueError, match="contamination must be"):
            KNN(contamination=0.6).fit(frame[["a"]].values)


# ---------------------------------------------------------------------------
# KNN — fit / predict
# ---------------------------------------------------------------------------


class TestKnnFitPredict:
    """KNN fit_predict smoke tests."""

    def test_knn_returns_detection_result(self):
        frame = _normal_frame(n=50, seed=1)
        result = fit_predict_knn(frame, ["a", "b", "c"])
        assert result.detector == "knn"
        assert len(result.scores) == 50
        assert len(result.is_anomaly) == 50

    def test_knn_outlier_gets_higher_score(self):
        frame = _normal_frame(n=50, seed=2)
        result = fit_predict_knn(frame, ["a", "b", "c"])
        outlier_score = result.scores[-1]
        normal_scores = list(result.scores[:-1])
        assert outlier_score > np.median(normal_scores)

    def test_knn_scores_are_all_finite(self):
        frame = _normal_frame(n=50, seed=3)
        result = fit_predict_knn(frame, ["a", "b", "c"])
        assert all(np.isfinite(s) for s in result.scores)

    def test_knn_row_indices_preserved(self):
        frame = _normal_frame(n=30, seed=4)
        frame.index = [f"row_{i:03d}" for i in range(30)]
        result = fit_predict_knn(frame, ["a", "b", "c"])
        assert result.row_indices == tuple(frame.index)

    def test_knn_higher_score_means_more_anomalous(self):
        frame = _normal_frame(n=50, seed=5)
        result = fit_predict_knn(frame, ["a", "b", "c"])
        max_score_idx = result.scores.index(max(result.scores))
        assert result.is_anomaly[max_score_idx]

    def test_knn_contamination_validated_by_adapter(self):
        frame = _normal_frame(n=20, seed=0)
        with pytest.raises(InvalidDetectorConfigError):
            fit_predict_knn(frame, ["a", "b", "c"], KNNConfig(contamination=0.0))
        with pytest.raises(InvalidDetectorConfigError):
            fit_predict_knn(frame, ["a", "b", "c"], KNNConfig(contamination=0.6))

    def test_knn_n_neighbors_validated(self):
        frame = _normal_frame(n=20, seed=0)
        with pytest.raises(InvalidDetectorConfigError):
            fit_predict_knn(frame, ["a", "b", "c"], KNNConfig(n_neighbors=0))

    def test_knn_method_validated(self):
        frame = _normal_frame(n=20, seed=0)
        with pytest.raises(InvalidDetectorConfigError):
            fit_predict_knn(frame, ["a", "b", "c"], KNNConfig(method="invalid"))


# ---------------------------------------------------------------------------
# Shared behavior
# ---------------------------------------------------------------------------


class TestSharedBehavior:
    """Tests common to all three adapters."""

    def test_all_detectors_flag_some_anomalies(self):
        """All detectors should flag anomalies when contamination=0.1 on a
        50-sample frame. This verifies the PyOD models are correctly
        instantiated, fitted, and produce anomaly labels."""
        frame = _normal_frame(n=50, seed=99)
        for fit_fn, name in [
            (fit_predict_ecod, "ecod"),
            (fit_predict_lof, "lof"),
            (fit_predict_knn, "knn"),
        ]:
            result = fit_fn(frame, ["a", "b", "c"])
            # With contamination=0.1, at least some points should be flagged
            flagged_count = sum(result.is_anomaly)
            assert flagged_count >= 1, (
                f"{name} flagged no anomalies at contamination=0.1 — "
                f"check PyOD model instantiation and predict()"
            )
            # Score list must be non-empty and ordered by row index
            assert len(result.scores) == 50
            # Higher score must correspond to is_anomaly=True
            max_score_idx = result.scores.index(max(result.scores))
            assert result.is_anomaly[max_score_idx], (
                f"{name}: highest-scoring row not flagged as anomaly"
            )

    def test_constant_feature_does_not_crash(self):
        """Adapter should handle constant features gracefully (not crash)."""
        frame = _constant_frame()
        # Should not raise — constant column gets handled
        for fit_fn in [fit_predict_ecod, fit_predict_lof, fit_predict_knn]:
            try:
                result = fit_fn(frame, ["x", "y"])
                assert len(result.scores) == 25, (
                    f"{fit_fn.__name__}: expected 25 scores (frame rows), got {len(result.scores)}"
                )
            except Exception as exc:
                pytest.fail(f"{fit_fn.__name__} crashed on constant feature: {exc}")

    def test_insufficient_samples_behavior(self):
        """Adapters must either return a valid DetectionResult or raise a
        typed AnomalyLab error for insufficient data — not silently wrong output
        and not an uncontrolled crash."""
        tiny = pd.DataFrame({"x": [1.0, 2.0]})
        from anomaly_lab.schema import AnomalyLabError

        for fit_fn, name in [
            (fit_predict_ecod, "ecod"),
            (fit_predict_lof, "lof"),
            (fit_predict_knn, "knn"),
        ]:
            try:
                result = fit_fn(tiny, ["x"])
                # If it returns a result, it must be valid
                assert isinstance(result.scores, tuple), (
                    f"{name}: scores must be a tuple, got {type(result.scores).__name__}"
                )
                assert len(result.scores) == 2, (
                    f"{name}: expected 2 scores for 2-row frame, got {len(result.scores)}"
                )
                assert all(isinstance(s, float) for s in result.scores), (
                    f"{name}: all scores must be floats"
                )
            except AnomalyLabError:
                # Typed AnomalyLab error is acceptable — data is genuinely insufficient
                pass
            except Exception as exc:
                pytest.fail(
                    f"{name} raised {type(exc).__name__} on tiny (2-row) data: {exc}. "
                    "Expected either a valid DetectionResult or a typed AnomalyLabError."
                )

    def test_scores_tuple_of_floats(self):
        """Scores must be a tuple of plain floats for JSON serialization."""
        frame = _normal_frame(n=25, seed=7)
        for fit_fn, name in [
            (fit_predict_ecod, "ecod"),
            (fit_predict_lof, "lof"),
            (fit_predict_knn, "knn"),
        ]:
            result = fit_fn(frame, ["a", "b", "c"])
            assert isinstance(result.scores, tuple)
            for s in result.scores:
                assert isinstance(s, float), f"{name} score {s!r} is not a float"


def test_ecod_runner_uses_ecod_class():
    """Regression: run_ecod must use pyod.models.ecod.ECOD, not PCA.

    Prior bug: run_ecod was implemented as `clf = PyODPCA(...)` — the benchmark
    labeled PCA results as ECOD. This test proves the identity collision is resolved.
    """
    import importlib.util
    import os
    import sys

    import numpy as np
    import pandas as pd
    from pyod.models.ecod import ECOD
    from pyod.models.pca import PCA

    repo_root = os.path.join(os.path.dirname(__file__), "..")
    sys.path.insert(0, repo_root)
    spec = importlib.util.spec_from_file_location(
        "pyod_candidates_test", os.path.join(repo_root, "benchmarks", "pyod_candidates.py")
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    run_ecod = mod.run_ecod

    rng = np.random.default_rng(12345)
    X = rng.standard_normal((60, 4))
    df = pd.DataFrame(X, columns=["a", "b", "c", "d"])

    ecod_scores, _ = run_ecod(df, ["a", "b", "c", "d"])

    # Ground truth: ECOD and PCA produce different scores on identical data
    ecod_ref = ECOD(contamination=0.1).fit(X)
    pca_ref = PCA(contamination=0.1).fit(X)

    assert not np.allclose(ecod_scores, pca_ref.decision_scores_, rtol=0.05), (
        "run_ecod produced PCA-identical scores — ECOD runner is still broken"
    )
    assert np.allclose(ecod_scores, ecod_ref.decision_scores_, rtol=0.001), (
        "run_ecod scores do not match ECOD ground truth"
    )


def test_pca_runner_uses_pca_class():
    """Regression: run_pca must use pyod.models.pca.PCA, not ECOD."""
    import importlib.util
    import os
    import sys

    import numpy as np
    import pandas as pd
    from pyod.models.ecod import ECOD
    from pyod.models.pca import PCA

    repo_root = os.path.join(os.path.dirname(__file__), "..")
    sys.path.insert(0, repo_root)
    spec = importlib.util.spec_from_file_location(
        "pyod_candidates_test", os.path.join(repo_root, "benchmarks", "pyod_candidates.py")
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    run_pca = mod.run_pca

    rng = np.random.default_rng(54321)
    X = rng.standard_normal((60, 4))
    df = pd.DataFrame(X, columns=["a", "b", "c", "d"])

    pca_scores, _ = run_pca(df, ["a", "b", "c", "d"])

    ecod_ref = ECOD(contamination=0.1).fit(X)
    pca_ref = PCA(contamination=0.1).fit(X)

    assert not np.allclose(pca_scores, ecod_ref.decision_scores_, rtol=0.05), (
        "run_pca produced ECOD-identical scores — runner is mislabeled"
    )
    assert np.allclose(pca_scores, pca_ref.decision_scores_, rtol=0.001), (
        "run_pca scores do not match PCA ground truth"
    )


def _import_safe_spearman():
    """Load safe_spearman from the local benchmarks directory."""
    import importlib.util
    import os
    import sys

    repo_root = os.path.join(os.path.dirname(__file__), "..")
    sys.path.insert(0, repo_root)
    spec = importlib.util.spec_from_file_location(
        "pyod_candidates_sp", os.path.join(repo_root, "benchmarks", "pyod_candidates.py")
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.safe_spearman


def test_safe_spearman_per_detector_scores_differ():
    """Different detector score vectors must produce different Spearman values."""
    safe_spearman = _import_safe_spearman()
    rng = np.random.default_rng(999)
    ground_truth = rng.standard_normal(50)
    scores_a = rng.standard_normal(50)
    scores_b = rng.standard_normal(50)
    rho_a = safe_spearman(ground_truth, scores_a)
    rho_b = safe_spearman(ground_truth, scores_b)
    assert rho_a != rho_b, (
        "Two different random score vectors gave identical Spearman — "
        "safe_spearman is not using detector scores"
    )


def test_safe_spearman_replacement_invariant():
    """Replacing detector scores MUST change the Spearman value."""
    safe_spearman = _import_safe_spearman()
    ground_truth = np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
    detector_scores = np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
    rho_original = safe_spearman(ground_truth, detector_scores)
    replacement = np.array([0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0])
    rho_replaced = safe_spearman(ground_truth, replacement)
    assert rho_replaced != rho_original, (
        "Replacing detector scores left Spearman unchanged — "
        "safe_spearman is not correlating detector scores"
    )


def test_safe_spearman_perfect_monotonic():
    """Perfect monotonic detector scores must give rho ≈ 1."""
    safe_spearman = _import_safe_spearman()
    ground_truth = np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
    detector_scores = np.array([0.05, 0.12, 0.19, 0.31, 0.41, 0.52, 0.61, 0.72, 0.81, 0.91])
    rho = safe_spearman(ground_truth, detector_scores)
    assert 0.99 <= rho <= 1.01, f"Perfect monotonic detector gave rho={rho}, expected ~1.0"


def test_safe_spearman_reversed():
    """Reversed detector scores must give rho ≈ -1."""
    safe_spearman = _import_safe_spearman()
    ground_truth = np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
    detector_scores = np.array([0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0])
    rho = safe_spearman(ground_truth, detector_scores)
    assert -1.01 <= rho <= -0.99, f"Reversed detector scores gave rho={rho}, expected ~-1.0"


def test_safe_spearman_constant_scores():
    """Constant detector scores must produce NaN (not 0.0)."""
    safe_spearman = _import_safe_spearman()
    ground_truth = np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
    constant_scores = np.ones(10)
    rho = safe_spearman(ground_truth, constant_scores)
    assert np.isnan(rho), f"Constant detector scores gave rho={rho}, expected NaN"


def test_safe_spearman_unexpected_exception_propagates():
    """Unexpected type errors must NOT be silently caught and turned into NaN."""
    safe_spearman = _import_safe_spearman()
    try:
        safe_spearman("not an array", [1, 2, 3])
    except (TypeError, ValueError):
        pass
    else:
        raise AssertionError("Expected TypeError or ValueError for incompatible types")


def test_safe_spearman_ground_truth_vs_detector_scores():
    """safe_spearman must correlate ground_truth severity with detector scores.

    Prior bug: safe_spearman(y, anomaly_scores) was called — correlating
    ground truth label y with ground truth severity anomaly_scores,
    giving every detector the SAME rho value. The correct call is
    safe_spearman(anomaly_scores, detector_scores).
    """
    safe_spearman = _import_safe_spearman()
    n = 20
    rng = np.random.default_rng(42)
    anomaly_scores = np.concatenate(
        [
            rng.uniform(0.0, 0.2, 10),
            rng.uniform(0.8, 1.0, 10),
        ]
    )
    detector_scores_a = anomaly_scores + rng.standard_normal(n) * 0.05
    detector_scores_b = 1.0 - anomaly_scores + rng.standard_normal(n) * 0.05
    rho_a = safe_spearman(anomaly_scores, detector_scores_a)
    rho_b = safe_spearman(anomaly_scores, detector_scores_b)
    assert rho_a > 0.9, f"Perfect detector should give rho > 0.9, got {rho_a}"
    assert rho_b < -0.9, f"Inverted detector should give rho < -0.9, got {rho_b}"
    assert abs(rho_a - rho_b) > 1.5, (
        f"Positive and negative detectors must give very different rho values, got {rho_a} vs {rho_b}"
    )


def _import_benchmark_runner(name):
    """Import a runner function from the local benchmarks directory."""
    import importlib.util
    import os
    import sys

    repo_root = os.path.join(os.path.dirname(__file__), "..")
    sys.path.insert(0, repo_root)
    spec = importlib.util.spec_from_file_location(
        f"pyod_candidates_{name}", os.path.join(repo_root, "benchmarks", "pyod_candidates.py")
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return getattr(mod, name)
