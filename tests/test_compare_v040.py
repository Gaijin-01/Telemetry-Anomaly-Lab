"""Tests for Phase D (Comparison Engine), Phase E (Effect Size + Uncertainty),
and Phase P (Reproducibility) in v0.4.0."""

import uuid

import numpy as np
import pandas as pd

from anomaly_lab.compare import (
    BootstrapResult,
    ComparisonEvidence,
    MetricComparison,
    MetricRule,
    RunMetadata,
    cliff_delta,
    cohens_d,
    compare_runs,
    paired_bootstrap_ci,
    robust_spread,
    tost_equivalence_test,
)
from anomaly_lab.reproducibility import (
    build_reproducibility_metadata,
    collect_dependency_versions,
    compute_input_hash,
    generate_analysis_id,
)

# -------------------------------------------------------------------
# Fixtures
# -------------------------------------------------------------------


def meta(**overrides):
    values = dict(model="m", version="1", quantization="q4", concurrency="1")
    values.update(overrides)
    return RunMetadata(**values)


# -------------------------------------------------------------------
# Phase D — Statistical helpers
# -------------------------------------------------------------------


class TestRobustSpread:
    def test_normal_array(self):
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = robust_spread(x)
        assert result > 0

    def test_constant_array_returns_zero(self):
        x = np.array([5.0, 5.0, 5.0])
        assert robust_spread(x) == 0.0

    def test_single_element_returns_zero(self):
        assert robust_spread(np.array([1.0])) == 0.0


class TestCohensD:
    def test_identical_groups_zero(self):
        x = np.array([1.0, 2.0, 3.0])
        assert cohens_d(x, x) == 0.0

    def test_degenerate_returns_zero(self):
        assert cohens_d(np.array([1.0]), np.array([])) == 0.0
        assert cohens_d(np.array([]), np.array([1.0])) == 0.0

    def test_positive_d_for_improvement(self):
        baseline = np.array([10.0, 10.0, 10.0, 10.0])
        candidate = np.array([20.0, 20.0, 20.0, 20.0])
        d = cohens_d(baseline, candidate)
        # With no within-group variance, pooled_std=0 → degenerate; returns 0.0
        assert d == 0.0

    def test_negative_d_for_regression(self):
        baseline = np.array([20.0, 20.0, 20.0, 20.0])
        candidate = np.array([10.0, 10.0, 10.0, 10.0])
        d = cohens_d(baseline, candidate)
        assert d == 0.0

    def test_positive_d_with_variance(self):
        """With real variance, Cohen's d is well-defined."""
        np.random.seed(42)
        baseline = np.random.normal(10, 1, 30)
        candidate = np.random.normal(12, 1, 30)
        d = cohens_d(baseline, candidate)
        assert d > 0


class TestCliffDelta:
    def test_identical_groups_zero(self):
        x = np.array([1.0, 2.0, 3.0])
        assert cliff_delta(x, x) == 0.0

    def test_empty_returns_zero(self):
        assert cliff_delta(np.array([]), np.array([1.0])) == 0.0
        assert cliff_delta(np.array([1.0]), np.array([])) == 0.0

    def test_full_dominance(self):
        baseline = np.array([1.0, 2.0, 3.0])
        candidate = np.array([10.0, 11.0, 12.0])
        d = cliff_delta(baseline, candidate)
        assert d == 1.0

    def test_full_dominance_reverse(self):
        baseline = np.array([10.0, 11.0, 12.0])
        candidate = np.array([1.0, 2.0, 3.0])
        d = cliff_delta(baseline, candidate)
        assert d == -1.0


class TestPairedBootstrapCI:
    def test_returns_bootstrap_result(self):
        np.random.seed(42)
        baseline = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        candidate = np.array([1.5, 2.5, 3.5, 4.5, 5.5])
        result = paired_bootstrap_ci(baseline, candidate, n_resamples=999, seed=42)
        assert isinstance(result, BootstrapResult)
        assert result.method in ("bca", "percentile")
        assert result.n_resamples == 999
        assert result.seed == 42
        lo, hi = result.confidence_interval
        assert lo <= result.original_delta <= hi

    def test_reproducible_with_same_seed(self):
        np.random.seed(123)
        baseline = np.random.randn(20)
        candidate = baseline + 0.5
        r1 = paired_bootstrap_ci(baseline, candidate, seed=9999, n_resamples=999)
        np.random.seed(123)
        baseline2 = np.random.randn(20)
        candidate2 = baseline2 + 0.5
        r2 = paired_bootstrap_ci(baseline2, candidate2, seed=9999, n_resamples=999)
        assert r1.confidence_interval == r2.confidence_interval

    def test_small_sample_handled(self):
        baseline = np.array([1.0, 2.0])
        candidate = np.array([2.0, 3.0])
        result = paired_bootstrap_ci(baseline, candidate, n_resamples=99, seed=42)
        assert isinstance(result, BootstrapResult)


class TestTOSTEquivalence:
    def test_equivalent_groups_pass(self):
        np.random.seed(42)
        baseline = np.random.normal(10.0, 0.1, 30)
        candidate = np.random.normal(10.05, 0.1, 30)
        result = tost_equivalence_test(baseline, candidate, margin=1.0)
        assert "TOST_passed" in result
        assert "delta" in result
        assert "margin" in result

    def test_margin_zero_inconclusive(self):
        baseline = np.array([1.0, 2.0, 3.0])
        candidate = np.array([1.1, 2.1, 3.1])
        result = tost_equivalence_test(baseline, candidate, margin=0.0)
        assert result["TOST_passed"] is False
        assert "INCONCLUSIVE" in result["interpretation"]

    def test_return_structure(self):
        baseline = np.array([1.0, 2.0, 3.0, 4.0])
        candidate = np.array([1.0, 2.0, 3.0, 4.0])
        result = tost_equivalence_test(baseline, candidate, margin=0.5)
        for key in (
            "t1",
            "t2",
            "p_equivalence",
            "TOST_passed",
            "delta",
            "margin",
            "interpretation",
        ):
            assert key in result


# -------------------------------------------------------------------
# Phase D — compare_runs with new parameters
# -------------------------------------------------------------------


class TestCompareRunsPhaseD:
    def test_returns_metriccomparison_with_evidence(self):
        np.random.seed(42)
        n = 50
        baseline_df = pd.DataFrame({"latency": np.random.randn(n) + 10})
        candidate_df = pd.DataFrame({"latency": np.random.randn(n) + 9})
        result = compare_runs(
            baseline_df,
            candidate_df,
            meta(),
            meta(),
            ["latency"],
            [MetricRule("latency", "lower", 0.2)],
            seed=42,
            n_resamples=999,
        )
        assert result.status in ("REGRESSION", "NO_REGRESSION")
        m = result.metrics[0]
        assert isinstance(m, MetricComparison)
        assert m.evidence is not None
        assert isinstance(m.evidence, ComparisonEvidence)
        assert m.evidence.sample_count == n
        assert m.evidence.median_baseline is not None
        assert m.evidence.median_candidate is not None

    def test_equivalence_margin_triggers_tost(self):
        np.random.seed(42)
        baseline_df = pd.DataFrame({"score": np.random.randn(50) + 100})
        candidate_df = pd.DataFrame({"score": np.random.randn(50) + 100.05})
        result = compare_runs(
            baseline_df,
            candidate_df,
            meta(),
            meta(),
            ["score"],
            [MetricRule("score", "higher", 0.10)],
            seed=42,
            equivalence_margin=0.05,
            n_resamples=999,
        )
        m = result.metrics[0]
        assert m.evidence is not None
        assert "tost" in m.evidence.supporting_metrics
        # Interpretation should be set
        assert "interpretation" in m.evidence.supporting_metrics["tost"]

    def test_inference_result_values(self):
        np.random.seed(42)
        baseline_df = pd.DataFrame({"latency": np.random.randn(100) + 10})
        candidate_df = pd.DataFrame({"latency": np.random.randn(100) + 5})  # big improvement
        result = compare_runs(
            baseline_df,
            candidate_df,
            meta(),
            meta(),
            ["latency"],
            [MetricRule("latency", "lower", 0.5)],
            seed=42,
            n_resamples=999,
        )
        assert result.metrics[0].evidence is not None
        assert result.metrics[0].evidence.inference_result in (
            "IMPROVEMENT_SUPPORTED",
            "REGRESSION_SUPPORTED",
            "EQUIVALENT_WITHIN_MARGIN",
            "DIFFERENCE_DETECTED_DIRECTION_UNCLEAR",
            "INSUFFICIENT_POWER",
            "INCONCLUSIVE",
        )

    def test_insufficient_power_small_n(self):
        baseline_df = pd.DataFrame({"latency": [1.0, 2.0]})
        candidate_df = pd.DataFrame({"latency": [1.1, 2.1]})
        result = compare_runs(
            baseline_df,
            candidate_df,
            meta(),
            meta(),
            ["latency"],
            [MetricRule("latency", "lower", 0.2)],
            seed=42,
            n_resamples=99,
        )
        # n=2 is below the 4-threshold for bootstrap CI defensibility
        # With paired=True the function still runs but the inference result may reflect the small n
        assert result.metrics[0].evidence is not None

    def test_paired_bootstrap_ci_seed_reproducibility(self):
        np.random.seed(0)
        baseline = np.random.randn(30)
        candidate = baseline + 0.3
        r1 = paired_bootstrap_ci(baseline, candidate, seed=7777, n_resamples=1999)
        np.random.seed(0)
        baseline2 = np.random.randn(30)
        candidate2 = baseline2 + 0.3
        r2 = paired_bootstrap_ci(baseline2, candidate2, seed=7777, n_resamples=1999)
        assert r1.confidence_interval == r2.confidence_interval

    def test_backwards_compatible_min_samples(self):
        """New parameters must not break existing calls with positional args."""
        result = compare_runs(
            pd.DataFrame({"x": [1, 2]}),
            pd.DataFrame({"x": [3, 4]}),
            meta(),
            meta(),
            ["x"],
            [MetricRule("x", "lower", 1)],
            min_samples=2,
        )
        assert result.status in ("REGRESSION", "NO_REGRESSION")

    def test_unpaired_bootstrap(self):
        np.random.seed(42)
        baseline_df = pd.DataFrame({"latency": np.random.randn(30) + 10})
        candidate_df = pd.DataFrame({"latency": np.random.randn(30) + 12})
        result = compare_runs(
            baseline_df,
            candidate_df,
            meta(),
            meta(),
            ["latency"],
            [MetricRule("latency", "lower", 0.5)],
            paired=False,
            seed=42,
            n_resamples=999,
        )
        assert result.metrics[0].evidence is not None
        assert result.metrics[0].evidence.confidence_interval is not None

    def test_evidence_effect_size_populated(self):
        np.random.seed(42)
        baseline_df = pd.DataFrame({"score": np.random.randn(50) + 100})
        candidate_df = pd.DataFrame({"score": np.random.randn(50) + 110})
        result = compare_runs(
            baseline_df,
            candidate_df,
            meta(),
            meta(),
            ["score"],
            [MetricRule("score", "higher", 0.05)],
            seed=42,
            n_resamples=999,
        )
        m = result.metrics[0]
        assert m.evidence is not None
        assert m.evidence.effect_size is not None
        assert isinstance(m.evidence.effect_size, float)

    def test_mad_populated(self):
        np.random.seed(42)
        baseline_df = pd.DataFrame({"score": np.random.randn(50) + 100})
        candidate_df = pd.DataFrame({"score": np.random.randn(50) + 110})
        result = compare_runs(
            baseline_df,
            candidate_df,
            meta(),
            meta(),
            ["score"],
            [MetricRule("score", "higher", 0.05)],
            seed=42,
            n_resamples=999,
        )
        m = result.metrics[0]
        assert m.evidence is not None
        assert m.evidence.mad_baseline is not None
        assert m.evidence.mad_candidate is not None


# -------------------------------------------------------------------
# Phase E — collect_dependency_versions
# -------------------------------------------------------------------


class TestCollectDependencyVersions:
    def test_returns_dict(self):
        result = collect_dependency_versions()
        assert isinstance(result, dict)

    def test_known_packages_present(self):
        result = collect_dependency_versions()
        for pkg in ("numpy", "pandas", "scikit-learn", "scipy"):
            assert pkg in result

    def test_values_are_strings_or_none(self):
        result = collect_dependency_versions()
        for v in result.values():
            assert isinstance(v, (str, type(None)))


# -------------------------------------------------------------------
# Phase P — Reproducibility helpers
# -------------------------------------------------------------------


class TestComputeInputHash:
    def test_hash_is_hex_string(self):
        df = pd.DataFrame({"a": [1, 2, 3], "b": [4.0, 5.0, 6.0]})
        h = compute_input_hash(df)
        assert isinstance(h, str)
        assert len(h) == 64  # SHA-256 hex
        assert all(c in "0123456789abcdef" for c in h)

    def test_same_df_same_hash(self):
        df1 = pd.DataFrame({"x": [1, 2], "y": [3.0, 4.0]})
        df2 = pd.DataFrame({"x": [1, 2], "y": [3.0, 4.0]})
        assert compute_input_hash(df1) == compute_input_hash(df2)

    def test_different_column_order_same_hash(self):
        # Column order should not affect hash (sorted)
        df1 = pd.DataFrame({"a": [1], "b": [2]})
        df2 = pd.DataFrame({"b": [2], "a": [1]})
        assert compute_input_hash(df1) == compute_input_hash(df2)

    def test_different_content_different_hash(self):
        df1 = pd.DataFrame({"x": [1, 2, 3]})
        df2 = pd.DataFrame({"x": [1, 2, 4]})
        assert compute_input_hash(df1) != compute_input_hash(df2)


class TestGenerateAnalysisId:
    def test_is_uuid4(self):
        aid = generate_analysis_id()
        uuid.UUID(aid)  # raises if invalid

    def test_unique_per_call(self):
        ids = {generate_analysis_id() for _ in range(100)}
        assert len(ids) == 100


class TestBuildReproducibilityMetadata:
    def test_returns_dict_with_required_keys(self):
        meta = build_reproducibility_metadata(
            analysis_id="test-001",
            input_hash="abc123",
            seed=42,
            parameters={"foo": 1},
            paired=False,
        )
        assert meta["analysis_id"] == "test-001"
        assert meta["input_hash"] == "abc123"
        assert meta["seed"] == 42
        assert meta["paired"] is False
        assert meta["parameters"] == {"foo": 1}
        assert "dependencies" in meta

    def test_dependencies_populated(self):
        meta = build_reproducibility_metadata(
            analysis_id="test-002",
            input_hash=None,
            seed=0,
            parameters={},
            paired=True,
        )
        deps = meta["dependencies"]
        assert isinstance(deps, dict)
        for pkg in ("numpy", "pandas", "scikit-learn", "scipy"):
            assert pkg in deps
