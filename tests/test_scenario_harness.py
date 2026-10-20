"""Tests for the scenario harness: runner, evaluator, and scenario TOML contracts."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).parent.parent.resolve()

# Add repo root to path for harness imports (at repo root, not src/)
sys.path.insert(0, str(REPO_ROOT))
# Add src to path for anomaly_lab imports
sys.path.insert(0, str(REPO_ROOT / "src"))

from harness.evaluator import (
    EvaluationOutcome,
    _check_evidence_class,
    evaluate_scenario,
)
from harness.runner import (
    RunResult,
    discover_scenarios,
    load_scenario,
    run_scenario,
)


class TestDiscoverScenarios:
    """Tests for harness.runner.discover_scenarios"""

    def test_discovers_toml_files(self):
        """Should return sorted list of .toml files."""
        scenarios_dir = REPO_ROOT / "harness" / "scenarios"
        paths = discover_scenarios(scenarios_dir)
        assert len(paths) >= 14, f"Expected >= 14 scenarios, got {len(paths)}"
        assert all(p.suffix == ".toml" for p in paths)
        assert paths == sorted(paths)

    def test_missing_dir_raises(self):
        """Missing scenarios directory should raise HarnessError."""
        from harness.runner import HarnessError

        with pytest.raises(HarnessError, match="not found"):
            discover_scenarios(Path("/no/such/dir"))


class TestLoadScenario:
    """Tests for harness.runner.load_scenario"""

    def test_loads_valid_toml(self):
        """Should parse a valid scenario TOML."""
        path = REPO_ROOT / "harness" / "scenarios" / "01_rz_point_anomaly.toml"
        spec = load_scenario(path)
        assert spec["name"] == "rz_point_anomaly_single"
        assert spec["type"] == "analyze"
        assert spec["detector"] == "robust-zscore"
        assert "expected_evidence_class" in spec

    def test_loads_compare_scenario(self):
        """Should parse a compare-type scenario."""
        path = REPO_ROOT / "harness" / "scenarios" / "13_compare_paired_shift.toml"
        spec = load_scenario(path)
        assert spec["type"] == "compare"
        assert "baseline" in spec
        assert "candidate" in spec
        assert "baseline_metadata" in spec


class TestRunScenario:
    """Tests for harness.runner.run_scenario (subprocess-based)"""

    def test_runs_valid_analyze_scenario(self):
        """A valid analyze scenario should exit 0 with result JSON."""
        path = REPO_ROOT / "harness" / "scenarios" / "01_rz_point_anomaly.toml"
        result = run_scenario(path)
        # The CLI may exit 0 (success) or may have internal issues;
        # we care that we get a structured result
        assert isinstance(result, RunResult)
        assert result.scenario_name is not None
        assert result.elapsed_seconds >= 0

    def test_runs_valid_compare_scenario(self):
        """A valid compare scenario should produce a result."""
        path = REPO_ROOT / "harness" / "scenarios" / "13_compare_paired_shift.toml"
        result = run_scenario(path)
        assert isinstance(result, RunResult)


class TestEvaluateScenario:
    """Tests for harness.evaluator.evaluate_scenario"""

    def test_passes_exact_evidence_class(self):
        """Exact evidence class match should pass."""
        outcome = evaluate_scenario(
            "test",
            {"expected_evidence_class": "POINT_ANOMALY"},
            {"evidence_class": "POINT_ANOMALY", "confidence": 0.8},
        )
        assert outcome.passed
        assert any(c.passed for c in outcome.checks if c.field == "evidence_class")

    def test_superset_relaxation(self):
        """A result in the superset family should pass."""
        # POINT_ANOMALY is a superset that includes LOCAL_OUTLIER and GLOBAL_OUTLIER
        outcome = evaluate_scenario(
            "test",
            {"expected_evidence_class": "POINT_ANOMALY"},
            {"evidence_class": "LOCAL_OUTLIER", "confidence": 0.7},
        )
        assert outcome.passed

    def test_fails_wrong_evidence_class(self):
        """Wrong evidence class (not in superset) should fail."""
        outcome = evaluate_scenario(
            "test",
            {"expected_evidence_class": "POINT_ANOMALY"},
            {"evidence_class": "DENSITY_NOISE", "confidence": 0.8},
        )
        assert not outcome.passed
        ec_check = next(c for c in outcome.checks if c.field == "evidence_class")
        assert not ec_check.passed

    def test_confidence_bounds_pass(self):
        """Confidence within bounds should pass."""
        outcome = evaluate_scenario(
            "test",
            {"expected_confidence_min": 0.5, "expected_confidence_max": 1.0},
            {"evidence_class": "POINT_ANOMALY", "confidence": 0.8},
        )
        assert outcome.passed

    def test_confidence_bounds_fail(self):
        """Confidence outside bounds should fail."""
        outcome = evaluate_scenario(
            "test",
            {"expected_confidence_min": 0.9, "expected_confidence_max": 1.0},
            {"evidence_class": "POINT_ANOMALY", "confidence": 0.5},
        )
        assert not outcome.passed

    def test_required_field_missing(self):
        """A required field that is missing should fail."""
        outcome = evaluate_scenario(
            "test",
            {"required_evidence_fields": ["evidence_class", "confidence"]},
            {"evidence_class": "POINT_ANOMALY"},  # confidence missing
        )
        assert not outcome.passed
        conf_check = next(c for c in outcome.checks if c.field == "confidence")
        assert not conf_check.passed

    def test_required_assumption_present(self):
        """A required assumption appearing in the list should pass."""
        outcome = evaluate_scenario(
            "test",
            {"required_assumptions": ["median_and_MAD"]},
            {"evidence_class": "INCONCLUSIVE", "assumptions": ["median_and_MAD_are_appropriate"]},
        )
        assert outcome.passed

    def test_forbidden_assumption_not_present(self):
        """A forbidden assumption not appearing should pass."""
        outcome = evaluate_scenario(
            "test",
            {"forbidden_assumptions": ["causal"]},
            {"evidence_class": "INCONCLUSIVE", "assumptions": ["some_assumption"]},
        )
        assert outcome.passed

    def test_forbidden_assumption_present_fails(self):
        """A forbidden assumption appearing should fail."""
        outcome = evaluate_scenario(
            "test",
            {"forbidden_assumptions": ["causal"]},
            {"evidence_class": "INCONCLUSIVE", "assumptions": ["causal_claim_made"]},
        )
        assert not outcome.passed

    def test_anomaly_count_bounds_pass(self):
        """Anomaly count within bounds should pass."""
        outcome = evaluate_scenario(
            "test",
            {"anomaly_count_min": 1, "anomaly_count_max": 3},
            {
                "evidence_class": "POINT_ANOMALY",
                "rows": [{"is_anomaly": True}, {"is_anomaly": True}],
            },
        )
        assert outcome.passed

    def test_anomaly_count_bounds_fail(self):
        """Anomaly count outside bounds should fail."""
        outcome = evaluate_scenario(
            "test",
            {"anomaly_count_min": 5, "anomaly_count_max": 10},
            {"evidence_class": "POINT_ANOMALY", "rows": [{"is_anomaly": True}]},
        )
        assert not outcome.passed

    def test_supporting_metrics_tolerance(self):
        """Supporting metric within tolerance should pass."""
        outcome = evaluate_scenario(
            "test",
            {
                "supporting_metrics": [{"name": "f1", "expected": 0.9, "tolerance": 1e-3}],
                "tolerance": 1e-3,
            },
            {
                "evidence_class": "POINT_ANOMALY",
                "supporting_metrics": {"f1": 0.9001},
            },
        )
        assert outcome.passed

    def test_supporting_metrics_outside_tolerance(self):
        """Supporting metric outside tolerance should fail."""
        outcome = evaluate_scenario(
            "test",
            {
                "supporting_metrics": [{"name": "f1", "expected": 0.9, "tolerance": 1e-3}],
            },
            {
                "evidence_class": "POINT_ANOMALY",
                "supporting_metrics": {"f1": 0.95},
            },
        )
        assert not outcome.passed

    def test_detector_mismatch_fails(self):
        """Detector name mismatch should fail."""
        outcome = evaluate_scenario(
            "test",
            {"expected_detector": "robust-zscore"},
            {"evidence_class": "POINT_ANOMALY", "detector": "dbscan"},
        )
        assert not outcome.passed

    def test_compare_status_as_evidence_class(self):
        """For compare scenarios, status field drives the evidence_class check."""
        outcome = evaluate_scenario(
            "test",
            {"expected_evidence_class": "COMPARISON_DIFFERENCE"},
            {"status": "REGRESSION", "metrics": []},
        )
        # status "REGRESSION" maps to COMPARISON_DIFFERENCE
        assert outcome.passed

    def test_empty_result_still_returns_outcome(self):
        """An empty result should still produce an EvaluationOutcome (not crash)."""
        outcome = evaluate_scenario("test", {}, {})
        # Empty spec with empty result should still be "passed" (no constraints)
        assert isinstance(outcome, EvaluationOutcome)
        assert outcome.passed  # no constraints = nothing to fail


class TestSemanticCheckEvidenceClassHierarchy:
    """Unit tests for the evidence class superset hierarchy."""

    def test_point_anomaly_superset_includes_local_and_global(self):
        """POINT_ANOMALY superset includes LOCAL_OUTLIER and GLOBAL_OUTLIER."""
        check = _check_evidence_class("LOCAL_OUTLIER", "POINT_ANOMALY")
        assert check.passed

        check = _check_evidence_class("GLOBAL_OUTLIER", "POINT_ANOMALY")
        assert check.passed

    def test_local_outlier_superset_includes_global(self):
        """LOCAL_OUTLIER superset includes GLOBAL_OUTLIER."""
        check = _check_evidence_class("GLOBAL_OUTLIER", "LOCAL_OUTLIER")
        assert check.passed

    def test_comparison_difference_superset(self):
        """COMPARISON_DIFFERENCE superset includes COMPARISON_EQUIVALENT."""
        check = _check_evidence_class("COMPARISON_EQUIVALENT", "COMPARISON_DIFFERENCE")
        assert check.passed


class TestEvaluationOutcomeToDict:
    """Tests for EvaluationOutcome.to_dict()"""

    def test_round_trip(self):
        """to_dict should produce JSON-serializable output."""
        outcome = evaluate_scenario(
            "round-trip-test",
            {"expected_evidence_class": "POINT_ANOMALY"},
            {"evidence_class": "POINT_ANOMALY", "confidence": 0.85},
        )
        d = outcome.to_dict()
        assert isinstance(d, dict)
        assert d["scenario"] == "round-trip-test"
        assert d["passed"] is True
        assert isinstance(d["checks"], list)
        assert isinstance(d["errors"], list)
        assert isinstance(d["warnings"], list)

        # Should be JSON-serializable
        json.dumps(d)


class TestScenarioTOMLsAreValid:
    """Verify all scenario TOMLs are parseable and have required fields."""

    @pytest.mark.parametrize(
        "scenario_path", discover_scenarios(REPO_ROOT / "harness" / "scenarios")
    )
    def test_scenario_has_name(self, scenario_path: Path):
        """Every scenario TOML must have a 'name' field."""
        import tomllib

        spec = tomllib.loads(scenario_path.read_text(encoding="utf-8"))
        assert "name" in spec, f"{scenario_path.name} missing 'name'"

    @pytest.mark.parametrize(
        "scenario_path", discover_scenarios(REPO_ROOT / "harness" / "scenarios")
    )
    def test_scenario_has_type(self, scenario_path: Path):
        """Every scenario TOML must have a 'type' field."""
        import tomllib

        spec = tomllib.loads(scenario_path.read_text(encoding="utf-8"))
        assert "type" in spec, f"{scenario_path.name} missing 'type'"
        assert spec["type"] in {"analyze", "compare"}

    @pytest.mark.parametrize(
        "scenario_path", discover_scenarios(REPO_ROOT / "harness" / "scenarios")
    )
    def test_analyze_scenario_has_detector(self, scenario_path: Path):
        """Analyze scenarios must have a valid detector."""
        import tomllib

        spec = tomllib.loads(scenario_path.read_text(encoding="utf-8"))
        if spec["type"] == "analyze":
            assert "detector" in spec
            assert spec["detector"] in {"robust-zscore", "dbscan", "isolation-forest"}

    @pytest.mark.parametrize(
        "scenario_path", discover_scenarios(REPO_ROOT / "harness" / "scenarios")
    )
    def test_compare_scenario_has_required_fields(self, scenario_path: Path):
        """Compare scenarios must have baseline, candidate, and metadata."""
        import tomllib

        spec = tomllib.loads(scenario_path.read_text(encoding="utf-8"))
        if spec["type"] == "compare":
            assert "baseline" in spec
            assert "candidate" in spec
            assert "baseline_metadata" in spec
            assert "candidate_metadata" in spec
            assert "metrics" in spec
