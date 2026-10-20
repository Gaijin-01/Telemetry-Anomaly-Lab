"""Tests for anomaly_lab.experiments.planner module."""

from __future__ import annotations

import pandas as pd
import pytest

from anomaly_lab.experiments.planner import (
    ApplicabilityCode,
    ComponentKind,
    ExperimentPlan,
    plan,
)


class TestPlan:
    """Tests for the plan() function."""

    def test_plan_returns_experiment_plan(self):
        """plan() should return an ExperimentPlan."""
        frame = pd.DataFrame({"x": [1.0, 2.0, 3.0, 10.0, 11.0]})
        result = plan(frame, ["x"])
        assert isinstance(result, ExperimentPlan)

    def test_plan_with_univariate_numeric(self):
        """plan() with univariate numeric data should return plan with detector + consensus."""
        frame = pd.DataFrame({"x": [1.0, 2.0, 3.0, 10.0, 11.0]})
        result = plan(frame, ["x"])

        assert result.n_samples == 5
        assert len(result.steps) >= 1

        # Should have at least one detector step
        detector_steps = [s for s in result.steps if s.kind == ComponentKind.DETECTOR]
        assert len(detector_steps) >= 1

        # All steps should have a valid status
        for step in result.steps:
            assert step.name
            assert step.kind in ComponentKind

    def test_plan_with_multivariate_numeric(self):
        """plan() with multivariate numeric data should include applicable detectors."""
        frame = pd.DataFrame(
            {
                "x": [1.0, 2.0, 3.0, 10.0, 11.0],
                "y": [0.5, 1.5, 2.5, 9.5, 10.5],
            }
        )
        result = plan(frame, ["x", "y"])

        assert len(result.steps) >= 1
        detector_steps = [s for s in result.steps if s.kind == ComponentKind.DETECTOR]
        assert len(detector_steps) >= 1

    def test_plan_preserves_features(self):
        """plan() should preserve the features list in the plan."""
        frame = pd.DataFrame({"x": [1.0, 2.0, 3.0, 10.0, 11.0], "y": [0.5, 1.0, 1.5, 5.0, 5.5]})
        result = plan(frame, ["x", "y"])

        # Features in plan should match requested features
        assert "x" in result.features
        assert "y" in result.features

    def test_plan_with_insufficient_samples(self):
        """plan() with very few samples should still return a plan."""
        frame = pd.DataFrame({"x": [1.0, 2.0]})
        result = plan(frame, ["x"])

        assert isinstance(result, ExperimentPlan)

    def test_plan_applicability_records(self):
        """Each planned step should have corresponding applicability records."""
        frame = pd.DataFrame({"x": [1.0, 2.0, 3.0, 10.0, 11.0]})
        result = plan(frame, ["x"])

        assert len(result.applicability_records) >= 1
        for record in result.applicability_records:
            assert record.name
            assert record.kind in ComponentKind
            assert record.code in ApplicabilityCode


class TestExperimentPlan:
    """Tests for ExperimentPlan dataclass."""

    def test_experiment_plan_is_frozen(self):
        """ExperimentPlan should be a frozen dataclass."""
        frame = pd.DataFrame({"x": [1.0, 2.0]})
        result = plan(frame, ["x"])

        with pytest.raises(AttributeError):  # frozen dataclass cannot be modified
            result.n_samples = 999

    def test_experiment_plan_to_dict(self):
        """ExperimentPlan should have a to_dict method."""
        frame = pd.DataFrame({"x": [1.0, 2.0, 3.0, 10.0, 11.0]})
        result = plan(frame, ["x"])

        d = result.to_dict()
        assert "profile" in d
        assert "applicable_detectors" in d
        assert "consensus_recommended" in d
        assert "steps" in d
        assert "summary" in d

    def test_experiment_plan_str_representation(self):
        """ExperimentPlan should have a string representation."""
        frame = pd.DataFrame({"x": [1.0, 2.0, 3.0, 10.0, 11.0]})
        result = plan(frame, ["x"])

        s = str(result)
        assert "ExperimentPlan" in s


class TestPlannedStep:
    """Tests for PlannedStep dataclass."""

    def test_planned_step_has_required_fields(self):
        """PlannedStep should have name, kind, rationale, and parameters."""
        frame = pd.DataFrame({"x": [1.0, 2.0, 3.0, 10.0, 11.0]})
        result = plan(frame, ["x"])

        for step in result.steps:
            assert step.name
            assert step.kind in ComponentKind
            assert step.rationale
            assert step.parameters is not None


class TestComponentKind:
    """Tests for ComponentKind enum."""

    def test_component_kind_values(self):
        """ComponentKind should have expected values."""
        assert ComponentKind.DETECTOR in ComponentKind
        assert ComponentKind.CHANGE_POINTS in ComponentKind
        assert ComponentKind.REGIMES in ComponentKind
        assert ComponentKind.CONSENSUS in ComponentKind

    def test_component_kind_is_enum(self):
        """ComponentKind should be an enum."""
        import enum

        assert issubclass(ComponentKind, enum.Enum)


class TestApplicabilityCode:
    """Tests for ApplicabilityCode enum."""

    def test_applicability_code_values(self):
        """ApplicabilityCode should have expected values."""
        assert ApplicabilityCode.APPLICABLE in ApplicabilityCode
        assert ApplicabilityCode.NOT_APPLICABLE in ApplicabilityCode
        assert ApplicabilityCode.INSUFFICIENT_DATA in ApplicabilityCode
        assert ApplicabilityCode.INVALID_INPUT in ApplicabilityCode

    def test_applicability_code_is_enum(self):
        """ApplicabilityCode should be an enum."""
        import enum

        assert issubclass(ApplicabilityCode, enum.Enum)
