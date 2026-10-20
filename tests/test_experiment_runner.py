"""Tests for anomaly_lab.experiments.runner module."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pandas as pd
import pytest

from anomaly_lab.experiments.planner import plan
from anomaly_lab.experiments.runner import (
    PlanExecutionResult,
    StepStatus,
    run,
)


class TestRun:
    """Tests for the run() function."""

    def test_run_returns_plan_execution_result(self):
        """run() should return a PlanExecutionResult."""
        frame = pd.DataFrame({"x": [1.0, 2.0, 3.0, 10.0, 11.0]})
        p = plan(frame, ["x"])

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "result.toml"
            result = run(p, frame, output_path)

        assert isinstance(result, PlanExecutionResult)

    def test_run_writes_toml_file(self):
        """run() should write a TOML file to the output path."""
        frame = pd.DataFrame({"x": [1.0, 2.0, 3.0, 10.0, 11.0]})
        p = plan(frame, ["x"])

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "result.toml"
            run(p, frame, output_path)

            assert output_path.exists()
            content = output_path.read_text()
            assert len(content) > 0

    def test_run_produces_valid_toml(self):
        """run() should produce valid TOML output."""
        frame = pd.DataFrame({"x": [1.0, 2.0, 3.0, 10.0, 11.0]})
        p = plan(frame, ["x"])

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "result.toml"
            run(p, frame, output_path)

            content = output_path.read_text()
            try:
                import tomllib

                parsed = tomllib.loads(content)
            except ImportError:
                try:
                    import tomli as tomllib

                    parsed = tomllib.loads(content)
                except ImportError:
                    pytest.skip("No TOML parser available")

            assert isinstance(parsed, dict)
            assert "schema_version" in parsed

    def test_run_all_succeeded_flag(self):
        """run() should set all_succeeded True when all steps succeed."""
        frame = pd.DataFrame({"x": [1.0, 2.0, 3.0, 10.0, 11.0]})
        p = plan(frame, ["x"])

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "result.toml"
            result = run(p, frame, output_path)

            assert result.all_succeeded is True

    def test_run_step_results_count(self):
        """run() should have step_results for each planned step."""
        frame = pd.DataFrame({"x": [1.0, 2.0, 3.0, 10.0, 11.0]})
        p = plan(frame, ["x"])

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "result.toml"
            result = run(p, frame, output_path)

            assert len(result.step_results) == result.total_steps

    def test_run_with_multivariate_data(self):
        """run() should handle multivariate data."""
        frame = pd.DataFrame(
            {
                "x": [1.0, 2.0, 3.0, 10.0, 11.0],
                "y": [0.5, 1.5, 2.5, 9.5, 10.5],
            }
        )
        p = plan(frame, ["x", "y"])

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "result.toml"
            result = run(p, frame, output_path)

            assert isinstance(result, PlanExecutionResult)
            assert result.all_succeeded is True


class TestPlanExecutionResult:
    """Tests for PlanExecutionResult dataclass."""

    def test_result_has_required_fields(self):
        """PlanExecutionResult should have all required fields."""
        frame = pd.DataFrame({"x": [1.0, 2.0, 3.0, 10.0, 11.0]})
        p = plan(frame, ["x"])

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "result.toml"
            result = run(p, frame, output_path)

            assert result.schema_version
            assert result.total_steps >= 0
            assert result.successful_steps >= 0
            assert isinstance(result.all_succeeded, bool)
            assert isinstance(result.step_results, tuple)

    def test_result_to_dict(self):
        """PlanExecutionResult should have a to_dict method."""
        frame = pd.DataFrame({"x": [1.0, 2.0, 3.0, 10.0, 11.0]})
        p = plan(frame, ["x"])

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "result.toml"
            result = run(p, frame, output_path)

            d = result.to_dict()
            assert isinstance(d, dict)

    def test_result_to_toml(self):
        """PlanExecutionResult should have a to_toml method."""
        frame = pd.DataFrame({"x": [1.0, 2.0, 3.0, 10.0, 11.0]})
        p = plan(frame, ["x"])

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "result.toml"
            result = run(p, frame, output_path)

            toml_str = result.to_toml()
            assert isinstance(toml_str, str)
            assert len(toml_str) > 0


class TestStepResult:
    """Tests for StepResult dataclass."""

    def test_step_result_has_required_fields(self):
        """StepResult should have kind, name, status, and payload."""
        frame = pd.DataFrame({"x": [1.0, 2.0, 3.0, 10.0, 11.0]})
        p = plan(frame, ["x"])

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "result.toml"
            result = run(p, frame, output_path)

            for step_result in result.step_results:
                assert step_result.kind.value in (
                    "detector",
                    "change_points",
                    "regimes",
                    "consensus",
                )
                assert step_result.name
                assert step_result.status in (
                    StepStatus.SUCCESS,
                    StepStatus.SKIPPED,
                    StepStatus.FAILED,
                )

    def test_step_status_values(self):
        """StepStatus should have expected values."""
        assert StepStatus.SUCCESS in StepStatus
        assert StepStatus.SKIPPED in StepStatus
        assert StepStatus.FAILED in StepStatus


class TestTomlOutput:
    """Tests for TOML output format."""

    def test_toml_has_expected_top_level_keys(self):
        """TOML output should have expected top-level keys."""
        frame = pd.DataFrame({"x": [1.0, 2.0, 3.0, 10.0, 11.0]})
        p = plan(frame, ["x"])

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "result.toml"
            run(p, frame, output_path)

            content = output_path.read_text()
            try:
                import tomllib

                parsed = tomllib.loads(content)
            except ImportError:
                try:
                    import tomli as tomllib

                    parsed = tomllib.loads(content)
                except ImportError:
                    pytest.skip("No TOML parser available")

            expected_keys = {
                "schema_version",
                "total_steps",
                "successful_steps",
                "all_succeeded",
                "step_results",
            }
            assert expected_keys.issubset(set(parsed.keys()))

    def test_toml_step_results_structure(self):
        """TOML step_results should be a list with correct structure."""
        frame = pd.DataFrame({"x": [1.0, 2.0, 3.0, 10.0, 11.0]})
        p = plan(frame, ["x"])

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "result.toml"
            run(p, frame, output_path)

            content = output_path.read_text()
            try:
                import tomllib

                parsed = tomllib.loads(content)
            except ImportError:
                try:
                    import tomli as tomllib

                    parsed = tomllib.loads(content)
                except ImportError:
                    pytest.skip("No TOML parser available")

            step_results = parsed["step_results"]
            assert isinstance(step_results, list)
            assert len(step_results) > 0

            # Each step should have required fields
            for step in step_results:
                assert "kind" in step
                assert "name" in step
                assert "status" in step
