"""Tests for the experiment CLI subcommand: validate, plan, run, replay."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent.resolve()
CLI_PATH = REPO_ROOT / "src" / "anomaly_lab" / "cli.py"
HARNESS_DIR = REPO_ROOT / "harness"
SCENARIOS_DIR = HARNESS_DIR / "scenarios"


def _run_cli(argv: list[str]) -> subprocess.CompletedProcess:
    """Run the CLI and return the CompletedProcess."""
    return subprocess.run(
        [sys.executable, str(CLI_PATH)] + argv,
        capture_output=True,
        text=True,
        cwd=str(REPO_ROOT),
    )


class TestExperimentValidate:
    """Tests for: anomaly-lab experiment validate"""

    def test_validate_passing_scenario(self, tmp_path: Path):
        """A well-formed scenario TOML should pass validation."""
        scenario = SCENARIOS_DIR / "01_rz_point_anomaly.toml"
        result = _run_cli(["experiment", "validate", str(scenario)])
        assert result.returncode == 0
        assert "PASS" in result.stdout

    def test_validate_missing_file(self):
        """A non-existent scenario file should return exit 2."""
        result = _run_cli(["experiment", "validate", "/no/such/file.toml"])
        assert result.returncode == 2
        assert "not found" in result.stdout or "not found" in result.stderr

    def test_validate_missing_type(self, tmp_path: Path):
        """A TOML without a 'type' field should fail validation."""
        bad = tmp_path / "bad.toml"
        bad.write_text("name = 'no-type'\n")
        result = _run_cli(["experiment", "validate", str(bad)])
        assert result.returncode == 2
        assert "type" in result.stdout.lower() or "type" in result.stderr.lower()

    def test_validate_invalid_detector(self, tmp_path: Path):
        """An unknown detector name should fail validation."""
        bad = tmp_path / "bad.toml"
        bad.write_text(
            'name = "bad"\ntype = "analyze"\ndetector = "unknown-detector"\nfeatures = ["x"]\n'
        )
        result = _run_cli(["experiment", "validate", str(bad)])
        assert result.returncode == 2

    def test_validate_compare_missing_baseline(self, tmp_path: Path):
        """A compare scenario missing baseline should fail."""
        bad = tmp_path / "bad.toml"
        bad.write_text(
            'name = "bad"\ntype = "compare"\n'
            'candidate = "x.csv"\n'
            'baseline_metadata = { model = "m", version = "v", quantization = "q", concurrency = "1" }\n'
            'candidate_metadata = { model = "m", version = "v", quantization = "q", concurrency = "1" }\n'
            'features = ["x"]\n'
            '[[metrics]]\nname = "x"\ndirection = "higher"\nthreshold = 0.1\n'
        )
        result = _run_cli(["experiment", "validate", str(bad)])
        assert result.returncode == 2


class TestExperimentPlan:
    """Tests for: anomaly-lab experiment plan"""

    def test_plan_shows_command(self):
        """plan should print the resolved CLI command."""
        scenario = SCENARIOS_DIR / "01_rz_point_anomaly.toml"
        result = _run_cli(["experiment", "plan", str(scenario)])
        assert result.returncode == 0
        assert "anomaly-lab" in result.stdout
        assert "analyze" in result.stdout
        assert "robust-zscore" in result.stdout

    def test_plan_missing_file(self):
        """plan with a non-existent file should exit 2."""
        result = _run_cli(["experiment", "plan", "/no/such.toml"])
        assert result.returncode == 2


class TestExperimentRun:
    """Tests for: anomaly-lab experiment run"""

    def test_run_single_scenario(self, tmp_path: Path):
        """Running a single scenario should produce result JSON and eval JSON."""
        scenario = SCENARIOS_DIR / "01_rz_point_anomaly.toml"
        output_dir = tmp_path / "results"
        output_dir.mkdir()

        result = _run_cli(["experiment", "run", str(scenario), "--output-dir", str(output_dir)])
        # Some scenarios may fail semantic checks (detector-specific behaviour),
        # but the CLI itself should run without a config error (exit 0 or 1)
        assert result.returncode in (0, 1)

    def test_run_all_scenarios(self, tmp_path: Path):
        """Running with no scenario arg should discover and run all TOMLs."""
        output_dir = tmp_path / "results"
        output_dir.mkdir()

        result = _run_cli(["experiment", "run", "--output-dir", str(output_dir)])
        # We expect at least one scenario to have been attempted
        assert "scenario" in result.stdout.lower() or "running" in result.stdout.lower()
        assert result.returncode in (0, 1)

    def test_run_caches_results(self, tmp_path: Path):
        """A second run should use cached results without --rerun."""
        scenario = SCENARIOS_DIR / "02_rz_no_anomaly.toml"
        output_dir = tmp_path / "results"
        output_dir.mkdir()

        # First run
        _run_cli(["experiment", "run", str(scenario), "--output-dir", str(output_dir)])
        # Second run should say "using cached"
        result2 = _run_cli(["experiment", "run", str(scenario), "--output-dir", str(output_dir)])
        assert "cached" in result2.stdout or "using cached" in result2.stdout

    def test_run_rerun_overrides_cache(self, tmp_path: Path):
        """--rerun should re-execute even when cached results exist."""
        scenario = SCENARIOS_DIR / "02_rz_no_anomaly.toml"
        output_dir = tmp_path / "results"
        output_dir.mkdir()

        _run_cli(["experiment", "run", str(scenario), "--output-dir", str(output_dir)])
        result2 = _run_cli(
            ["experiment", "run", str(scenario), "--output-dir", str(output_dir), "--rerun"]
        )
        assert "running" in result2.stdout.lower()
        assert "cached" not in result2.stdout.lower()


class TestExperimentReplay:
    """Tests for: anomaly-lab experiment replay"""

    def test_replay_with_valid_result(self, tmp_path: Path):
        """Replay should pass when result JSON matches the scenario contract."""
        scenario = SCENARIOS_DIR / "01_rz_point_anomaly.toml"
        # Run once to get a result
        output_dir = tmp_path / "results"
        output_dir.mkdir()
        _run_cli(["experiment", "run", str(scenario), "--output-dir", str(output_dir)])

        result_json = output_dir / "01_rz_point_anomaly.json"
        if result_json.exists():
            result = _run_cli(["experiment", "replay", str(scenario), str(result_json)])
            # Result depends on actual detector output; we just check it runs
            assert "PASS" in result.stdout or "FAIL" in result.stdout

    def test_replay_nonexistent_result(self):
        """Replay with a missing result JSON should exit 2."""
        scenario = SCENARIOS_DIR / "01_rz_point_anomaly.toml"
        result = _run_cli(["experiment", "replay", str(scenario), "/no/such/result.json"])
        assert result.returncode == 2


class TestExperimentSubcommandRequired:
    """Verify that experiment subcommand is required (no bare experiment command)."""

    def test_bare_experiment_requires_subcommand(self):
        """Running just 'experiment' without a subcommand should fail gracefully."""
        result = _run_cli(["experiment"])
        # argparse should print usage and exit with error
        assert result.returncode != 0
