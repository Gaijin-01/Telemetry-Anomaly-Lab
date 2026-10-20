"""Scenario harness runner: discovers, loads, runs, and reports on scenario TOMLs.

The runner is a pure Python API (no CLI dependency) that:
1. Discovers scenario TOMLs under a scenarios directory.
2. Resolves fixture references and inline fixture data.
3. Invokes the appropriate anomaly-lab command (analyze, compare, etc.)
4. Returns structured RunResult objects for evaluation by harness/evaluator.py.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

# The runner can be invoked as a library or as a CLI entry point.
REPO_ROOT = Path(__file__).parent.parent.resolve()


# ─────────────────────────────────────────────────────────────────────────────
# Exceptions
# ─────────────────────────────────────────────────────────────────────────────


class HarnessError(Exception):
    """Raised when the harness cannot execute a scenario."""


# ─────────────────────────────────────────────────────────────────────────────
# Result types
# ─────────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True, kw_only=True)
class CommandSpec:
    """Describes one CLI command to run within a scenario."""

    command: str  # e.g. "analyze", "compare", "baseline"
    args: list[str]
    env: dict[str, str] = field(default_factory=dict)

    def argv(self) -> list[str]:
        return [self.command] + self.args


@dataclass(frozen=True, kw_only=True)
class RunResult:
    """Result of a single scenario execution attempt."""

    scenario_name: str
    scenario_path: Path
    command: CommandSpec
    returncode: int
    stdout: str
    stderr: str
    result_json: dict[str, Any] | None
    elapsed_seconds: float
    seed: int | None = None

    @property
    def success(self) -> bool:
        """True only when the CLI exited 0 and produced parseable JSON."""
        return self.returncode == 0 and self.result_json is not None


# ─────────────────────────────────────────────────────────────────────────────
# Scenario loading
# ─────────────────────────────────────────────────────────────────────────────


def discover_scenarios(scenarios_dir: Path) -> list[Path]:
    """Return sorted list of .toml files under scenarios_dir."""
    if not scenarios_dir.is_dir():
        raise HarnessError(f"scenarios directory not found: {scenarios_dir}")
    return sorted(scenarios_dir.glob("*.toml"))


def load_scenario(toml_path: Path) -> dict[str, Any]:
    """Parse a scenario TOML, inlining any referenced fixtures."""
    import tomllib

    raw = tomllib.loads(toml_path.read_text(encoding="utf-8"))
    spec = dict(raw)

    # Resolve fixture references
    if "fixture" in spec:
        fixture_name = spec["fixture"]
        fixture_path = toml_path.parent.parent / "fixtures" / fixture_name
        if not fixture_path.exists():
            raise HarnessError(
                f"scenario {toml_path.name} references unknown fixture: "
                f"{fixture_name} (not found at {fixture_path})"
            )
        df = pd.read_csv(fixture_path)
        spec["_inline_fixture"] = df

    return spec


# ─────────────────────────────────────────────────────────────────────────────
# Fixture helpers (inline fixtures written to temp CSV for CLI consumption)
# ─────────────────────────────────────────────────────────────────────────────


def _write_fixture_csv(df: pd.DataFrame, suffix: str = "") -> Path:
    """Write a DataFrame to a temporary CSV, return the path."""
    fd, path = tempfile.mkstemp(suffix=f"{suffix}.csv", prefix="harness_fixture_")
    df.to_csv(path, index=False)
    return Path(path)


# ─────────────────────────────────────────────────────────────────────────────
# Command builders
# ─────────────────────────────────────────────────────────────────────────────


def _build_analyze_command(spec: dict[str, Any], tmp_dir: Path) -> CommandSpec:
    """Build a CommandSpec for an 'analyze'-type scenario."""
    detector = spec.get("detector", "robust-zscore")
    features = spec.get("features", [])

    # Write inline fixture if present
    if "_inline_fixture" in spec:
        df: pd.DataFrame = spec["_inline_fixture"]
        csv_path = _write_fixture_csv(df, suffix=spec.get("name", "inline"))
        input_arg = str(csv_path)
    else:
        input_path_str = spec.get("input")
        if not input_path_str:
            raise HarnessError("scenario has no 'input' or 'fixture' field")
        input_arg = str(REPO_ROOT / input_path_str)

    args = [
        str(input_arg),
        "--detector",
        detector,
        "--features",
        ",".join(features) if isinstance(features, list) else features,
    ]

    output = tmp_dir / f"result_{spec.get('name', 'analyze')}.json"
    args += ["--output-json", str(output)]

    seed = spec.get("seed")
    if seed is not None:
        args += ["--seed", str(seed)]

    return CommandSpec(command="analyze", args=args)


def _build_compare_command(spec: dict[str, Any], tmp_dir: Path) -> CommandSpec:
    """Build a CommandSpec for a 'compare'-type scenario."""
    baseline_path = str(REPO_ROOT / spec["baseline"])
    candidate_path = str(REPO_ROOT / spec["candidate"])
    features = spec.get("features", [])
    baseline_meta = spec.get("baseline_metadata", {})
    candidate_meta = spec.get("candidate_metadata", {})
    metrics = spec.get("metrics", [])

    args = [baseline_path, candidate_path, "--features", ",".join(features)]

    for k, v in baseline_meta.items():
        args += ["--baseline-metadata", f"{k}={v}"]
    for k, v in candidate_meta.items():
        args += ["--candidate-metadata", f"{k}={v}"]
    for m in metrics:
        direction = m.get("direction", "higher")
        args += ["--metric", f"{m['name']}:{direction}:{m['threshold']}"]

    output = tmp_dir / f"result_{spec.get('name', 'compare')}.json"
    args += ["--output-json", str(output)]

    return CommandSpec(command="compare", args=args)


def _build_command(spec: dict[str, Any], tmp_dir: Path) -> CommandSpec:
    """Route to the appropriate command builder based on scenario type."""
    cmd_type = spec.get("type", "analyze")
    if cmd_type == "analyze":
        return _build_analyze_command(spec, tmp_dir)
    if cmd_type == "compare":
        return _build_compare_command(spec, tmp_dir)
    raise HarnessError(f"unsupported scenario type: {cmd_type!r}")


# ─────────────────────────────────────────────────────────────────────────────
# Execution
# ─────────────────────────────────────────────────────────────────────────────


def run_scenario(
    scenario_path: Path,
    cli_module: str | None = None,
) -> RunResult:
    """Execute a single scenario TOML and return the structured result.

    Parameters
    ----------
    scenario_path : Path
        Path to a scenario .toml file.
    cli_module : str | None
        Dotted import path to the CLI main function's module.
        Defaults to "anomaly_lab.cli".  Useful for testing with a test double.

    Returns
    -------
    RunResult
    """
    import time

    spec = load_scenario(scenario_path)
    scenario_name = spec.get("name", scenario_path.stem)

    with tempfile.TemporaryDirectory(prefix="harness_run_") as tmp_dir_str:
        tmp_dir = Path(tmp_dir_str)
        cmd = _build_command(spec, tmp_dir)

        # Locate the CLI entry point
        if cli_module is None:
            cli_path = REPO_ROOT / "src" / "anomaly_lab" / "cli.py"
            cli_cmd = [sys.executable, str(cli_path), *cmd.argv()]
        else:
            cli_cmd = [sys.executable, "-m", cli_module, *cmd.argv()]

        t0 = time.perf_counter()
        proc = subprocess.run(
            cli_cmd,
            capture_output=True,
            text=True,
            cwd=str(REPO_ROOT),
            env={**os.environ.copy()},
        )
        elapsed = time.perf_counter() - t0

        # Parse result JSON if written
        result_json: dict[str, Any] | None = None
        output_arg = next(
            (cmd.args[i + 1] for i, a in enumerate(cmd.args) if a == "--output-json"),
            None,
        )
        if output_arg and Path(output_arg).exists():
            try:
                result_json = json.loads(Path(output_arg).read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                pass  # leave result_json as None — success=False

        return RunResult(
            scenario_name=scenario_name,
            scenario_path=scenario_path,
            command=cmd,
            returncode=proc.returncode,
            stdout=proc.stdout,
            stderr=proc.stderr,
            result_json=result_json,
            elapsed_seconds=elapsed,
            seed=spec.get("seed"),
        )


# ─────────────────────────────────────────────────────────────────────────────
# Batch runner
# ─────────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True, kw_only=True)
class BatchResult:
    """Aggregated results from running all scenarios."""

    results: list[RunResult]
    total: int
    succeeded: int
    failed: int

    def summary(self) -> str:
        lines = [
            f"Harness batch: {self.succeeded}/{self.total} scenarios succeeded",
        ]
        for r in self.results:
            status = "OK" if r.success else f"FAIL({r.returncode})"
            lines.append(f"  [{status}] {r.scenario_name} ({r.elapsed_seconds:.2f}s)")
            if not r.success and r.stderr:
                for ln in r.stderr.splitlines()[:3]:
                    lines.append(f"       {ln}")
        return "\n".join(lines)


def run_all(
    scenarios_dir: Path | str | None = None,
    cli_module: str | None = None,
    verbose: bool = False,
) -> BatchResult:
    """Discover and run all scenarios in the scenarios directory.

    Parameters
    ----------
    scenarios_dir : Path | str | None
        Directory containing scenario TOMLs.
        Defaults to <repo_root>/harness/scenarios.
    cli_module : str | None
        Override for the CLI module path.
    verbose : bool
        Print progress to stdout.

    Returns
    -------
    BatchResult
    """
    if scenarios_dir is None:
        scenarios_dir = REPO_ROOT / "harness" / "scenarios"
    scenarios_dir = Path(scenarios_dir)

    toml_paths = discover_scenarios(scenarios_dir)
    results: list[RunResult] = []

    for p in toml_paths:
        if verbose:
            print(f"Running scenario: {p.name}")
        results.append(run_scenario(p, cli_module=cli_module))

    succeeded = sum(1 for r in results if r.success)
    return BatchResult(
        results=results,
        total=len(results),
        succeeded=succeeded,
        failed=len(results) - succeeded,
    )
