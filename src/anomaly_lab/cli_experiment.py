"""CLI experiment subcommand implementation.

Implements the 'experiment' subcommand family: validate, plan, run, replay.
Uses the harness/ package for execution and evaluation.

Exit codes
----------
0  – success (validate passed / plan printed / run completed without errors / replay matched)
1  – harness gate condition triggered (run only)
2  – configuration / validation error
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# The harness package lives at the repo root, one level above src/
# Ensure it is on the import path when running as a subprocess.
_REPO_ROOT = Path(__file__).parent.parent.parent.resolve()
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import tempfile
from typing import Any

from anomaly_lab.schema import AnomalyLabError

_HARNESS_ROOT = _REPO_ROOT / "harness"


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────


def _resolve_harness_dir(harness_dir: Path | None) -> Path:
    """Resolve the harness directory, defaulting to <repo_root>/harness."""
    if harness_dir is not None:
        if not harness_dir.is_dir():
            raise AnomalyLabError(f"harness directory not found: {harness_dir}")
        return harness_dir.resolve()
    return _HARNESS_ROOT.resolve()


def _load_scenario_toml(scenario_path: Path) -> dict[str, Any]:
    """Parse and return a scenario TOML dict."""
    import tomllib

    if not scenario_path.exists():
        raise AnomalyLabError(f"scenario file not found: {scenario_path}")
    try:
        return tomllib.loads(scenario_path.read_text(encoding="utf-8"))
    except tomllib.TOMLDecodeError as exc:
        raise AnomalyLabError(f"TOML syntax error in {scenario_path}: {exc}") from exc


def _check_fixture_exists(scenario_path: Path, spec: dict[str, Any]) -> list[str]:
    """Check that referenced fixtures exist. Returns list of warnings (not errors)."""
    warnings: list[str] = []
    harness_root = scenario_path.parent.parent  # scenarios/.. = harness/

    fixture_name = spec.get("fixture")
    if fixture_name:
        fixture_path = harness_root / "fixtures" / fixture_name
        if not fixture_path.exists():
            raise AnomalyLabError(
                f"fixture {fixture_name!r} referenced in {scenario_path.name} "
                f"not found at {fixture_path}"
            )

    for key in ("baseline", "candidate"):
        path_val = spec.get(key)
        if path_val:
            p = Path(path_val)
            if not p.is_absolute():
                p = (_REPO_ROOT / path_val).resolve()
            if not p.exists():
                warnings.append(
                    f"{key} path {path_val!r} does not yet exist on disk (will fail at run time)"
                )
    return warnings


# ─────────────────────────────────────────────────────────────────────────────
# experiment validate
# ─────────────────────────────────────────────────────────────────────────────


def _run_experiment_validate(scenario_path: Path, harness_dir: Path) -> bool:
    """Validate a scenario TOML: syntax, required fields, fixture references.

    Returns True if validation passes, False otherwise.
    Prints findings to stdout.
    """
    print(f"Validating scenario: {scenario_path}")

    try:
        spec = _load_scenario_toml(scenario_path)
    except AnomalyLabError as exc:
        print(f"FAIL: {exc}")
        return False

    # Required top-level fields
    required_fields = ["name", "type"]
    missing = [f for f in required_fields if f not in spec]
    if missing:
        print(f"FAIL: missing required field(s): {', '.join(missing)}")
        return False

    scenario_type = spec.get("type")
    valid_types = {"analyze", "compare"}
    if scenario_type not in valid_types:
        print(f"FAIL: type {scenario_type!r} not in supported types: {valid_types}")
        return False

    # Type-specific required fields
    if scenario_type == "analyze":
        for f in ("detector", "features"):
            if f not in spec:
                print(f"FAIL: missing required field for type=analyze: {f}")
                return False
        valid_detectors = {"robust-zscore", "dbscan", "isolation-forest"}
        if spec["detector"] not in valid_detectors:
            print(
                f"FAIL: detector {spec['detector']!r} not in supported detectors: {valid_detectors}"
            )
            return False

    elif scenario_type == "compare":
        for f in ("baseline", "candidate", "baseline_metadata", "candidate_metadata", "metrics"):
            if f not in spec:
                print(f"FAIL: missing required field for type=compare: {f}")
                return False

    # Check fixtures
    fixture_warnings = _check_fixture_exists(scenario_path, spec)
    if fixture_warnings:
        for w in fixture_warnings:
            print(f"WARNING: {w}")

    # Validate expected_evidence_class if present
    ec = spec.get("expected_evidence_class")
    if ec:
        from anomaly_lab.evidence import EvidenceClass

        valid_ecs = {e.value for e in EvidenceClass}
        if ec not in valid_ecs:
            print(
                f"WARNING: expected_evidence_class {ec!r} not in known EvidenceClass "
                f"values; scenario may be using a custom label"
            )

    print("PASS: scenario TOML is structurally valid")
    return True


# ─────────────────────────────────────────────────────────────────────────────
# experiment plan
# ─────────────────────────────────────────────────────────────────────────────


def _run_experiment_plan(scenario_path: Path, harness_dir: Path) -> None:
    """Print the resolved CLI commands for a scenario without executing them."""
    from harness.runner import _build_command, load_scenario

    if not scenario_path.exists():
        raise AnomalyLabError(f"scenario file not found: {scenario_path}")

    spec = load_scenario(scenario_path)
    scenario_name = spec.get("name", scenario_path.stem)

    with tempfile.TemporaryDirectory(prefix="harness_plan_") as tmp_dir_str:
        cmd = _build_command(spec, Path(tmp_dir_str))

    print(f"Scenario : {scenario_name}")
    print(f"Type     : {spec.get('type')}")
    print(f"Command  : anomaly-lab {cmd.command} {' '.join(cmd.args)}")
    print()
    print("Resolved CLI invocation:")
    print(f"  anomaly-lab {cmd.command} \\")
    indent = "  "
    for _i, arg in enumerate(cmd.args):
        if arg.startswith("--"):
            print(f"{indent}{arg}")
        else:
            print(f"{indent}{arg} \\")
        indent = "    "
    print()
    print("Note: plan does not execute the command. Use 'experiment run' to execute.")


# ─────────────────────────────────────────────────────────────────────────────
# experiment run
# ─────────────────────────────────────────────────────────────────────────────


def _run_experiment_run(
    scenario_path: Path | None,
    harness_dir: Path,
    output_dir: Path | None,
    rerun: bool,
) -> int:
    """Execute one scenario or all scenarios in the harness directory.

    Returns 0 on success, 1 if any scenario failed (exit-code gate), 2 on config error.
    Writes JSON results to output_dir (default: <harness_dir>/results/).
    """
    from harness.evaluator import evaluate_scenario
    from harness.runner import (
        discover_scenarios,
        load_scenario,
        run_scenario,
    )

    scenarios_dir = harness_dir / "scenarios"
    if not scenarios_dir.is_dir():
        raise AnomalyLabError(f"scenarios directory not found: {scenarios_dir}")

    # Determine output directory
    if output_dir is None:
        output_dir = harness_dir / "results"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Collect TOML paths to run
    if scenario_path is None:
        toml_paths = discover_scenarios(scenarios_dir)
        print(f"Running all {len(toml_paths)} scenarios in {scenarios_dir}")
    else:
        if not scenario_path.exists():
            raise AnomalyLabError(f"scenario file not found: {scenario_path}")
        toml_paths = [scenario_path]
        print(f"Running scenario: {scenario_path}")

    total = len(toml_paths)
    succeeded = 0
    failed = 0

    for i, toml_path in enumerate(toml_paths, 1):
        scenario_name = toml_path.stem
        result_path = output_dir / f"{scenario_name}.json"

        # Check for cached result
        if result_path.exists() and not rerun:
            print(f"  [{i}/{total}] {scenario_name}: using cached result (use --rerun to override)")
            try:
                result_payload = json.loads(result_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                print(f"  [{i}/{total}] {scenario_name}: cached result is invalid JSON, re-running")
                result_payload = None
        else:
            result_payload = None

        if result_payload is None:
            print(f"  [{i}/{total}] {scenario_name}: running...")
            run_result = run_scenario(toml_path)
            if not run_result.success:
                print(f"  [{i}/{total}] {scenario_name}: FAILED (exit {run_result.returncode})")
                if run_result.stderr:
                    for ln in run_result.stderr.splitlines()[:3]:
                        print(f"         {ln}")
                failed += 1
                continue

            result_payload = run_result.result_json
            if result_payload is None:
                print(f"  [{i}/{total}] {scenario_name}: no result JSON produced")
                failed += 1
                continue
            # Write result JSON
            result_path.write_text(
                json.dumps(result_payload, indent=2, allow_nan=False) + "\n", encoding="utf-8"
            )
            print(f"  [{i}/{total}] {scenario_name}: OK ({run_result.elapsed_seconds:.2f}s)")

        # result_payload is guaranteed non-None from here (see continue above)
        assert result_payload is not None
        # Evaluate semantic correctness
        try:
            scenario_spec = load_scenario(toml_path)
        except Exception as exc:
            print(f"  [{i}/{total}] {scenario_name}: could not load spec: {exc}")
            failed += 1
            continue

        outcome = evaluate_scenario(scenario_name, scenario_spec, result_payload)
        outcome_path = output_dir / f"{scenario_name}_eval.json"
        outcome_path.write_text(
            json.dumps(outcome.to_dict(), indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )

        if outcome.passed:
            print(f"  [{i}/{total}] {scenario_name}: SEMANTIC CHECK PASSED")
            succeeded += 1
        else:
            print(f"  [{i}/{total}] {scenario_name}: SEMANTIC CHECK FAILED")
            for err in outcome.errors:
                print(f"         ERROR: {err}")
            failed += 1

    print()
    print(f"Results: {succeeded}/{total} passed, {failed}/{total} failed")
    print(f"JSON results written to: {output_dir}")

    if failed > 0:
        return 1  # gate condition
    return 0


# ─────────────────────────────────────────────────────────────────────────────
# experiment replay
# ─────────────────────────────────────────────────────────────────────────────


def _run_experiment_replay(
    scenario_path: Path,
    result_json_path: Path,
    harness_dir: Path,
) -> bool:
    """Re-evaluate a saved JSON result against a scenario spec.

    Returns True if the semantic checks pass, False otherwise.
    """
    from harness.evaluator import evaluate_scenario

    if not result_json_path.exists():
        print(f"error: result JSON not found: {result_json_path}", file=__import__("sys").stderr)
        return False

    try:
        result_payload = json.loads(result_json_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        print(f"error: result JSON is not valid JSON: {exc}", file=__import__("sys").stderr)
        return False

    scenario_spec = _load_scenario_toml(scenario_path)
    scenario_name = scenario_spec.get("name", scenario_path.stem)

    print(f"Replaying scenario : {scenario_name}")
    print(f"Result JSON        : {result_json_path}")

    outcome = evaluate_scenario(scenario_name, scenario_spec, result_payload)

    print()
    if outcome.passed:
        print("SEMANTIC CHECK PASSED")
    else:
        print("SEMANTIC CHECK FAILED")
        for err in outcome.errors:
            print(f"  ERROR: {err}")

    for warn in outcome.warnings:
        print(f"  WARNING: {warn}")

    print()
    print(f"{len(outcome.checks)} checks performed:")
    for check in outcome.checks:
        status = "PASS" if check.passed else "FAIL"
        print(
            f"  [{status}] {check.field}: expected={check.expected!r}, actual={check.actual!r} {check.note}"
        )

    return outcome.passed
