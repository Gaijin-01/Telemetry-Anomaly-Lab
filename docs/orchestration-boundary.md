# Orchestration Boundary

This document describes the division of responsibility between the **scenario harness** and the **anomaly-lab CLI** — what each layer owns, what it delegates, and what it must never depend on.

---

## Layer Overview

```
┌─────────────────────────────────────────────────────────────┐
│  User / CI pipeline                                         │
│  $ anomaly-lab experiment run --harness-dir ./harness       │
└──────────────────────────┬──────────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────────┐
│  CLI (cli.py)                                               │
│  - Parses top-level arguments                               │
│  - Delegates to cli_experiment.py for experiment commands   │
│  - Handles exit codes and JSON output for non-experiment   │
│    commands                                                 │
└──────────────────────────┬──────────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────────┐
│  cli_experiment.py                                          │
│  - experiment validate  → TOML syntax + fixture checks       │
│  - experiment plan      → Print resolved CLI command         │
│  - experiment run       → Discover + run + evaluate          │
│  - experiment replay    → Re-evaluate saved JSON             │
│  Owns: result JSON paths, output_dir, error messages        │
└──────────────────────────┬──────────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────────┐
│  harness/runner.py                                         │
│  - Discovers scenario TOMLs                                 │
│  - Loads and resolves fixtures                              │
│  - Builds CommandSpec objects                               │
│  - Invokes CLI as subprocess                                │
│  - Returns RunResult (returncode, stdout, stderr, JSON)    │
└──────────────────────────┬──────────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────────┐
│  harness/evaluator.py                                       │
│  - Semantic correctness checks (NOT exit codes)             │
│  - Evidence class hierarchy                                 │
│  - Numeric tolerance checking                               │
│  - Returns EvaluationOutcome                                │
└──────────────────────────┬──────────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────────┐
│  anomaly_lab core (detectors, compare, evidence, etc.)       │
│  - Pure computation; no CLI awareness                       │
│  - Produces DetectionResult / ComparisonResult              │
│  - Contains no references to harness/ or cli_experiment/  │
└─────────────────────────────────────────────────────────────┘
```

---

## Strict Invariants

1. **Core has no harness dependency.** The `src/anomaly_lab/` package does not import `harness`, `cli_experiment`, or any scenario-specific code. The core modules (`detectors/`, `compare.py`, `evidence.py`, etc.) are pure computation libraries.

2. **Harness has no test doubles.** The harness (`harness/runner.py`) invokes the CLI as a subprocess using `sys.executable` and the real entry point. It does not import and call `main()` directly. This ensures that the integration test is a genuine end-to-end test.

3. **CLI is the only entry point that knows about the experiment subcommand.** `cli.py` imports `cli_experiment.py` lazily (inside the experiment command handler) to avoid hard-loading the harness module when not needed.

4. **Evaluator checks semantics, not mechanics.** `harness/evaluator.py` never checks `returncode`. It only examines the JSON payload for semantic correctness. Mechanical success/failure (exit codes) is handled by `cli_experiment.py`'s `run_all()` loop.

5. **Scenario TOMLs are pure data.** They contain no Python code. All logic lives in `runner.py`, `evaluator.py`, and `cli_experiment.py`.

---

## What Each Layer Owns

### CLI (`cli.py`)

- Parsing `argparse` arguments
- Dispatching to the correct command handler
- Exit codes: `0` success, `1` gate condition, `2` config error
- `--output-json` for non-experiment commands

### `cli_experiment.py`

- Top-level experiment commands: `validate`, `plan`, `run`, `replay`
- Harness directory resolution
- Scenario discovery (calls `harness.runner`)
- Result caching (skip if `harness/results/<name>.json` exists)
- Calling `harness.evaluator.evaluate_scenario()`

### `harness/runner.py`

- Loading scenario TOMLs (`load_scenario`)
- Building `CommandSpec` from scenario spec
- Subprocess invocation of the CLI
- Returning `RunResult` dataclass
- `discover_scenarios()`, `run_all()` public API

### `harness/evaluator.py`

- Pure function `evaluate_scenario(scenario_name, scenario_spec, result_payload)`
- Returns `EvaluationOutcome` with per-check details
- No subprocess calls, no file I/O
- No knowledge of the CLI or experiment commands

### `anomaly_lab` core

- Pure computation: detection, comparison, evidence classification
- Returns structured result dataclasses
- No knowledge of the harness or CLI

---

## Data Flow for `experiment run`

```
User runs: anomaly-lab experiment run

cli.py:
  - Parses args
  - Calls cli_experiment._run_experiment_run()

cli_experiment._run_experiment_run():
  - Resolves harness_dir
  - Calls harness.runner.discover_scenarios() → list of .toml paths
  - For each scenario:
      - Calls harness.runner.run_scenario(toml_path)
        → builds CommandSpec
        → subprocess.run([sys.executable, cli.py, ...])
        → returns RunResult
      - Writes RunResult.result_json to harness/results/<name>.json
      - Calls harness.evaluator.evaluate_scenario(spec, result_json)
        → returns EvaluationOutcome
      - Writes EvaluationOutcome to harness/results/<name>_eval.json
      - Prints pass/fail summary

  - Returns 0 (all passed) or 1 (any failed)
```

---

## Replay Path

```
User runs: anomaly-lab experiment replay scenario.toml result.json

cli_experiment._run_experiment_replay():
  - Loads result_json
  - Loads scenario.toml
  - Calls harness.evaluator.evaluate_scenario(spec, result)
  - Prints per-check outcomes
  - Returns True/False
```

Replay is useful for:
- Investigating a CI failure without re-running the experiment
- Comparing a locally-cached result against an updated scenario spec
- Auditing historical results
