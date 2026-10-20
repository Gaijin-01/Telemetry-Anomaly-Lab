from __future__ import annotations

import argparse
import json
import sys
import tomllib
from pathlib import Path

import pandas as pd

from anomaly_lab.detectors import dbscan as dbscan_detector
from anomaly_lab.detectors import isolation_forest as isolation_detector
from anomaly_lab.detectors import robust_zscore as robust_detector
from anomaly_lab.ingest import read_csv
from anomaly_lab.results import DetectionResult
from anomaly_lab.schema import AnomalyLabError


def analyze(input_path: str, detector: str, features: list[str]) -> DetectionResult:
    frame = read_csv(input_path)
    if detector == "isolation-forest":
        return isolation_detector.fit_predict(frame, features)
    if detector == "dbscan":
        return dbscan_detector.fit_predict(frame, features)
    if detector == "robust-zscore":
        return robust_detector.fit_predict(frame, features)
    if detector in ("ecod", "lof", "knn"):
        try:
            from anomaly_lab.detectors import pyod_adapters

            if detector == "ecod":
                return pyod_adapters.fit_predict_ecod(frame, features)
            if detector == "lof":
                return pyod_adapters.fit_predict_lof(frame, features)
            if detector == "knn":
                return pyod_adapters.fit_predict_knn(frame, features)
        except ImportError as exc:
            raise AnomalyLabError(
                f"detector {detector!r} requires the optional 'pyod' dependency. "
                "Install it with: pip install 'anomaly-lab[pyod]'"
            ) from exc
    raise AnomalyLabError(f"unknown detector: {detector}")


# Exit codes
_EXIT_GATE = 1  # CI gate condition triggered
_EXIT_CFG = 2  # invalid input/configuration


def _features(value: str) -> list[str]:
    return [feature.strip() for feature in value.split(",") if feature.strip()]


def _write_json(path: Path | None, payload: dict) -> bool:
    if path is None:
        return True
    try:
        path.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    except (OSError, ValueError) as exc:
        print(f"error: cannot write JSON output: {exc}", file=sys.stderr)
        return False
    return True


# ─────────────────────────────────────────────────────────────────────────────
# TOML configuration layer
# ─────────────────────────────────────────────────────────────────────────────

_VALID_COMMANDS = frozenset(["compare"])
_REQUIRED_TOP_LEVEL = frozenset(
    [
        "command",
        "baseline",
        "candidate",
        "features",
        "baseline_metadata",
        "candidate_metadata",
        "metrics",
    ]
)
_METADATA_REQUIRED = frozenset(["model", "version", "quantization", "concurrency"])
_METADATA_OPTIONAL_KEYS = frozenset(["allow_difference"])
_METRIC_REQUIRED = frozenset(["name", "direction", "threshold"])
_METRIC_OPTIONAL = frozenset(["min_samples"])
_GATE_VALUES = frozenset(["regression", "not-comparable", "insufficient-sample"])


def _resolve_path(toml_path: Path, value: str) -> Path:
    """Resolve a possibly-relative path against the TOML file's directory."""
    p = Path(value)
    if p.is_absolute():
        return p
    return (toml_path.parent / p).resolve()


def _load_config(toml_path: Path) -> dict:
    """Parse and validate a TOML config file, returning the raw dict."""
    try:
        raw = tomllib.loads(toml_path.read_text(encoding="utf-8"))
    except tomllib.TOMLDecodeError as exc:
        raise AnomalyLabError(f"TOML syntax error in {toml_path}: {exc}") from exc

    unknown = set(raw.keys()) - _REQUIRED_TOP_LEVEL - {"gate"}
    if unknown:
        raise AnomalyLabError(
            f"TOML config contains unknown field(s): {', '.join(sorted(unknown))}. "
            f"Allowed: {', '.join(sorted(_REQUIRED_TOP_LEVEL))}. "
            f"Known optional: gate"
        )

    for field in _REQUIRED_TOP_LEVEL:
        if field not in raw:
            raise AnomalyLabError(f"TOML config is missing required field: {field}")

    if raw["command"] not in _VALID_COMMANDS:
        raise AnomalyLabError(
            f"TOML command {raw['command']!r} is not supported. "
            f"Supported: {', '.join(sorted(_VALID_COMMANDS))}"
        )

    # Validate metadata blocks
    for side in ("baseline_metadata", "candidate_metadata"):
        block = raw.get(side)
        if not isinstance(block, dict):
            raise AnomalyLabError(f"TOML {side} must be a table (dict); got {type(block).__name__}")
        missing = _METADATA_REQUIRED - set(block.keys())
        if missing:
            raise AnomalyLabError(
                f"TOML {side} is missing required key(s): {', '.join(sorted(missing))}"
            )
        unknown_meta = set(block.keys()) - _METADATA_REQUIRED - _METADATA_OPTIONAL_KEYS
        if unknown_meta:
            raise AnomalyLabError(
                f"TOML {side} contains unknown key(s): {', '.join(sorted(unknown_meta))}. "
                f"Allowed: {', '.join(sorted(_METADATA_REQUIRED | _METADATA_OPTIONAL_KEYS))}"
            )

    # Validate features
    feats = raw.get("features")
    if not isinstance(feats, list) or not all(isinstance(f, str) and f.strip() for f in feats):
        raise AnomalyLabError("TOML features must be a non-empty list of strings")
    if not feats:
        raise AnomalyLabError("TOML features list is empty")

    # Validate metrics
    metrics = raw.get("metrics")
    if not isinstance(metrics, list) or not metrics:
        raise AnomalyLabError("TOML metrics must be a non-empty list")
    for m in metrics:
        if not isinstance(m, dict):
            raise AnomalyLabError(f"TOML metric entry must be a table; got {type(m).__name__}")
        missing_m = _METRIC_REQUIRED - set(m.keys())
        if missing_m:
            raise AnomalyLabError(
                f"TOML metric is missing required key(s): {', '.join(sorted(missing_m))}"
            )
        unknown_m = set(m.keys()) - _METRIC_REQUIRED - _METRIC_OPTIONAL
        if unknown_m:
            raise AnomalyLabError(
                f"TOML metric contains unknown key(s): {', '.join(sorted(unknown_m))}. "
                f"Allowed: {', '.join(sorted(_METRIC_REQUIRED | _METRIC_OPTIONAL))}"
            )

    # Validate paths (exist but don't load yet)
    for side in ("baseline", "candidate"):
        p = _resolve_path(toml_path, str(raw.get(side, "")))
        if not p.exists():
            raise AnomalyLabError(f"TOML {side} path does not exist: {p}")

    # Validate gate if present
    if "gate" in raw:
        gate = raw["gate"]
        if not isinstance(gate, dict):
            raise AnomalyLabError("TOML gate must be a table")
        if "fail_on" not in gate:
            raise AnomalyLabError("TOML gate is missing required key: fail_on")
        fail_on = gate["fail_on"]
        if isinstance(fail_on, str):
            fail_on = [fail_on]
        if not isinstance(fail_on, list) or not all(isinstance(v, str) for v in fail_on):
            raise AnomalyLabError("TOML gate.fail_on must be a string or list of strings")
        unknown_gate = set(fail_on) - _GATE_VALUES
        if unknown_gate:
            raise AnomalyLabError(
                f"TOML gate.fail_on contains unknown value(s): {', '.join(sorted(unknown_gate))}. "
                f"Allowed: {', '.join(sorted(_GATE_VALUES))}"
            )
        raw["gate"]["fail_on"] = list(fail_on)

    return raw


def _run_compare_from_config(toml_path: Path) -> tuple[dict, list[str] | None]:
    """
    Execute a compare command from a validated TOML config.
    Returns (payload_dict, fail_on_list).
    fail_on_list is None when no gate is configured.
    """
    from anomaly_lab.compare import MetricRule, RunMetadata, compare_runs

    cfg = _load_config(toml_path)

    baseline_path = _resolve_path(toml_path, str(cfg["baseline"]))
    candidate_path = _resolve_path(toml_path, str(cfg["candidate"]))

    baseline_meta_dict = dict(cfg["baseline_metadata"])
    baseline_allow = tuple(baseline_meta_dict.pop("allow_difference", ()))
    baseline_meta = RunMetadata(**baseline_meta_dict, allow_difference=baseline_allow)

    candidate_meta_dict = dict(cfg["candidate_metadata"])
    candidate_allow = tuple(candidate_meta_dict.pop("allow_difference", ()))
    candidate_meta = RunMetadata(**candidate_meta_dict, allow_difference=candidate_allow)

    features = cfg["features"]

    rules = []
    for m in cfg["metrics"]:
        rules.append(MetricRule(m["name"], m["direction"], float(m["threshold"])))

    min_samples = 2
    for m in cfg["metrics"]:
        if "min_samples" in m:
            min_samples = int(m["min_samples"])

    result = compare_runs(
        read_csv(baseline_path),
        read_csv(candidate_path),
        baseline_meta,
        candidate_meta,
        features,
        rules,
        min_samples,
    )
    return result.to_dict(), cfg.get("gate", {}).get("fail_on")


# ─────────────────────────────────────────────────────────────────────────────
# Argument parser
# ─────────────────────────────────────────────────────────────────────────────


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="anomaly-lab")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # run  (TOML config)
    run_parser = subparsers.add_parser("run", help="run a command from a TOML config file")
    run_parser.add_argument("config", type=Path, help="path to TOML configuration file")
    run_parser.add_argument("--output-json", type=Path)

    # analyze
    analyze_parser = subparsers.add_parser("analyze", help="analyze a telemetry CSV")
    analyze_parser.add_argument("input", type=Path)
    analyze_parser.add_argument(
        "--detector", choices=("isolation-forest", "dbscan", "robust-zscore"), required=True
    )
    analyze_parser.add_argument("--features", required=True, help="comma-separated feature names")
    analyze_parser.add_argument("--output-json", type=Path)

    # compare
    compare_parser = subparsers.add_parser(
        "compare", help="compare baseline and candidate telemetry"
    )
    compare_parser.add_argument("baseline", type=Path)
    compare_parser.add_argument("candidate", type=Path)
    compare_parser.add_argument("--features", required=True, help="comma-separated metric names")
    compare_parser.add_argument(
        "--baseline-metadata", action="append", required=True, metavar="KEY=VALUE"
    )
    compare_parser.add_argument(
        "--candidate-metadata", action="append", required=True, metavar="KEY=VALUE"
    )
    compare_parser.add_argument(
        "--metric", action="append", required=True, metavar="NAME:higher|lower:THRESHOLD"
    )
    compare_parser.add_argument(
        "--fail-on",
        action="append",
        dest="fail_on",
        metavar="regression|not-comparable|insufficient-sample",
        help="exit with code 1 when the named condition occurs",
    )
    compare_parser.add_argument("--min-samples", type=int, default=2)
    compare_parser.add_argument("--output-json", type=Path)

    # baseline
    baseline_parser = subparsers.add_parser(
        "baseline", help="compute a strictly past-only baseline"
    )
    baseline_parser.add_argument("input", type=Path)
    baseline_parser.add_argument(
        "--features", required=True, help="comma-separated numeric features"
    )
    baseline_parser.add_argument("--method", choices=("ewma", "rolling"), required=True)
    baseline_parser.add_argument("--alpha", type=float)
    baseline_parser.add_argument("--window", type=int)
    baseline_parser.add_argument("--order-by")
    baseline_parser.add_argument("--output-json", type=Path)

    # change-points
    points_parser = subparsers.add_parser(
        "change-points", help="scan for retrospective median shifts"
    )
    points_parser.add_argument("input", type=Path)
    points_parser.add_argument("--feature", required=True)
    points_parser.add_argument("--min-segment", type=int, required=True)
    points_parser.add_argument("--threshold", type=float, required=True)
    points_parser.add_argument("--order-by")
    points_parser.add_argument("--output-json", type=Path)

    # regimes
    regimes_parser = subparsers.add_parser("regimes", help="cluster telemetry with DBSCAN")
    regimes_parser.add_argument("input", type=Path)
    regimes_parser.add_argument(
        "--features", required=True, help="comma-separated numeric features"
    )
    regimes_parser.add_argument("--eps", type=float, default=0.5)
    regimes_parser.add_argument("--min-samples", type=int, default=5)
    regimes_parser.add_argument("--metadata", action="append", default=[], metavar="COLUMN")
    regimes_parser.add_argument("--output-json", type=Path)

    # report
    report_parser = subparsers.add_parser(
        "report", help="produce an offline HTML report from a JSON result file"
    )
    report_parser.add_argument(
        "result_json",
        type=Path,
        help="path to a result JSON file from analyze, compare, baseline, change-points or regimes",
    )
    report_parser.add_argument(
        "--output",
        type=Path,
        default=Path("report.html"),
        help="output HTML file (default: report.html)",
    )

    # experiment
    experiment_parser = subparsers.add_parser(
        "experiment", help="run and evaluate experiment scenarios from TOML specs"
    )
    exp_subparsers = experiment_parser.add_subparsers(dest="experiment_command", required=True)

    # experiment validate
    exp_validate = exp_subparsers.add_parser(
        "validate", help="validate a scenario TOML and its fixture references"
    )
    exp_validate.add_argument("scenario", type=Path, help="path to scenario TOML file")
    exp_validate.add_argument("--harness-dir", type=Path, default=None)

    # experiment plan
    exp_plan = exp_subparsers.add_parser(
        "plan", help="show the resolved CLI commands for a scenario without running"
    )
    exp_plan.add_argument("scenario", type=Path, help="path to scenario TOML file")
    exp_plan.add_argument("--harness-dir", type=Path, default=None)

    # experiment run
    exp_run = exp_subparsers.add_parser(
        "run", help="execute one or all scenario TOMLs and write JSON results"
    )
    exp_run.add_argument(
        "scenario", nargs="?", type=Path, default=None, help="scenario TOML (omit for all)"
    )
    exp_run.add_argument("--harness-dir", type=Path, default=None)
    exp_run.add_argument("--output-dir", type=Path, default=None)
    exp_run.add_argument(
        "--rerun", action="store_true", help="re-run even if result JSON already exists"
    )

    # experiment replay
    exp_replay = exp_subparsers.add_parser(
        "replay", help="re-evaluate a saved JSON result against a scenario spec"
    )
    exp_replay.add_argument("scenario", type=Path, help="path to scenario TOML file")
    exp_replay.add_argument("result_json", type=Path, help="path to previously saved JSON result")
    exp_replay.add_argument("--harness-dir", type=Path, default=None)

    # consensus
    consensus_parser = subparsers.add_parser(
        "consensus", help="compute detector consensus across multiple JSON results"
    )
    consensus_parser.add_argument(
        "input",
        type=Path,
        help="path to CSV file that was analyzed (for row alignment)",
    )
    consensus_parser.add_argument(
        "--results",
        action="append",
        required=True,
        metavar="RESULT_JSON",
        help="path to a detector result JSON (may be specified multiple times)",
    )
    consensus_parser.add_argument("--output-json", type=Path)

    # inspect
    inspect_parser = subparsers.add_parser(
        "inspect", help="classify dataset semantics and show applicable detectors"
    )
    inspect_parser.add_argument("data", type=Path, help="path to telemetry CSV")
    inspect_parser.add_argument(
        "--features", help="comma-separated feature columns (auto-detected if omitted)"
    )
    inspect_parser.add_argument("--output-json", type=Path)

    return parser


def _metadata(items: list[str]):
    from anomaly_lab.compare import RunMetadata

    parsed = {}
    allow_diff = []
    for item in items:
        if "=" not in item:
            raise AnomalyLabError("metadata values must use KEY=VALUE")
        key, value = item.split("=", 1)
        if key == "allow_difference":
            allow_diff = [v.strip() for v in value.split(",") if v.strip()]
            continue
        if not key or key in parsed:
            raise AnomalyLabError("metadata keys must be non-empty and unique")
        parsed[key] = value
    required = {"model", "version", "quantization", "concurrency"}
    if set(parsed) != required:
        raise AnomalyLabError("metadata requires model, version, quantization, concurrency")
    return RunMetadata(**parsed, allow_difference=tuple(allow_diff))


def _fail_on_codes(fail_on: list[str] | None, result: object) -> int | None:
    """Return exit code 1 if a configured gate condition is met, else None."""
    if not fail_on:
        return None
    status = getattr(result, "status", None)
    if status is None:
        return None
    mapping = {
        "regression": "REGRESSION",
        "not-comparable": "NOT_COMPARABLE",
        "insufficient-sample": "INSUFFICIENT_SAMPLE",
    }
    for gate in fail_on:
        expected = mapping.get(gate, "")
        if status == expected:
            return _EXIT_GATE
    return None


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    payload: dict = {}  # may be set by specific command branches; used by final _write_json
    try:
        if args.command == "run":
            config_path = args.config
            cfg = _load_config(config_path)
            print(f"Config: {config_path.resolve()}")
            print(f"Command: {cfg['command']}")
            if cfg["command"] == "compare":
                from anomaly_lab.compare import MetricRule, compare_runs

                payload, fail_on = _run_compare_from_config(config_path)
                print(f"Status: {payload['status']}; Causality: {payload['causality']}")
                for m in payload.get("metrics", []):
                    print(
                        f"{m['name']}: {m['status']} "
                        f"(baseline={m['baseline']}, "
                        f"candidate={m['candidate']}, delta={m['delta']})"
                    )
                if payload.get("mismatches"):
                    print("Metadata mismatches: " + ", ".join(payload["mismatches"]))
                if payload.get("warnings"):
                    for w in payload["warnings"]:
                        print(f"Warning: {w}")
                gate_exit = _fail_on_codes(fail_on, type("R", (), payload)())
                if gate_exit is not None:
                    _write_json(args.output_json, payload)
                    return gate_exit
            else:
                raise AnomalyLabError(
                    f"run: command {cfg['command']!r} not yet implemented in config mode"
                )

        elif args.command == "compare":
            from anomaly_lab.compare import MetricRule, compare_runs

            rules = []
            for item in args.metric:
                parts = item.split(":")
                if len(parts) != 3:
                    raise AnomalyLabError("metric must use NAME:higher|lower:THRESHOLD")
                name, direction, threshold = parts
                rules.append(MetricRule(name, direction, float(threshold)))
            result = compare_runs(
                read_csv(args.baseline),
                read_csv(args.candidate),
                _metadata(args.baseline_metadata),
                _metadata(args.candidate_metadata),
                _features(args.features),
                rules,
                args.min_samples,
            )
            payload = result.to_dict()
            print(f"Status: {result.status}; Causality: {result.causality}")
            for metric in result.metrics:
                print(
                    f"{metric.name}: {metric.status} "
                    f"(baseline={metric.baseline}, candidate={metric.candidate}, "
                    f"delta={metric.delta})"
                )
            if result.mismatches:
                print("Metadata mismatches: " + ", ".join(result.mismatches))
            for warning in result.warnings:
                print(f"Warning: {warning}")
            gate_exit = _fail_on_codes(args.fail_on, result)
            if gate_exit is not None:
                _write_json(args.output_json, payload)
                return gate_exit

        elif args.command == "baseline":
            from anomaly_lab.past_baseline import compute_past_baseline

            result = compute_past_baseline(
                read_csv(args.input),
                _features(args.features),
                method=args.method,
                alpha=args.alpha,
                window=args.window,
                order_by=args.order_by,
            )
            payload = {
                "method": result.method,
                "order_by": result.order_by,
                "baseline": {
                    key: [None if pd.isna(value) else float(value) for value in values]
                    for key, values in result.baseline.items()
                },
                "residual": {
                    key: [None if pd.isna(value) else float(value) for value in values]
                    for key, values in result.residual.items()
                },
            }
            print(f"Method: {result.method}; features: {', '.join(_features(args.features))}")
            print("Each baseline uses prior rows only; initial warmup values are null.")

        elif args.command == "change-points":
            from anomaly_lab.change_points import detect_change_points

            result = detect_change_points(
                read_csv(args.input),
                args.feature,
                min_segment=args.min_segment,
                threshold=args.threshold,
                order_by=args.order_by,
            )
            payload = result.to_dict()
            print(
                f"Feature: {result.feature}; candidates: {len(result.candidates)} (retrospective)"
            )
            print("Candidate boundaries are descriptive; root cause is not established.")

        elif args.command == "regimes":
            from anomaly_lab.detectors.dbscan import DBSCANConfig

            result = dbscan_detector.fit_predict(
                read_csv(args.input),
                _features(args.features),
                DBSCANConfig(args.eps, args.min_samples),
                metadata_columns=args.metadata,
            )
            payload = result.to_dict()
            if result.clusters is not None:
                payload["clusters"] = list(result.clusters)
            print(f"DBSCAN rows: {len(result.row_indices)}; noise points: {sum(result.is_anomaly)}")
            print("Cluster IDs are opaque; noise label -1 is not a causal diagnosis.")

        elif args.command == "report":
            try:
                import matplotlib

                matplotlib.use("Agg")
                import matplotlib.pyplot as plt  # noqa: F401
            except ImportError:
                raise AnomalyLabError(
                    "matplotlib is not installed. Install with: pip install '.[viz]'"
                ) from None

            result_path = args.result_json
            output_path = args.output
            payload = json.loads(result_path.read_text(encoding="utf-8"))
            _produce_html_report(payload, output_path)
            print(f"Report: {output_path}")

        elif args.command == "consensus":
            from anomaly_lab.consensus import compute_consensus
            from anomaly_lab.results import DetectionResult

            loaded_results: list[DetectionResult] = []
            for result_path_str in args.results:
                result_path = Path(result_path_str)
                if not result_path.exists():
                    raise AnomalyLabError(f"result file not found: {result_path}")
                raw = json.loads(result_path.read_text(encoding="utf-8"))
                rows_data = raw["rows"]
                scores = [r["score"] for r in rows_data]
                is_anomaly = [r["is_anomaly"] for r in rows_data]
                loaded_results.append(
                    DetectionResult(
                        detector=raw["detector"],
                        row_indices=tuple(r["index"] for r in rows_data),
                        is_anomaly=tuple(is_anomaly),
                        features=tuple(raw["features"]),
                        constant_features=tuple(raw.get("constant_features", [])),
                        parameters=raw.get("parameters", {}),
                        scores=tuple(scores),
                        metadata=raw.get("metadata"),
                    )
                )
            consensus_result = compute_consensus(loaded_results)
            payload = consensus_result.to_dict()
            print(
                f"Consensus: {consensus_result.n_detectors} detectors, "
                f"{consensus_result.n_anomalies_total} total anomaly flags, "
                f"evidence_class={consensus_result.evidence_class}"
            )
            print(
                "Consensus is statistical agreement, NOT ground truth; "
                "3 similar methods count as 1 independent source."
            )

        elif args.command == "inspect":
            from anomaly_lab.registry import ALL_DETECTOR_CONTRACTS, check_applicability
            from anomaly_lab.semantics import classify

            frame = read_csv(args.data)
            profile = classify(frame)

            # Check applicability for each detector
            applicable = []
            not_applicable = []
            for name, contract in ALL_DETECTOR_CONTRACTS.items():
                code, reason = check_applicability(
                    contract,
                    profile,
                    len(frame),
                    constant_features=(),
                    has_missing=frame.isnull().any().any(),
                )
                entry = {"detector": name, "code": code.value, "reason": reason}
                if code.value == "APPLICABLE":
                    applicable.append(entry)
                else:
                    not_applicable.append(entry)

            payload = {
                "schema_version": "0.4",
                "semantics": profile.semantics.value,
                "has_timestamp": profile.has_timestamp,
                "has_group_key": profile.has_group_key,
                "has_pairing_key": profile.has_pairing_key,
                "ordering_field": profile.ordering_field,
                "metric_columns": list(profile.metric_columns),
                "label_column": profile.label_column,
                "anomaly_intervals": [list(iv) for iv in profile.anomaly_intervals],
                "assumptions": list(profile.assumptions),
                "warnings": list(profile.warnings),
                "applicable_detectors": applicable,
                "not_applicable_detectors": not_applicable,
            }

            print(f"Dataset semantics: {profile.semantics.value}")
            print(f"Rows: {len(frame)}; Columns: {len(frame.columns)}")
            print(f"Timestamp detected: {profile.has_timestamp}")
            print(f"Group key detected: {profile.has_group_key}")
            print(f"Pairing key detected: {profile.has_pairing_key}")
            print(f"Ordering field: {profile.ordering_field or '(none)'}")
            print(f"Metric columns: {', '.join(profile.metric_columns) or '(none)'}")
            print(f"Label column: {profile.label_column or '(none)'}")
            if profile.anomaly_intervals:
                print(f"Anomaly intervals: {len(profile.anomaly_intervals)}")
                for start, end in profile.anomaly_intervals:
                    print(f"  [{start}, {end}] inclusive")
            if profile.assumptions:
                print("Assumptions:")
                for a in profile.assumptions:
                    print(f"  - {a}")
            if profile.warnings:
                print("Warnings:")
                for w in profile.warnings:
                    print(f"  ! {w}")
            print(f"\nApplicable detectors ({len(applicable)}):")
            for d in applicable:
                print(f"  [APPLICABLE] {d['detector']}")
            if not_applicable:
                print(f"\nNot applicable detectors ({len(not_applicable)}):")
                for d in not_applicable:
                    print(f"  [{d['code']}] {d['detector']}: {d['reason']}")
            print(
                "\nIMPORTANT: This classification is advisory only. "
                "Telemetry-Anomaly-Lab does NOT silently coerce one dataset type into another."
            )

        elif args.command == "experiment":
            # Dynamically import to avoid hard dependency on harness at module load
            from anomaly_lab.cli_experiment import (
                _resolve_harness_dir,
                _run_experiment_plan,
                _run_experiment_replay,
                _run_experiment_run,
                _run_experiment_validate,
            )

            harness_dir = _resolve_harness_dir(args.harness_dir)
            exp_cmd = args.experiment_command

            if exp_cmd == "validate":
                ok = _run_experiment_validate(args.scenario, harness_dir)
                return 0 if ok else _EXIT_CFG

            elif exp_cmd == "plan":
                try:
                    _run_experiment_plan(args.scenario, harness_dir)
                except AnomalyLabError as exc:
                    print(f"error: {exc}")
                    return _EXIT_CFG
                return 0

            elif exp_cmd == "run":
                return _run_experiment_run(
                    args.scenario,
                    harness_dir,
                    args.output_dir,
                    args.rerun,
                )

            elif exp_cmd == "replay":
                ok = _run_experiment_replay(args.scenario, args.result_json, harness_dir)
                return 0 if ok else _EXIT_CFG

        else:  # analyze
            result = analyze(str(args.input), args.detector, _features(args.features))
            payload = result.to_dict()
            # Enrich with semantic evidence classification
            try:
                from anomaly_lab.evidence import classify_detection_evidence

                record = classify_detection_evidence(result)
                payload["evidence_class"] = record.evidence_class
                payload["confidence"] = record.confidence
                payload["assumptions"] = list(record.assumptions)
                payload["limitations"] = list(record.limitations)
                payload["interpretation"] = record.interpretation
                payload["supporting_metrics"] = record.supporting_metrics
            except Exception:
                pass  # Evidence enrichment is best-effort; core result is primary
            print(
                f"Detector: {result.detector}; rows: {len(result.row_indices)}; "
                f"anomalies: {sum(result.is_anomaly)}"
            )
            print(
                f"Features: {', '.join(result.features)}; constants removed: "
                f"{', '.join(result.constant_features) or 'none'}"
            )
            print("Anomalies are statistical observations, not diagnosed causes.")

    except (AnomalyLabError, ValueError, TypeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return _EXIT_CFG
    return 0 if _write_json(getattr(args, "output_json", None), payload) else _EXIT_CFG


# ─────────────────────────────────────────────────────────────────────────────
# Offline HTML reporting (matplotlib required)
# ─────────────────────────────────────────────────────────────────────────────


def _produce_html_report(payload: dict, output_path: Path) -> None:
    """Write a self-contained HTML report from a result payload dict."""
    import matplotlib

    matplotlib.use("Agg")

    sections = []
    if "detector" in payload:  # analyze result
        sections.append(_analyze_section(payload))
    elif "status" in payload and "metrics" in payload:  # compare result
        sections.append(_compare_section(payload))
    elif "baseline" in payload:  # baseline result
        sections.append(_baseline_section(payload))
    elif "candidates" in payload:  # change-points result
        sections.append(_change_points_section(payload))
    elif "clusters" in payload:  # regimes result
        sections.append(_regimes_section(payload))
    else:
        sections.append("<p>Unknown result type.</p>")

    html = HTML_TEMPLATE.format(
        title="Telemetry Anomaly Lab Report",
        generated=pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
        sections="\n".join(sections),
    )
    output_path.write_text(html, encoding="utf-8")


def _analyze_section(payload: dict) -> str:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = payload["rows"]
    positions = [r["position"] for r in rows]  # noqa: F841
    scores = [r.get("score") for r in rows]
    anomalies = [r["is_anomaly"] for r in rows]

    fig, ax = plt.subplots(figsize=(8, 3))
    colors = ["red" if a else "steelblue" for a in anomalies]
    x = range(len(rows))
    ax.bar(x, [s if s is not None else 0 for s in scores], color=colors, width=0.6)
    ax.set_xlabel("Row position")
    ax.set_ylabel("Anomaly score")
    ax.set_title(f"Detector: {payload['detector']} — {sum(anomalies)} anomaly/ies")
    buf = _fig_to_base64(fig)
    plt.close(fig)
    return (
        f"<h2>Analyze — {payload['detector']}</h2>"
        f"<p>Features: {', '.join(payload['features'])}; "
        f"Constants removed: {', '.join(payload.get('constant_features') or [])}</p>"
        f'<img src="data:image/png;base64,{buf}" />'
    )


def _compare_section(payload: dict) -> str:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    metrics = payload.get("metrics", [])
    names = [m["name"] for m in metrics]
    baselines = [m["baseline"] for m in metrics]
    candidates = [m["candidate"] for m in metrics]

    fig, axes = plt.subplots(1, 2, figsize=(10, 3))
    x = range(len(names))
    axes[0].bar([i - 0.2 for i in x], baselines, 0.4, label="baseline", color="steelblue")
    axes[0].bar([i + 0.2 for i in x], candidates, 0.4, label="candidate", color="coral")
    axes[0].set_xticks(list(x))
    axes[0].set_xticklabels(names, rotation=30, ha="right")
    axes[0].set_ylabel("Value")
    axes[0].set_title("Baseline vs Candidate")
    axes[0].legend()

    statuses = [m["status"] for m in metrics]
    colors = [
        "red" if s == "REGRESSION" else "green" if s == "NO_REGRESSION" else "gray"
        for s in statuses
    ]
    axes[1].bar(list(x), [abs(m.get("relative_change") or 0) for m in metrics], color=colors)
    axes[1].set_xticks(list(x))
    axes[1].set_xticklabels(names, rotation=30, ha="right")
    axes[1].set_ylabel("|Relative change|")
    axes[1].set_title("Change magnitude")
    buf = _fig_to_base64(fig)
    plt.close(fig)

    table_rows = "".join(
        f"<tr><td>{m['name']}</td><td>{m['status']}</td>"
        f"<td>{m.get('baseline')}</td><td>{m.get('candidate')}</td>"
        f"<td>{m.get('delta')}</td></tr>"
        for m in metrics
    )
    return (
        f"<h2>Compare</h2>"
        f"<p>Status: <strong>{payload['status']}</strong>; "
        f"Causality: {payload.get('causality', 'NOT ESTABLISHED')}</p>"
        f"<table><tr><th>Metric</th><th>Status</th><th>Baseline</th>"
        f"<th>Candidate</th><th>Delta</th></tr>{table_rows}</table>"
        f'<img src="data:image/png;base64,{buf}" />'
    )


def _baseline_section(payload: dict) -> str:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    baseline = payload.get("baseline", {})
    residual = payload.get("residual", {})
    if not baseline:
        return "<p>Baseline result has no data.</p>"
    feat_name = next(iter(baseline))
    b_vals = baseline[feat_name]
    r_vals = residual.get(feat_name, [None] * len(b_vals))
    x = range(len(b_vals))

    fig, axes = plt.subplots(1, 2, figsize=(10, 3))
    axes[0].plot(list(x), [v if v is not None else float("nan") for v in b_vals], "b-")
    axes[0].set_title(f"Past-only baseline: {feat_name}")
    axes[0].set_xlabel("Row index")
    axes[0].set_ylabel(feat_name)
    valid_r = [i for i, v in enumerate(r_vals) if v is not None]
    axes[1].plot(valid_r, [r_vals[i] for i in valid_r], "r-")
    axes[1].set_title(f"Residual: {feat_name}")
    axes[1].set_xlabel("Row index")
    buf = _fig_to_base64(fig)
    plt.close(fig)
    return (
        f"<h2>Baseline — {payload.get('method', '?')}</h2>"
        f"<p>Order-by: {payload.get('order_by', 'none')}</p>"
        f'<img src="data:image/png;base64,{buf}" />'
    )


def _change_points_section(payload: dict) -> str:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    candidates = payload.get("candidates", [])
    if not candidates:
        return "<h2>Change Points</h2><p>No candidates found.</p>"

    names = [f"split={c['split_position']}" for c in candidates]
    shifts = [c["median_shift"] for c in candidates]
    supports = [c.get("support", 0) for c in candidates]

    fig, axes = plt.subplots(1, 2, figsize=(10, 3))
    axes[0].bar(names, shifts, color="steelblue")
    axes[0].set_title("Median shift by candidate")
    axes[0].set_ylabel("Shift magnitude")
    axes[0].tick_params(axis="x", rotation=45)
    axes[1].bar(names, supports, color="coral")
    axes[1].set_title("Support (min segment size)")
    axes[1].set_ylabel("Support")
    axes[1].tick_params(axis="x", rotation=45)
    buf = _fig_to_base64(fig)
    plt.close(fig)

    table = "".join(
        f"<tr><td>{c['split_position']}</td><td>{c['left_median']:.4f}</td>"
        f"<td>{c['right_median']:.4f}</td><td>{c['median_shift']:.4f}</td>"
        f"<td>{c.get('left_size', '?')}</td><td>{c.get('right_size', '?')}</td>"
        f"<td>{c.get('support', '?')}</td></tr>"
        for c in candidates
    )
    return (
        f"<h2>Change Points — {payload.get('feature', '?')}</h2>"
        f"<p>{len(candidates)} candidate(s); threshold={payload.get('threshold', '?')}</p>"
        f"<table><tr><th>Split</th><th>Left median</th><th>Right median</th>"
        f"<th>Shift</th><th>Left size</th><th>Right size</th><th>Support</th></tr>"
        f"{table}</table>"
        f'<img src="data:image/png;base64,{buf}" />'
    )


def _regimes_section(payload: dict) -> str:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = payload.get("rows", [])
    clusters = [r.get("cluster", -1) for r in rows]
    scores = [r.get("score") or 0 for r in rows]

    fig, ax = plt.subplots(figsize=(6, 4))
    unique = sorted(set(clusters))
    cmap = plt.cm.get_cmap("tab10", len(unique))
    for i, cl in enumerate(unique):
        xs = [j for j, c in enumerate(clusters) if c == cl]
        ys = [scores[j] for j in xs]
        label = f"cluster {cl}" if cl != -1 else "noise"
        ax.scatter(xs, ys, c=[cmap(i)], label=label, s=20)
    ax.set_xlabel("Row index")
    ax.set_ylabel("Score")
    ax.set_title("DBSCAN Regime Clusters")
    ax.legend()
    buf = _fig_to_base64(fig)
    plt.close(fig)
    return (
        f"<h2>Regimes / DBSCAN Clustering</h2>"
        f"<p>{len([c for c in clusters if c == -1])} noise point(s); "
        f"{len([c for c in clusters if c != -1])} clustered.</p>"
        f'<img src="data:image/png;base64,{buf}" />'
    )


def _fig_to_base64(fig) -> str:
    import base64
    import io

    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=100)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode()


HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>{title}</title>
<style>
body{{ font-family: sans-serif; margin: 2rem; max-width: 900px; margin: auto }}
h2{{ border-bottom: 1px solid #ccc; padding-bottom: .25em; margin-top: 1.5em }}
table{{ border-collapse: collapse; width: 100%; margin: 1em 0 }}
th,td{{ border: 1px solid #ccc; padding: .4em .7em; text-align: left }}
th{{ background: #f5f5f5 }}
img{{ max-width: 100%; height: auto; margin: 1em 0 }}
</style>
</head>
<body>
<h1>{title}</h1>
<p>Generated: {generated}</p>
<p><strong>Causality: NOT ESTABLISHED</strong> — all results are statistical observations.</p>
{sections}
</body>
</html>"""


if __name__ == "__main__":
    raise SystemExit(main())
