from __future__ import annotations

import argparse
import json
import sys
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
    raise AnomalyLabError(f"unknown detector: {detector}")


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


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="anomaly-lab")
    subparsers = parser.add_subparsers(dest="command", required=True)
    analyze_parser = subparsers.add_parser("analyze", help="analyze a telemetry CSV")
    analyze_parser.add_argument("input", type=Path)
    analyze_parser.add_argument(
        "--detector", choices=("isolation-forest", "dbscan", "robust-zscore"), required=True
    )
    analyze_parser.add_argument("--features", required=True, help="comma-separated feature names")
    analyze_parser.add_argument("--output-json", type=Path)

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
    compare_parser.add_argument("--min-samples", type=int, default=2)
    compare_parser.add_argument("--output-json", type=Path)

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

    points_parser = subparsers.add_parser(
        "change-points", help="scan for retrospective median shifts"
    )
    points_parser.add_argument("input", type=Path)
    points_parser.add_argument("--feature", required=True)
    points_parser.add_argument("--min-segment", type=int, required=True)
    points_parser.add_argument("--threshold", type=float, required=True)
    points_parser.add_argument("--order-by")
    points_parser.add_argument("--output-json", type=Path)

    regimes_parser = subparsers.add_parser("regimes", help="cluster telemetry with DBSCAN")
    regimes_parser.add_argument("input", type=Path)
    regimes_parser.add_argument(
        "--features", required=True, help="comma-separated numeric features"
    )
    regimes_parser.add_argument("--eps", type=float, default=0.5)
    regimes_parser.add_argument("--min-samples", type=int, default=5)
    regimes_parser.add_argument("--metadata", action="append", default=[], metavar="COLUMN")
    regimes_parser.add_argument("--output-json", type=Path)
    return parser


def _metadata(items: list[str]):
    from anomaly_lab.compare import RunMetadata

    parsed = {}
    for item in items:
        if "=" not in item:
            raise AnomalyLabError("metadata values must use KEY=VALUE")
        key, value = item.split("=", 1)
        if not key or key in parsed:
            raise AnomalyLabError("metadata keys must be non-empty and unique")
        parsed[key] = value
    required = {"model", "version", "quantization", "concurrency"}
    if set(parsed) != required:
        raise AnomalyLabError("metadata requires model, version, quantization, concurrency")
    return RunMetadata(**parsed)


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "compare":
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
        else:
            result = analyze(str(args.input), args.detector, _features(args.features))
            payload = result.to_dict()
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
        return 2
    return 0 if _write_json(args.output_json, payload) else 2


if __name__ == "__main__":
    raise SystemExit(main())
