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


def main(argv: list[str] | None = None) -> int:
    import argparse
    import json
    import sys
    from pathlib import Path

    parser = argparse.ArgumentParser(prog="anomaly-lab")
    subparsers = parser.add_subparsers(dest="command", required=True)
    analyze_parser = subparsers.add_parser("analyze", help="analyze a telemetry CSV")
    analyze_parser.add_argument("input", type=Path)
    analyze_parser.add_argument(
        "--detector",
        choices=("isolation-forest", "dbscan", "robust-zscore"),
        required=True,
    )
    analyze_parser.add_argument("--features", required=True, help="comma-separated feature names")
    analyze_parser.add_argument("--output-json", type=Path)
    args = parser.parse_args(argv)
    features = [feature.strip() for feature in args.features.split(",") if feature.strip()]
    try:
        result = analyze(str(args.input), args.detector, features)
    except AnomalyLabError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    payload = result.to_dict()
    detected = sum(result.is_anomaly)
    print(f"Detector: {result.detector}; rows: {len(result.row_indices)}; anomalies: {detected}")
    feature_text = ", ".join(result.features)
    constant_text = ", ".join(result.constant_features) or "none"
    print(f"Features: {feature_text}; constants removed: {constant_text}")
    print("Anomalies are statistical observations, not diagnosed causes.")
    if args.output_json:
        try:
            args.output_json.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
        except OSError as exc:
            print(f"error: cannot write JSON output: {exc}", file=sys.stderr)
            return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
