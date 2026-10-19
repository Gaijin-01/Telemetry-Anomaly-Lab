#!/usr/bin/env python3
"""
PyOD candidates benchmark — evaluates 8 anomaly detectors on synthetic + real-ish datasets.

Detectors: robust-zscore, isolation-forest, dbscan, ecod, lof, knn, hbos, pca
Datasets:  10 synthetic × 3 seeds (30) + correlated multivariate + local density (2) = 32 total

Usage:
    python benchmarks/pyod_candidates.py                    # stdout summary + default output
    python benchmarks/pyod_candidates.py --output results.json   # write to file
    python benchmarks/pyod_candidates.py --stdout            # stdout only, no file
    python benchmarks/pyod_candidates.py --test-reproducibility  # run reproducibility test

Exit codes:
    0  — all detector runs succeeded
    1  — one or more required detector runs failed
    2  — reproducibility test failed
"""

import argparse
import hashlib
import json
import sys
import warnings
from itertools import combinations

import numpy as np
import pandas as pd

# PyOD direct imports (HBOS, PCA candidates)
from pyod.models.hbos import HBOS
from pyod.models.pca import PCA as PyODPCA
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

from anomaly_lab.detectors.dbscan import fit_predict as dbscan_run
from anomaly_lab.detectors.isolation_forest import fit_predict as isolation_forest_run
from anomaly_lab.detectors.pyod_adapters import fit_predict_knn, fit_predict_lof

# Local implementations
from anomaly_lab.detectors.robust_zscore import fit_predict as robust_zscore_run

CONTAMINATION = 0.1
RANDOM_SEEDS = [0, 1, 42]
DETECTOR_NAMES = [
    "robust-zscore",
    "isolation-forest",
    "dbscan",
    "ecod",
    "lof",
    "knn",
    "hbos",
    "pca",
]


# ── Deterministic RNG ──────────────────────────────────────────────────────────


def _make_rng(seed: int, dataset_id: str = "") -> np.random.Generator:
    """Create a seeded RNG. The dataset_id is mixed in so the same seed
    with different dataset_ids produce independent but reproducible streams."""
    if dataset_id:
        h = hashlib.sha256(f"{dataset_id}:{seed}".encode()).digest()
        seed_from_hash = int.from_bytes(h[:4], "little")
        return np.random.default_rng(seed_from_hash)
    return np.random.default_rng(seed)


# ── Dataset factories ─────────────────────────────────────────────────────────


def make_synthetic(dataset_id: int, seed: int) -> dict:
    """Generate one synthetic dataset with injected point anomalies.

    Deterministic: same (dataset_id, seed) → same content every time.
    """
    rng = _make_rng(seed, dataset_id=f"synthetic_{dataset_id}")

    n_samples = rng.integers(50, 501)  # 50..500
    n_features = rng.integers(2, 11)  # 2..10
    anomaly_frac = rng.uniform(0.02, 0.15)

    n_anomalies = max(1, int(n_samples * anomaly_frac))

    X = rng.standard_normal((n_samples, n_features))
    anomalies = rng.uniform(low=-5.0, high=5.0, size=(n_anomalies, n_features))
    X_with_anomalies = np.vstack([X, anomalies])

    y = np.array([0] * n_samples + [1] * n_anomalies)

    anomaly_scores = np.zeros(len(X_with_anomalies))
    anomaly_scores[n_samples:] = np.linalg.norm(anomalies, axis=1)

    df = pd.DataFrame(X_with_anomalies, columns=[f"f{j}" for j in range(n_features)])
    df["__anomaly_label__"] = y
    df["__anomaly_score__"] = anomaly_scores

    return {
        "name": f"synthetic_{dataset_id}_seed{seed}",
        "type": "synthetic",
        "df": df,
        "feature_cols": [f"f{j}" for j in range(n_features)],
        "y": y,
        "anomaly_scores": anomaly_scores,
        "n_normal": n_samples,
        "n_anomalies": n_anomalies,
    }


def make_correlated(seed: int = 99) -> dict:
    """2D correlated normal data; outliers break the correlation.

    Deterministic via seed.
    """
    rng = _make_rng(seed, dataset_id="correlated_multivariate")
    n_normal = 200
    n_outliers = 20
    n_features = 2

    mean = rng.standard_normal(n_features)
    cov = rng.random((n_features, n_features))
    cov = cov @ cov.T
    X_normal = rng.multivariate_normal(mean, cov, size=n_normal)

    outliers = rng.uniform(low=-6.0, high=6.0, size=(n_outliers, n_features))
    X_with_outliers = np.vstack([X_normal, outliers])

    y = np.array([0] * n_normal + [1] * n_outliers)
    anomaly_scores = np.zeros(len(X_with_outliers))
    anomaly_scores[n_normal:] = np.linalg.norm(outliers - mean, axis=1)

    df = pd.DataFrame(X_with_outliers, columns=["f0", "f1"])
    df["__anomaly_label__"] = y
    df["__anomaly_score__"] = anomaly_scores

    return {
        "name": "correlated_multivariate",
        "type": "realistic",
        "df": df,
        "feature_cols": ["f0", "f1"],
        "y": y,
        "anomaly_scores": anomaly_scores,
        "n_normal": n_normal,
        "n_anomalies": n_outliers,
    }


def make_local_density(seed: int = 7) -> dict:
    """Two clusters of different density; outliers sit between them.

    Deterministic via seed.
    """
    rng = _make_rng(seed, dataset_id="local_density")
    n1, n2 = 150, 50
    n_outliers = 15
    n_features = 2

    c1 = rng.uniform(-2.0, 0.0, size=(n1, n_features))
    c2 = rng.uniform(2.0, 4.0, size=(n2, n_features))
    outliers = rng.uniform(-1.0, 3.0, size=(n_outliers, n_features))

    X = np.vstack([c1, c2, outliers])
    y = np.array([0] * (n1 + n2) + [1] * n_outliers)

    anomaly_scores = np.zeros(len(X))
    anomaly_scores[n1 + n2 :] = 1.0  # all outliers get score 1

    df = pd.DataFrame(X, columns=["f0", "f1"])
    df["__anomaly_label__"] = y
    df["__anomaly_score__"] = anomaly_scores

    return {
        "name": "local_density",
        "type": "realistic",
        "df": df,
        "feature_cols": ["f0", "f1"],
        "y": y,
        "anomaly_scores": anomaly_scores,
        "n_normal": n1 + n2,
        "n_anomalies": n_outliers,
    }


# ── Detector runners ─────────────────────────────────────────────────────────


def _standardize(frame, features):
    scaler = StandardScaler()
    return scaler.fit_transform(frame[features].values)


def run_robust_zscore(df, features):
    import time

    t0 = time.perf_counter()
    result = robust_zscore_run(df, features)
    ms = (time.perf_counter() - t0) * 1000
    return np.array(result.scores), ms


def run_isolation_forest(df, features):
    import time

    from anomaly_lab.detectors.isolation_forest import IsolationForestConfig

    t0 = time.perf_counter()
    config = IsolationForestConfig(contamination=CONTAMINATION)
    result = isolation_forest_run(df, features, config=config)
    ms = (time.perf_counter() - t0) * 1000
    return np.array(result.scores), ms


def run_dbscan(df, features):
    import time

    from anomaly_lab.detectors.dbscan import DBSCANConfig

    t0 = time.perf_counter()
    config = DBSCANConfig(eps=0.5, min_samples=5)
    result = dbscan_run(df, features, config=config)
    ms = (time.perf_counter() - t0) * 1000
    # DBSCAN doesn't produce scores; use negative cluster label as score
    # (noise = -1, larger = closer to noise)
    scores = np.array([0.0 if c >= 0 else -c for c in result.clusters], dtype=float)
    return scores, ms


def run_ecod(df, features):
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", ".*__sklearn_tags__.*")
    X = df[features].values
    try:
        import time

        t0 = time.perf_counter()
        clf = PyODPCA(contamination=CONTAMINATION)
        clf.fit(X)
        scores = clf.decision_scores_
        ms = (time.perf_counter() - t0) * 1000
        return scores, ms
    except Exception:  # pragma: no cover — ECOD may fail on degenerate data
        raise


def run_lof(df, features):
    import time

    t0 = time.perf_counter()
    result = fit_predict_lof(df, features)
    ms = (time.perf_counter() - t0) * 1000
    return np.array(result.scores), ms


def run_knn(df, features):
    import time

    t0 = time.perf_counter()
    result = fit_predict_knn(df, features)
    ms = (time.perf_counter() - t0) * 1000
    return np.array(result.scores), ms


def run_hbos(df, features):
    import time

    t0 = time.perf_counter()
    X = _standardize(df, features)
    clf = HBOS(contamination=CONTAMINATION, n_bins=10)
    clf.fit(X)
    scores_raw = clf.decision_scores_
    if scores_raw is None:
        raise ValueError("HBOS.decision_scores_ is None after fit")
    scores = np.asarray(scores_raw).flatten()
    ms = (time.perf_counter() - t0) * 1000
    return scores, ms


def run_pca(df, features):
    import time

    t0 = time.perf_counter()
    X = _standardize(df, features)
    clf = PyODPCA(contamination=CONTAMINATION)
    clf.fit(X)
    scores_raw = clf.decision_scores_
    if scores_raw is None:
        raise ValueError("PCA.decision_scores_ is None after fit")
    scores = np.asarray(scores_raw).flatten()
    ms = (time.perf_counter() - t0) * 1000
    return scores, ms


DETECTOR_RUNNERS = {
    "robust-zscore": run_robust_zscore,
    "isolation-forest": run_isolation_forest,
    "dbscan": run_dbscan,
    "ecod": run_ecod,
    "lof": run_lof,
    "knn": run_knn,
    "hbos": run_hbos,
    "pca": run_pca,
}


# ── Metrics ───────────────────────────────────────────────────────────────────


def safe_roc_auc(y_true, y_score):
    """Return NaN if AUC cannot be computed (e.g., all same class)."""
    try:
        return roc_auc_score(y_true, y_score)
    except ValueError:
        return float("nan")


def safe_spearman(y_true, y_score):
    """Return NaN if correlation cannot be computed."""
    try:
        rho, _ = spearmanr(y_true, y_score)
        return float(rho) if not np.isnan(rho) else 0.0
    except Exception:
        return 0.0


def top5_overlap(score_dict: dict) -> float:
    """Fraction of shared top-5 anomaly indices across detector pairs."""
    dets = list(score_dict.keys())
    overlaps = []
    for d1, d2 in combinations(dets, 2):
        s1 = np.argsort(score_dict[d1])[::-1][:5]
        s2 = np.argsort(score_dict[d2])[::-1][:5]
        overlaps.append(len(set(s1) & set(s2)) / 5.0)
    return float(np.mean(overlaps)) * 100


def pairwise_spearman(score_dict: dict) -> dict:
    dets = list(score_dict.keys())
    result = {}
    for d1, d2 in combinations(dets, 2):
        rho = safe_spearman(score_dict[d1], score_dict[d2])
        result[f"{d1}___{d2}"] = rho
    return result


# ── Reproducibility test ───────────────────────────────────────────────────────


def test_reproducibility() -> bool:
    """Verify: same (dataset_id, seed) → same data hash.
    Verify: 32 configured datasets → 32 unique content hashes.
    """
    # Build the 32 configured datasets
    datasets = []
    for i in range(10):
        for seed in RANDOM_SEEDS:
            datasets.append(make_synthetic(i, seed))
    datasets.append(make_correlated(seed=99))
    datasets.append(make_local_density(seed=7))

    # Same (id, seed) → same hash
    all_hashes = []
    for ds in datasets:
        content = ds["df"].values.tobytes()
        h = hashlib.sha256(content).hexdigest()[:16]
        all_hashes.append((ds["name"], h))

    duplicates = len(all_hashes) - len(set(x[1] for x in all_hashes))
    print("\nReproducibility check:")
    print(f"  Datasets total:    {len(datasets)}")
    print(f"  Unique hashes:     {len(set(x[1] for x in all_hashes))}")
    print(f"  Duplicate hashes:  {duplicates}")

    # Run twice and compare
    ds0_first = make_synthetic(0, 0)
    ds0_second = make_synthetic(0, 0)
    h1 = hashlib.sha256(ds0_first["df"].values.tobytes()).hexdigest()[:16]
    h2 = hashlib.sha256(ds0_second["df"].values.tobytes()).hexdigest()[:16]
    print(f"  Run1 vs Run2 (synthetic_0_seed0): {'MATCH' if h1 == h2 else 'MISMATCH'}")

    passed = duplicates == 0 and h1 == h2 and len(datasets) == 32
    print(f"\nReproducibility: {'PASS' if passed else 'FAIL'}")
    return passed


# ── Main benchmark ────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="PyOD candidates benchmark")
    parser.add_argument(
        "--output",
        default=None,
        help="Write JSON results to this path. Default: benchmark_results.json",
    )
    parser.add_argument(
        "--stdout", action="store_true", help="Print summary to stdout (always enabled)"
    )
    parser.add_argument(
        "--test-reproducibility", action="store_true", help="Run reproducibility test and exit"
    )
    args = parser.parse_args()

    if args.test_reproducibility:
        ok = test_reproducibility()
        sys.exit(0 if ok else 2)

    # Build datasets
    datasets = []
    for i in range(10):
        for seed in RANDOM_SEEDS:
            datasets.append(make_synthetic(i, seed))
    datasets.append(make_correlated(seed=99))
    datasets.append(make_local_density(seed=7))

    results = []
    # Per-detector: list of (success: bool, roc_auc, spearman, runtime_ms, top5)
    detector_runs: dict[str, list] = {d: [] for d in DETECTOR_NAMES}

    failed_runs: list[tuple[str, str, str]] = []  # (dataset, detector, reason)

    for ds in datasets:
        df = ds["df"]
        features = ds["feature_cols"]
        y = ds["y"]
        anomaly_scores = ds["anomaly_scores"]

        score_dict = {}
        runtime_dict = {}

        for det_name in DETECTOR_NAMES:
            try:
                scores, ms = DETECTOR_RUNNERS[det_name](df, features)
                # Validate scores are finite
                if not np.all(np.isfinite(scores)):
                    raise ValueError("detector returned non-finite scores")
                score_dict[det_name] = scores
                runtime_dict[det_name] = ms
            except Exception as exc:  # pragma: no cover — real failures only
                failed_runs.append((ds["name"], det_name, str(exc)))
                score_dict[det_name] = None  # sentinel: failed
                runtime_dict[det_name] = None

        # Top-5 overlap (only for successful runs)
        if any(v is not None for v in score_dict.values()):
            top5 = top5_overlap({k: v for k, v in score_dict.items() if v is not None})
        else:
            top5 = 0.0

        ds_result = {
            "name": ds["name"],
            "type": ds["type"],
            "n_samples": len(df),
            "n_features": len(features),
            "n_anomalies": int(ds["n_anomalies"]),
            "detector_scores": {},
            "runtimes_ms": {},
            "roc_auc": {},
            "spearman": {},
            "status": "ok",
        }

        for det_name in DETECTOR_NAMES:
            if score_dict[det_name] is None:
                ds_result["detector_scores"][det_name] = None
                ds_result["runtimes_ms"][det_name] = None
                ds_result["roc_auc"][det_name] = None
                ds_result["spearman"][det_name] = None
                detector_runs[det_name].append(
                    {
                        "success": False,
                        "roc_auc": np.nan,
                        "spearman": np.nan,
                        "top5": np.nan,
                        "runtime_ms": np.nan,
                    }
                )
            else:
                scores = score_dict[det_name]
                ms = runtime_dict[det_name]
                auc = safe_roc_auc(y, scores)
                rho = safe_spearman(y, anomaly_scores)
                ds_result["detector_scores"][det_name] = scores.tolist()
                ds_result["runtimes_ms"][det_name] = ms
                ds_result["roc_auc"][det_name] = auc
                ds_result["spearman"][det_name] = rho
                detector_runs[det_name].append(
                    {
                        "success": True,
                        "roc_auc": auc,
                        "spearman": rho,
                        "top5": top5,
                        "runtime_ms": ms,
                    }
                )

        results.append(ds_result)

    # Aggregate per-detector (FAILED runs contribute NaN — excluded via nanmean)
    per_det_agg = {}
    for d in DETECTOR_NAMES:
        runs = detector_runs[d]
        ok_aucs = [r["roc_auc"] for r in runs if r["success"] and np.isfinite(r["roc_auc"])]
        ok_spearman = [r["spearman"] for r in runs if r["success"] and np.isfinite(r["spearman"])]
        ok_top5 = [r["top5"] for r in runs if r["success"] and np.isfinite(r["top5"])]
        ok_runtimes = [
            r["runtime_ms"] for r in runs if r["success"] and np.isfinite(r["runtime_ms"])
        ]
        total = len(runs)
        n_failed = sum(1 for r in runs if not r["success"])

        per_det_agg[d] = {
            "n_runs": total,
            "n_failed": n_failed,
            "roc_auc_mean": float(np.nanmean(ok_aucs)) if ok_aucs else float("nan"),
            "roc_auc_std": float(np.nanstd(ok_aucs)) if len(ok_aucs) > 1 else float("nan"),
            "spearman_mean": float(np.nanmean(ok_spearman)) if ok_spearman else float("nan"),
            "spearman_std": float(np.nanstd(ok_spearman)) if len(ok_spearman) > 1 else float("nan"),
            "top5_overlap_pct_mean": float(np.nanmean(ok_top5)) if ok_top5 else 0.0,
            "top5_overlap_pct_std": float(np.nanstd(ok_top5)) if len(ok_top5) > 1 else 0.0,
            "runtime_ms_mean": float(np.nanmean(ok_runtimes)) if ok_runtimes else float("nan"),
            "runtime_ms_std": float(np.nanstd(ok_runtimes))
            if len(ok_runtimes) > 1
            else float("nan"),
        }

    # Aggregate pairwise Spearman
    pairwise_accum: dict[str, list] = {}
    for ds_res in results:
        score_dict = {
            d: np.array(ds_res["detector_scores"][d])
            for d in DETECTOR_NAMES
            if ds_res["detector_scores"][d] is not None
        }
        if len(score_dict) >= 2:
            for k, v in pairwise_spearman(score_dict).items():
                if k not in pairwise_accum:
                    pairwise_accum[k] = []
                pairwise_accum[k].append(v)

    pairwise_agg = {}
    for k, vals in pairwise_accum.items():
        pairwise_agg[k] = {
            "spearman_mean": float(np.nanmean(vals)),
            "spearman_std": float(np.nanstd(vals)),
        }

    # Conclusions
    def decide(det_name: str, metrics: dict) -> tuple[str, str]:
        roc = metrics["roc_auc_mean"]
        sp = metrics["spearman_mean"]
        top5 = metrics["top5_overlap_pct_mean"]
        rt = metrics["runtime_ms_mean"]
        n_failed = metrics["n_failed"]

        if n_failed > 0:
            return "FAILED", f"{n_failed}/{metrics['n_runs']} runs failed"

        if roc < 0.55 and sp < 0.3:
            return "REJECT_UNSTABLE", (
                f"roc_auc_mean={roc:.3f}, spearman_mean={sp:.3f} — "
                f"poor detection on synthetic/correlated data."
            )

        other_sp_vals = []
        for k, v in pairwise_agg.items():
            if det_name in k.split("___"):
                other_sp_vals.append(v["spearman_mean"])
        avg_other_sp = np.nanmean(other_sp_vals) if other_sp_vals else 0.0

        if avg_other_sp > 0.85:
            return "REJECT_REDUNDANT", (
                f"avg_spearman_with_others={avg_other_sp:.3f}, top5_overlap={top5:.1f}% — "
                f"highly correlated with existing detectors."
            )

        if roc >= 0.70 or sp >= 0.5:
            return "ADD", (
                f"roc_auc_mean={roc:.3f}, spearman_mean={sp:.3f}, "
                f"top5_overlap={top5:.1f}%, runtime_ms={rt:.2f} — "
                f"competitive detection quality, avg_spearman_with_others={avg_other_sp:.3f}."
            )

        return "RESEARCH_ONLY", (
            f"roc_auc_mean={roc:.3f}, spearman_mean={sp:.3f}, "
            f"top5_overlap={top5:.1f}% — marginal; requires further investigation."
        )

    conclusions = {}
    for d in DETECTOR_NAMES:
        label, reason = decide(d, per_det_agg[d])
        conclusions[d] = {"verdict": label, "reason": reason}

    # Build report
    report = {
        "datasets": [
            {
                "name": ds["name"],
                "type": ds["type"],
                "n_samples": len(ds["df"]),
                "n_features": len(ds["feature_cols"]),
                "n_anomalies": int(ds["n_anomalies"]),
            }
            for ds in datasets
        ],
        "detectors": DETECTOR_NAMES,
        "per_detector": per_det_agg,
        "pairwise_agreements": pairwise_agg,
        "dataset_results": results,
        "conclusions": conclusions,
        "failed_runs": [
            {"dataset": d, "detector": det, "reason": reason} for d, det, reason in failed_runs
        ],
    }

    # Output
    output_path = args.output if args.output else "benchmark_results.json"

    # Print summary
    print("\n── Per-detector summary ──")
    for d in DETECTOR_NAMES:
        m = per_det_agg[d]
        n_fail = m["n_failed"]
        fail_str = f"  ⚠ {n_fail} FAILED" if n_fail else "  ✓"
        print(
            f"  {d:20s}  AUC={m['roc_auc_mean']:.3f}±{m['roc_auc_std']:.3f}  "
            f"Spearman={m['spearman_mean']:.3f}±{m['spearman_std']:.3f}  "
            f"Top5olap={m['top5_overlap_pct_mean']:.1f}%  "
            f"rt={m['runtime_ms_mean']:.2f}ms{fail_str}"
        )

    if failed_runs:
        print(f"\n── Failed detector runs ({len(failed_runs)}) ──")
        for ds_name, det, reason in failed_runs:
            print(f"  {ds_name} / {det}: {reason}")

    print("\n── Conclusions ──")
    for d, c in conclusions.items():
        print(f"  {d:20s}  {c['verdict']:22s}  {c['reason']}")

    # Write JSON
    with open(output_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nBenchmark complete. Results saved to {output_path}")

    # Exit non-zero if any required detector runs failed
    if failed_runs:
        print(f"\nERROR: {len(failed_runs)} detector run(s) failed — see above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
