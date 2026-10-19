#!/usr/bin/env python3
"""
Empirical benchmark: 8 anomaly detectors on synthetic + real-ish datasets.
Detectors: robust-zscore, isolation-forest, dbscan, ecod, lof, knn, hbos, pca
"""

import json
import os
import time
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
from anomaly_lab.detectors.pyod_adapters import fit_predict_ecod, fit_predict_knn, fit_predict_lof

# Local implementations
from anomaly_lab.detectors.robust_zscore import fit_predict as robust_zscore_run

warnings.filterwarnings("ignore")

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


# ─────────────────────────────────────────────────────────────────────────────
# Dataset generators
# ─────────────────────────────────────────────────────────────────────────────

# Per-seed RNG state tracker — incremented by i so each (i,seed) combo is unique
_RNG_STATE = {}


def make_synthetic(i: int, seed: int) -> dict:
    """Generate one synthetic dataset with injected point anomalies.

    Varies: n_samples (50-500), n_features (2-10), anomaly_fraction (0.02-0.15).
    Each (i, seed) pair is unique because we advance the RNG state i times
    before generation, rather than resetting to the same seed each time.
    """
    # Initialize or resume the shared RNG for this seed
    if seed not in _RNG_STATE:
        _RNG_STATE[seed] = np.random.default_rng(seed)
    rng = _RNG_STATE[seed]

    # Advance RNG state by i steps so config i with same seed produces
    # different data than config i-1 ( 而不是每次从相同种子重新开始 )
    for _ in range(i):
        rng.integers(0, 2**31)

    n_samples = rng.integers(50, 501)  # 50..500
    n_features = rng.integers(2, 11)  # 2..10
    anomaly_frac = rng.uniform(0.02, 0.15)

    n_anomalies = max(1, int(n_samples * anomaly_frac))

    # Normal data: standard normal
    X = rng.standard_normal((n_samples, n_features))

    # Inject anomalies: random uniform in [mean-5*std, mean+5*std] per feature
    anomalies = rng.uniform(low=-5.0, high=5.0, size=(n_anomalies, n_features))
    X_with_anomalies = np.vstack([X, anomalies])

    # Ground-truth labels: 0=normal, 1=anomaly
    y = np.array([0] * n_samples + [1] * n_anomalies)

    # Injected anomaly "score": distance from origin as proxy (larger = more extreme)
    anomaly_scores = np.zeros(len(X_with_anomalies))
    anomaly_scores[n_samples:] = np.linalg.norm(anomalies, axis=1)

    df = pd.DataFrame(X_with_anomalies, columns=[f"f{j}" for j in range(n_features)])
    df["__anomaly_label__"] = y
    df["__anomaly_score__"] = anomaly_scores

    return {
        "name": f"synthetic_{i}_seed{seed}",
        "type": "synthetic",
        "df": df,
        "feature_cols": [f"f{j}" for j in range(n_features)],
        "y": y,
        "anomaly_scores": anomaly_scores,
        "n_normal": n_samples,
        "n_anomalies": n_anomalies,
    }


def make_correlated(seed: int = 99) -> dict:
    """2D correlated normal data; outliers break the correlation."""
    rng = np.random.default_rng(seed)
    n_normal = 200
    n_anomalies = 20

    # Normal: x ~ N(0,1), y = 0.8*x + N(0,0.6)  (correlation ≈ 0.8)
    x_normal = rng.standard_normal(n_normal)
    y_normal = 0.8 * x_normal + 0.6 * rng.standard_normal(n_normal)

    X_normal = np.column_stack([x_normal, y_normal])

    # Anomalies: uniform random — break the correlation
    x_anom = rng.uniform(-4, 4, n_anomalies)
    y_anom = rng.uniform(-4, 4, n_anomalies)
    X_anom = np.column_stack([x_anom, y_anom])

    X = np.vstack([X_normal, X_anom])
    y = np.array([0] * n_normal + [1] * n_anomalies)

    # Anomaly score: perpendicular distance from the correlation line
    slope = 0.8
    intercept = 0.0
    residuals = np.abs(y_normal - slope * x_normal - intercept) / np.sqrt(1 + slope**2)
    # For injected anomalies: larger when far from the line
    anom_residuals = np.abs(y_anom - slope * x_anom - intercept) / np.sqrt(1 + slope**2)
    anomaly_scores = np.zeros(len(X))
    anomaly_scores[n_normal:] = anom_residuals
    anomaly_scores[:n_normal] = residuals

    df = pd.DataFrame(X, columns=["x", "y"])
    df["__anomaly_label__"] = y
    df["__anomaly_score__"] = anomaly_scores

    return {
        "name": "correlated_multivariate",
        "type": "correlated",
        "df": df,
        "feature_cols": ["x", "y"],
        "y": y,
        "anomaly_scores": anomaly_scores,
        "n_normal": n_normal,
        "n_anomalies": n_anomalies,
    }


def make_local_density(seed: int = 7) -> dict:
    """Clusters of normal points + outliers between clusters.

    3 clusters of 50 normal points each + 15 outliers in the gaps.
    """
    rng = np.random.default_rng(seed)

    # 3 cluster centres
    centres = np.array([[0, 0], [10, 0], [5, 8]])
    cluster_std = 0.8

    X_list = []
    for c in centres:
        X_list.append(c + cluster_std * rng.standard_normal((50, 2)))

    # 15 outliers between clusters
    outlier_centre = np.array([5.0, 3.5])  # between cluster 1 and 2 centre
    X_outliers = outlier_centre + 1.5 * rng.standard_normal((15, 2))

    X = np.vstack(X_list + [X_outliers])
    y = np.array([0] * 150 + [1] * 15)

    # Normal points get low scores; outliers get high
    # Score = distance to nearest cluster centre
    anomaly_scores = np.zeros(len(X))
    for i, x_row in enumerate(X):
        if i >= 150:
            # outlier
            anomaly_scores[i] = np.linalg.norm(x_row - outlier_centre)
        else:
            # normal: distance to own cluster centre
            ci = i // 50
            anomaly_scores[i] = np.linalg.norm(x_row - centres[ci]) if ci < 3 else 0.0

    df = pd.DataFrame(X, columns=["x", "y"])
    df["__anomaly_label__"] = y
    df["__anomaly_score__"] = anomaly_scores

    return {
        "name": "local_density",
        "type": "local_density",
        "df": df,
        "feature_cols": ["x", "y"],
        "y": y,
        "anomaly_scores": anomaly_scores,
        "n_normal": 150,
        "n_anomalies": 15,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Detector wrappers (return raw scores + runtime ms)
# ─────────────────────────────────────────────────────────────────────────────


def _standardize(frame: pd.DataFrame, features: list[str]) -> np.ndarray:
    scaler = StandardScaler()
    return scaler.fit_transform(frame[features].values)


def run_robust_zscore(df, features):
    t0 = time.perf_counter()
    result = robust_zscore_run(df, features)
    ms = (time.perf_counter() - t0) * 1000
    return np.array(result.scores), ms


def run_isolation_forest(df, features):
    from anomaly_lab.detectors.isolation_forest import IsolationForestConfig

    t0 = time.perf_counter()
    config = IsolationForestConfig(contamination=CONTAMINATION)
    result = isolation_forest_run(df, features, config=config)
    ms = (time.perf_counter() - t0) * 1000
    return np.array(result.scores), ms


def run_dbscan(df, features):
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
    t0 = time.perf_counter()
    result = fit_predict_ecod(df, features)
    ms = (time.perf_counter() - t0) * 1000
    return np.array(result.scores), ms


def run_lof(df, features):
    t0 = time.perf_counter()
    result = fit_predict_lof(df, features)
    ms = (time.perf_counter() - t0) * 1000
    return np.array(result.scores), ms


def run_knn(df, features):
    t0 = time.perf_counter()
    result = fit_predict_knn(df, features)
    ms = (time.perf_counter() - t0) * 1000
    return np.array(result.scores), ms


def run_hbos(df, features):
    t0 = time.perf_counter()
    X = _standardize(df, features)
    model = HBOS(contamination=CONTAMINATION, n_bins=10)
    model.fit(X)
    scores = model.decision_scores_.flatten()
    ms = (time.perf_counter() - t0) * 1000
    return scores, ms


def run_pca(df, features):
    t0 = time.perf_counter()
    X = _standardize(df, features)
    model = PyODPCA(contamination=CONTAMINATION)
    model.fit(X)
    scores = model.decision_scores_.flatten()
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


# ─────────────────────────────────────────────────────────────────────────────
# Metrics
# ─────────────────────────────────────────────────────────────────────────────


def roc_auc(y_true, y_score):
    try:
        return roc_auc_score(y_true, y_score)
    except ValueError:
        return float("nan")


def spearman(y_true, y_score):
    # Use only anomaly-labelled points for spearman
    mask = y_true >= 0
    if mask.sum() < 3:
        return float("nan")
    rho, _ = spearmanr(y_true[mask], y_score[mask])
    return float(rho) if not np.isnan(rho) else float("nan")


def top5_overlap(score_dict: dict) -> float:
    """Fraction of overlapping top-5 flagged anomalies across all detector pairs."""
    dets = list(score_dict.keys())
    top5_sets = []
    for d in dets:
        scores = score_dict[d]
        top5 = set(np.argsort(scores)[-5:][::-1])
        top5_sets.append(top5)

    n_pairs = 0
    total_overlap = 0
    for i, j in combinations(range(len(dets)), 2):
        intersection = len(top5_sets[i] & top5_sets[j])
        total_overlap += intersection
        n_pairs += 1

    # Average overlap per pair, expressed as fraction of 5
    if n_pairs == 0:
        return float("nan")
    return total_overlap / (n_pairs * 5) * 100


def pairwise_spearman(score_dict: dict) -> dict:
    """Spearman correlation between all detector pairs."""
    dets = list(score_dict.keys())
    result = {}
    for i, j in combinations(range(len(dets)), 2):
        rho, _ = spearmanr(score_dict[dets[i]], score_dict[dets[j]])
        key = f"{dets[i]}___{dets[j]}"
        result[key] = float(rho) if not np.isnan(rho) else float("nan")
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────


def main():
    # Build datasets
    datasets = []

    # 10 synthetic × 3 seeds = 30 datasets
    for i in range(10):
        for seed in RANDOM_SEEDS:
            datasets.append(make_synthetic(i, seed))

    # Correlated multivariate
    datasets.append(make_correlated(seed=99))

    # Local density
    datasets.append(make_local_density(seed=7))

    results = []
    per_detector = {
        d: {"roc_auc": [], "spearman": [], "top5_overlaps": [], "runtime_ms": []}
        for d in DETECTOR_NAMES
    }

    pairwise_accum = None

    for _ds_idx, ds in enumerate(datasets):
        df = ds["df"]
        features = ds["feature_cols"]
        y = ds["y"]
        anomaly_scores = ds["anomaly_scores"]

        score_dict = {}
        runtime_dict = {}

        for det_name in DETECTOR_NAMES:
            try:
                scores, ms = DETECTOR_RUNNERS[det_name](df, features)
            except Exception as exc:
                print(f"  [{ds['name']}] {det_name} FAILED: {exc}")
                scores = np.zeros(len(df))
                ms = float("nan")

            score_dict[det_name] = scores
            runtime_dict[det_name] = ms
            per_detector[det_name]["runtime_ms"].append(ms)

            auc = roc_auc(y, scores)
            rho = spearman(y, anomaly_scores)
            per_detector[det_name]["roc_auc"].append(auc)
            per_detector[det_name]["spearman"].append(rho)

        # Top-5 overlap for this dataset
        top5_pct = top5_overlap(score_dict)
        for det_name in DETECTOR_NAMES:
            per_detector[det_name]["top5_overlaps"].append(top5_pct)

        # Pairwise Spearman
        pairwise = pairwise_spearman(score_dict)
        if pairwise_accum is None:
            pairwise_accum = {k: [] for k in pairwise}
        for k, v in pairwise.items():
            pairwise_accum[k].append(v)

        ds_result = {
            "name": ds["name"],
            "type": ds["type"],
            "n_samples": len(df),
            "n_features": len(features),
            "n_anomalies": int(ds["n_anomalies"]),
            "detector_scores": {d: score_dict[d].tolist() for d in DETECTOR_NAMES},
            "runtimes_ms": runtime_dict,
            "roc_auc": {d: per_detector[d]["roc_auc"][-1] for d in DETECTOR_NAMES},
            "spearman": {d: per_detector[d]["spearman"][-1] for d in DETECTOR_NAMES},
        }
        results.append(ds_result)

    # Aggregate per-detector
    per_det_agg = {}
    for d in DETECTOR_NAMES:
        roc_list = per_detector[d]["roc_auc"]
        sp_list = per_detector[d]["spearman"]
        top5_list = per_detector[d]["top5_overlaps"]
        rt_list = per_detector[d]["runtime_ms"]

        per_det_agg[d] = {
            "roc_auc_mean": float(np.nanmean(roc_list)),
            "roc_auc_std": float(np.nanstd(roc_list)),
            "spearman_mean": float(np.nanmean(sp_list)),
            "spearman_std": float(np.nanstd(sp_list)),
            "top5_overlap_pct_mean": float(np.nanmean(top5_list)),
            "top5_overlap_pct_std": float(np.nanstd(top5_list)),
            "runtime_ms_mean": float(np.nanmean(rt_list)),
            "runtime_ms_std": float(np.nanstd(rt_list)),
        }

    # Aggregate pairwise
    pairwise_agg = {}
    for k, vals in pairwise_accum.items():
        pairwise_agg[k] = {
            "spearman_mean": float(np.nanmean(vals)),
            "spearman_std": float(np.nanstd(vals)),
        }

    # ─── Conclusions ────────────────────────────────────────────────────────
    # Grounded in actual numbers
    def decide(det_name: str, metrics: dict) -> tuple[str, str]:
        roc = metrics["roc_auc_mean"]
        sp = metrics["spearman_mean"]
        top5 = metrics["top5_overlap_pct_mean"]
        rt = metrics["runtime_ms_mean"]

        # Compute avg spearman with all other detectors
        other_sp_vals = []
        for k, v in pairwise_agg.items():
            if det_name in k.split("___"):
                other_sp_vals.append(v["spearman_mean"])
        avg_other_sp = np.nanmean(other_sp_vals) if other_sp_vals else 0.0

        # Low ROC-AUC + low spearman → unstable / poor performance
        if roc < 0.55 and sp < 0.3:
            return "REJECT_UNSTABLE", (
                f"roc_auc_mean={roc:.3f}, spearman_mean={sp:.3f} — "
                f"poor detection on synthetic/correlated data."
            )

        # High overlap with others (avg spearman > 0.85) → redundant
        if avg_other_sp > 0.85:
            return "REJECT_REDUNDANT", (
                f"avg_spearman_with_others={avg_other_sp:.3f}, top5_overlap={top5:.1f}% — "
                f"highly correlated with existing detectors."
            )

        # Good ROC-AUC and spearman, not redundant → ADD
        if roc >= 0.70 or sp >= 0.5:
            return "ADD", (
                f"roc_auc_mean={roc:.3f}, spearman_mean={sp:.3f}, "
                f"top5_overlap={top5:.1f}%, runtime_ms={rt:.2f} — "
                f"competitive detection quality, avg_spearman_with_others={avg_other_sp:.3f}."
            )

        # Default: RESEARCH_ONLY
        return "RESEARCH_ONLY", (
            f"roc_auc_mean={roc:.3f}, spearman_mean={sp:.3f}, "
            f"top5_overlap={top5:.1f}% — marginal; requires further investigation."
        )

    conclusions = {}
    for d in DETECTOR_NAMES:
        label, reason = decide(d, per_det_agg[d])
        conclusions[d] = {"verdict": label, "reason": reason}

    # ─── Build report ────────────────────────────────────────────────────────
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
    }

    out_path = os.environ.get("BENCHMARK_OUTPUT", "benchmark_results_fixed.json")
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)

    print(f"\nBenchmark complete. Results saved to {out_path}")
    print("\n── Per-detector summary ──")
    for d in DETECTOR_NAMES:
        m = per_det_agg[d]
        print(
            f"  {d:20s}  AUC={m['roc_auc_mean']:.3f}±{m['roc_auc_std']:.3f}  "
            f"Spearman={m['spearman_mean']:.3f}±{m['spearman_std']:.3f}  "
            f"Top5olap={m['top5_overlap_pct_mean']:.1f}%  "
            f"rt={m['runtime_ms_mean']:.2f}ms"
        )

    # ─── Spearman redundancy warning ──────────────────────────────────────────
    print("\n── Detector-pair Spearman redundancy check (>0.95 = redundant) ──")
    redundant_pairs = []
    for k, v in pairwise_agg.items():
        mean_sp = v["spearman_mean"]
        if mean_sp > 0.95:
            d1, d2 = k.split("___")
            print(f"  ⚠️  {d1} ↔ {d2}: mean Spearman = {mean_sp:.3f} — REDUNDANT")
            redundant_pairs.append(k)
    if not redundant_pairs:
        print("  (none above 0.95 threshold)")

    print("\n── Conclusions ──")
    for d, c in conclusions.items():
        print(f"  {d:20s}  {c['verdict']:22s}  {c['reason']}")


if __name__ == "__main__":
    main()
