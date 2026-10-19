from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Literal

import numpy as np
import pandas as pd
from scipy import stats

from anomaly_lab.schema import AnomalyLabError

# -------------------------------------------------------------------
# Inference result constants
# -------------------------------------------------------------------
INFERENCE_CHOICES = (
    "IMPROVEMENT_SUPPORTED",
    "REGRESSION_SUPPORTED",
    "EQUIVALENT_WITHIN_MARGIN",
    "DIFFERENCE_DETECTED_DIRECTION_UNCLEAR",
    "INSUFFICIENT_POWER",
    "INCONCLUSIVE",
)


# -------------------------------------------------------------------
# Dataclasses
# -------------------------------------------------------------------


@dataclass(frozen=True, kw_only=True)
class BootstrapResult:
    """Result of a paired bootstrap confidence interval procedure."""

    confidence_interval: tuple[float, float]  # (lower, upper) BCa bootstrap
    original_delta: float
    n_resamples: int
    seed: int
    method: str  # "bca" | "percentile"


@dataclass(frozen=True, kw_only=True)
class ComparisonEvidence:
    """Statistical evidence backing a metric comparison inference."""

    # IMPROVEMENT_SUPPORTED | REGRESSION_SUPPORTED | EQUIVALENT_WITHIN_MARGIN
    # | DIFFERENCE_DETECTED_DIRECTION_UNCLEAR | INSUFFICIENT_POWER | INCONCLUSIVE
    inference_result: str
    evidence_class: str = "COMPARISON_DIFFERENCE"
    delta: float
    relative_delta: float | None
    confidence_interval: tuple[float, float] | None  # (lower, upper) BCa bootstrap
    sample_count: int
    median_baseline: float
    median_candidate: float
    mad_baseline: float | None
    mad_candidate: float | None
    effect_size: float | None  # Cohen's d or Cliff's delta
    equivalence_margin: float | None
    assumptions: tuple[str, ...]
    limitations: tuple[str, ...]
    supporting_metrics: dict


@dataclass(frozen=True)
class MetricComparison:
    """Result of comparing one metric across two runs."""

    name: str
    baseline: float
    candidate: float
    delta: float
    relative_change: float | None
    direction: str
    threshold: float
    threshold_kind: str
    status: str
    # New v0.4 fields:
    evidence: ComparisonEvidence | None = None


@dataclass(frozen=True)
class ComparisonResult:
    status: str
    metrics: tuple[MetricComparison, ...]
    mismatches: tuple[str, ...]
    warnings: tuple[str, ...]
    causality: str = "NOT ESTABLISHED"

    def to_dict(self) -> dict:
        return asdict(self)


# -------------------------------------------------------------------
# Statistical helper functions
# -------------------------------------------------------------------


def robust_spread(x: np.ndarray) -> float:
    """MAD-based robust spread. Returns 0.0 for degenerate inputs."""
    if x.size < 2:
        return 0.0
    median = np.median(x)
    if not math.isfinite(median):
        return 0.0
    deviations = np.abs(x - median)
    mad = np.median(deviations)
    if mad == 0:
        return 0.0
    # Scale factor for consistency with normal distribution
    return float(mad * 1.4826)


def cohens_d(baseline: np.ndarray, candidate: np.ndarray) -> float:
    """Pooled Cohen's d. Returns 0.0 for degenerate inputs."""
    nb = baseline.size
    nc = candidate.size
    if nb < 2 or nc < 2:
        return 0.0
    var_b = float(np.var(baseline, ddof=1))
    var_c = float(np.var(candidate, ddof=1))
    # Degenerate: identical constant groups
    if var_b == 0 and var_c == 0:
        if np.allclose(baseline, candidate):
            return 0.0
        # Different constant means: pooled_std = 0 → undefined; return 0
        return 0.0
    pooled_std = np.sqrt(((nb - 1) * var_b + (nc - 1) * var_c) / (nb + nc - 2))
    if pooled_std == 0:
        return 0.0
    return float((np.mean(candidate) - np.mean(baseline)) / pooled_std)


def cliff_delta(baseline: np.ndarray, candidate: np.ndarray) -> float:
    """Cliff's delta ordinal effect size. Returns 0.0 for degenerate."""
    nb = baseline.size
    nc = candidate.size
    if nb == 0 or nc == 0:
        return 0.0
    # Brute-force dominance count for small-to-medium samples
    n_bigger = n_smaller = n_tied = 0
    for b in baseline:
        for c in candidate:
            if b < c:
                n_bigger += 1
            elif b > c:
                n_smaller += 1
            else:
                n_tied += 1
    total = nb * nc
    if total == 0:
        return 0.0
    return (n_bigger - n_smaller) / total


def _bca_acceleration(x: np.ndarray) -> float:
    """BCa acceleration factor (Efron 1987)."""
    n = x.size
    if n < 3:
        return 0.0
    mean_boot = np.mean(x)
    numerator = np.sum((mean_boot - x) ** 3)
    denominator = 6 * (np.sum((x - mean_boot) ** 2) ** 1.5)
    if denominator == 0:
        return 0.0
    return numerator / denominator


def _bca_bias_correction(observed_theta: float, boot_dist: np.ndarray) -> float:
    """BCa bias correction factor z0."""
    boot_sorted = np.sort(boot_dist)
    # Count how many boot values are below the observed
    n_less = np.searchsorted(boot_sorted, observed_theta, side="left")
    # Proportion below observed
    p_less = n_less / boot_dist.size
    if p_less <= 0 or p_less >= 1:
        return 0.0
    return stats.norm.ppf(p_less)


def paired_bootstrap_ci(
    baseline: np.ndarray,
    candidate: np.ndarray,
    n_resamples: int = 9999,
    seed: int = 42,
    alpha: float = 0.05,
) -> BootstrapResult:
    """Paired bootstrap BCa confidence interval for median difference.

    Uses paired resampling (same indices reshuffled).
    BCa corrects for bias and skewness.
    Returns (lower, upper) percentile bounds.
    """
    rng = np.random.default_rng(seed)
    observed_deltas = candidate - baseline
    observed_median = np.median(observed_deltas)

    boot_deltas = np.empty(n_resamples)
    indices = np.arange(baseline.size)
    for i in range(n_resamples):
        perm = rng.permuted(indices)
        boot_baseline = baseline[perm]
        boot_candidate = candidate[perm]
        boot_deltas[i] = np.median(boot_candidate - boot_baseline)

    # --- BCa correction ---
    accel = _bca_acceleration(observed_deltas)
    z0 = _bca_bias_correction(observed_median, boot_deltas)

    # Adjusted percentile boundaries
    alpha_lower = alpha / 2
    alpha_upper = 1 - alpha / 2

    # Quantile of standard normal
    try:
        z_lower = stats.norm.ppf(alpha_lower)
        z_upper = stats.norm.ppf(alpha_upper)
    except Exception:
        # Fallback to percentile interval on any numerical failure
        lower = np.percentile(boot_deltas, 100 * alpha_lower)
        upper = np.percentile(boot_deltas, 100 * alpha_upper)
        return BootstrapResult(
            confidence_interval=(float(lower), float(upper)),
            original_delta=float(observed_median),
            n_resamples=n_resamples,
            seed=seed,
            method="percentile",
        )

    # Adjusted percentile indices (proportions) for BCa
    adj_lower = stats.norm.cdf(z0 + z_lower / (1 - accel * z_lower))
    adj_upper = stats.norm.cdf(z0 + z_upper / (1 - accel * z_upper))

    adj_lower = float(np.clip(adj_lower, 0, 1))
    adj_upper = float(np.clip(adj_upper, 0, 1))

    lower = np.percentile(boot_deltas, 100 * adj_lower)
    upper = np.percentile(boot_deltas, 100 * adj_upper)

    return BootstrapResult(
        confidence_interval=(float(lower), float(upper)),
        original_delta=float(observed_median),
        n_resamples=n_resamples,
        seed=seed,
        method="bca",
    )


def tost_equivalence_test(
    baseline: np.ndarray,
    candidate: np.ndarray,
    margin: float,
) -> dict:
    """Two One-Sided T-tests for equivalence.

    Returns dict with:
    - t1, t2 (one-sided t statistics)
    - p_equivalence = max(p1, p2)
    - TOST_passed: bool
    - delta, margin
    - interpretation
    Uses scipy.stats by default.
    """
    nb = baseline.size
    nc = candidate.size
    if nb < 2 or nc < 2 or margin <= 0:
        return {
            "t1": 0.0,
            "t2": 0.0,
            "p_equivalence": 1.0,
            "TOST_passed": False,
            "delta": 0.0,
            "margin": margin,
            "interpretation": "INCONCLUSIVE_INSUFFICIENT_DATA",
        }

    # Paired or independent based on size — use independent by default (safer)
    delta = np.mean(candidate) - np.mean(baseline)
    # Pooled standard error
    var_b = np.var(baseline, ddof=1)
    var_c = np.var(candidate, ddof=1)
    se = np.sqrt(var_b / nb + var_c / nc)
    if se == 0:
        if delta == 0:
            return {
                "t1": 0.0,
                "t2": 0.0,
                "p_equivalence": 0.0,
                "TOST_passed": True,
                "delta": 0.0,
                "margin": margin,
                "interpretation": "EQUIVALENT_WITHIN_MARGIN",
            }
        return {
            "t1": 0.0,
            "t2": 0.0,
            "p_equivalence": 1.0,
            "TOST_passed": False,
            "delta": float(delta),
            "margin": margin,
            "interpretation": "INCONCLUSIVE_NO_VARIANCE",
        }

    # TOST: two one-sided t-tests (Welch version)
    # t_lower: reject H0 that delta <= -margin (want large positive)
    # t_upper: reject H0 that delta >= +margin (want large negative)
    t_lower = (delta + margin) / se
    t_upper = (delta - margin) / se

    df = min(nb, nc) - 1  # Welch-Satterthwaite approximation
    p1 = 1 - stats.t.cdf(t_lower, df)  # p for rejecting delta < -margin
    p2 = stats.t.cdf(t_upper, df)  # p for rejecting delta > +margin

    p_equivalence = max(p1, p2)
    tost_passed = p_equivalence < 0.05

    if tost_passed:
        interpretation = "EQUIVALENT_WITHIN_MARGIN"
    elif delta > margin:
        interpretation = "CANDIDATE_OUTPERFORMS_BEYOND_MARGIN"
    elif delta < -margin:
        interpretation = "BASELINE_OUTPERFORMS_BEYOND_MARGIN"
    else:
        interpretation = "DIFFERENCE_DETECTED_WITHIN_MARGIN"

    return {
        "t1": float(t_lower),
        "t2": float(t_upper),
        "p_equivalence": float(p_equivalence),
        "TOST_passed": tost_passed,
        "delta": float(delta),
        "margin": float(margin),
        "interpretation": interpretation,
    }


# -------------------------------------------------------------------
# Core comparison engine
# -------------------------------------------------------------------


def _finite_mean(series: pd.Series, feature: str, label: str) -> float:
    # Scale before summing so finite values near float limits do not overflow.
    values = [float(value) for value in series]
    scale = max(abs(value) for value in values)
    mean = 0.0 if scale == 0 else math.fsum(value / scale for value in values) / len(values) * scale
    if not math.isfinite(mean):
        raise AnomalyLabError(
            f"computed mean for feature {feature!r} in {label} is not finite; rescale input values"
        )
    return mean


def _finite_difference(after: float, before: float, feature: str) -> float:
    difference = after - before
    if not math.isfinite(difference):
        raise AnomalyLabError(
            f"computed delta for feature {feature!r} is not finite; rescale input values"
        )
    return difference


def _sanitize_floats(obj: dict) -> dict:
    """Replace non-finite float values in a dict with None for JSON safety."""
    result = {}
    for k, v in obj.items():
        if isinstance(v, float):
            result[k] = None if not math.isfinite(v) else v
        else:
            result[k] = v
    return result


def _determine_inference(
    n: int,
    ci_lower: float,
    ci_upper: float,
    direction: str,
    equivalence_margin: float | None,
    tost_passed: bool,
) -> str:
    """Determine inference result from bootstrap CI and TOST."""
    if n < 4:  # too small for defensible CI
        return "INSUFFICIENT_POWER"
    if equivalence_margin is not None and tost_passed:
        return "EQUIVALENT_WITHIN_MARGIN"
    ci_includes_zero = ci_lower <= 0 <= ci_upper
    if ci_lower > 0 and direction == "higher":
        return "IMPROVEMENT_SUPPORTED"
    if ci_upper < 0 and direction == "higher":
        return "REGRESSION_SUPPORTED"
    if ci_lower > 0 and direction == "lower":
        return "REGRESSION_SUPPORTED"
    if ci_upper < 0 and direction == "lower":
        return "IMPROVEMENT_SUPPORTED"
    if ci_includes_zero:
        return "DIFFERENCE_DETECTED_DIRECTION_UNCLEAR"
    return "INCONCLUSIVE"


def compare_runs(
    baseline: pd.DataFrame,
    candidate: pd.DataFrame,
    baseline_metadata: RunMetadata,
    candidate_metadata: RunMetadata,
    features: list[str],
    rules: list[MetricRule],
    min_samples: int = 2,
    # --- Phase D new parameters ---
    seed: int = 42,
    equivalence_margin: float | None = None,
    n_resamples: int = 9999,
    paired: bool = False,
) -> ComparisonResult:
    """Compare two telemetry runs and return per-metric ComparisonResults with evidence.

    Parameters
    ----------
    seed : int
        Random seed for bootstrap resampling.
    equivalence_margin : float | None
        If set, run TOST equivalence test with this margin (as a fraction of baseline
        scale, e.g. 0.10 means 10% equivalence margin).
    n_resamples : int
        Number of bootstrap resamples for confidence intervals.
    paired : bool
        If True, use paired bootstrap (indices shuffled together).
        If False, use unpaired (independent) bootstrap.
    """
    if not isinstance(baseline, pd.DataFrame) or not isinstance(candidate, pd.DataFrame):
        raise AnomalyLabError("baseline and candidate must be DataFrames")
    if not isinstance(baseline_metadata, RunMetadata) or not isinstance(
        candidate_metadata, RunMetadata
    ):
        raise AnomalyLabError("baseline and candidate metadata must be RunMetadata")
    if (
        not isinstance(features, list)
        or not features
        or any(not isinstance(f, str) or not f.strip() for f in features)
    ):
        raise AnomalyLabError("features must be a non-empty list of names")
    if not isinstance(rules, list) or any(not isinstance(rule, MetricRule) for rule in rules):
        raise AnomalyLabError("rules must be a list of MetricRule values")
    if isinstance(min_samples, bool) or not isinstance(min_samples, int) or min_samples < 1:
        raise AnomalyLabError("min_samples must be a positive integer")
    mismatches = tuple(
        key
        for key in asdict(baseline_metadata)
        if getattr(baseline_metadata, key) != getattr(candidate_metadata, key)
        and key not in baseline_metadata.allow_difference
    )
    if mismatches:
        return ComparisonResult("NOT_COMPARABLE", (), mismatches, ())
    if (
        len(set(features)) != len(features)
        or len(rules) != len(features)
        or set(features) != {rule.name for rule in rules}
    ):
        raise AnomalyLabError("features must have exactly one metric rule each")
    missing = (set(features) - set(baseline.columns)) | (set(features) - set(candidate.columns))
    if missing:
        raise AnomalyLabError(f"missing comparison feature(s): {', '.join(sorted(missing))}")
    numeric: dict[tuple[str, str], pd.Series] = {}
    for label, frame in (("baseline", baseline), ("candidate", candidate)):
        if frame.empty:
            raise AnomalyLabError(f"feature data in {label} is empty")
        for name in features:
            series = frame[name]
            if not pd.api.types.is_numeric_dtype(series.dtype):
                raise AnomalyLabError(f"feature {name!r} in {label} must be numeric")
            if series.empty or not bool(series.map(math.isfinite).all()):
                raise AnomalyLabError(f"feature {name!r} in {label} must be non-empty and finite")
            numeric[(label, name)] = series
    rules_by_name = {rule.name: rule for rule in rules}

    # --- Insufficient sample path (unchanged, no evidence) ---
    if len(baseline) < min_samples or len(candidate) < min_samples:
        metrics = []
        for name in features:
            before = _finite_mean(numeric[("baseline", name)], name, "baseline")
            after = _finite_mean(numeric[("candidate", name)], name, "candidate")
            delta = _finite_difference(after, before, name)
            relative_change = None if before == 0 else delta / abs(before)
            if relative_change is not None and not math.isfinite(relative_change):
                raise AnomalyLabError(
                    f"computed relative change for feature {name!r} is not finite; "
                    "rescale input values"
                )
            metrics.append(
                MetricComparison(
                    name,
                    before,
                    after,
                    delta,
                    relative_change,
                    rules_by_name[name].direction,
                    rules_by_name[name].threshold,
                    "relative_change_proportion",
                    "INSUFFICIENT_SAMPLE",
                    evidence=None,
                )
            )
        return ComparisonResult(
            "INSUFFICIENT_SAMPLE",
            tuple(metrics),
            (),
            (f"Small sample: each run requires at least {min_samples} rows.",),
        )

    # --- Full comparison with evidence ---
    metrics = []
    warnings: list[str] = []

    for name in features:
        rule = rules_by_name[name]
        before_arr = numeric[("baseline", name)].to_numpy()
        after_arr = numeric[("candidate", name)].to_numpy()

        before_mean = _finite_mean(numeric[("baseline", name)], name, "baseline")
        after_mean = _finite_mean(numeric[("candidate", name)], name, "candidate")
        delta = _finite_difference(after_mean, before_mean, name)
        relative_change = None if before_mean == 0 else delta / abs(before_mean)
        if relative_change is not None and not math.isfinite(relative_change):
            raise AnomalyLabError(
                f"computed relative change for feature {name!r} is not finite; rescale input values"
            )

        # --- Bootstrap CI ---
        n = min(len(before_arr), len(after_arr))
        ci: tuple[float, float] | None = None
        bootstrap_result: BootstrapResult | None = None

        try:
            if paired:
                bootstrap_result = paired_bootstrap_ci(
                    before_arr, after_arr, n_resamples=n_resamples, seed=seed
                )
            else:
                # Unpaired: resample independently from each group
                rng = np.random.default_rng(seed)
                boot_deltas = np.empty(n_resamples)
                for i in range(n_resamples):
                    boot_b = rng.choice(before_arr, size=len(before_arr), replace=True)
                    boot_c = rng.choice(after_arr, size=len(after_arr), replace=True)
                    boot_deltas[i] = np.median(boot_c) - np.median(boot_b)
                ci_lower = np.percentile(boot_deltas, 100 * 0.025)
                ci_upper = np.percentile(boot_deltas, 100 * 0.975)
                bootstrap_result = BootstrapResult(
                    confidence_interval=(float(ci_lower), float(ci_upper)),
                    original_delta=float(np.median(after_arr) - np.median(before_arr)),
                    n_resamples=n_resamples,
                    seed=seed,
                    method="percentile",
                )
            ci = bootstrap_result.confidence_interval
            # Reject CI with non-finite bounds (can arise from overflow)
            if ci is not None and (not math.isfinite(ci[0]) or not math.isfinite(ci[1])):
                warnings.append(f"Bootstrap CI for {name!r} has non-finite bounds; discarding.")
                ci = None
        except Exception as e:
            warnings.append(f"Bootstrap CI failed for {name!r}: {e}")
            ci = None

        # --- TOST equivalence ---
        tost_result: dict = {"TOST_passed": False, "interpretation": "TOST_NOT_RUN"}
        margin_used: float | None = None
        if equivalence_margin is not None:
            # Convert relative margin to absolute
            abs_margin = (
                equivalence_margin * abs(before_mean) if before_mean != 0 else equivalence_margin
            )
            margin_used = abs_margin
            try:
                tost_result = tost_equivalence_test(before_arr, after_arr, abs_margin)
            except Exception as e:
                warnings.append(f"TOST failed for {name!r}: {e}")

        # --- Effect sizes ---
        effect_size: float | None = None
        try:
            es = float(cohens_d(before_arr, after_arr))
            effect_size = es if math.isfinite(es) else None
        except Exception:
            pass
        try:
            cd = float(cliff_delta(before_arr, after_arr))
            # Use Cliff's delta when Cohen's d is near zero and Cliff's is not
            if effect_size is None or (abs(effect_size) < 0.01 and abs(cd) > 0.01):
                effect_size = cd if math.isfinite(cd) else None
        except Exception:
            pass

        # --- Median and MAD ---
        median_baseline_raw = np.median(before_arr)
        median_candidate_raw = np.median(after_arr)
        # Guard against non-finite medians (e.g. from overflow with extreme values)
        median_baseline = (
            float(median_baseline_raw) if math.isfinite(median_baseline_raw) else before_mean
        )
        median_candidate = (
            float(median_candidate_raw) if math.isfinite(median_candidate_raw) else after_mean
        )
        mad_baseline: float | None = robust_spread(before_arr) if before_arr.size >= 2 else None
        mad_candidate: float | None = robust_spread(after_arr) if after_arr.size >= 2 else None

        # --- Inference result ---
        if ci is not None:
            ci_lower, ci_upper = ci
            inference_result = _determine_inference(
                n,
                ci_lower,
                ci_upper,
                rule.direction,
                equivalence_margin,
                tost_result.get("TOST_passed", False),
            )
        else:
            inference_result = "INCONCLUSIVE"

        # --- Regressions status (backwards-compatible) ---
        regression = relative_change is not None and (
            relative_change < -rule.threshold
            if rule.direction == "higher"
            else relative_change > rule.threshold
        )
        status = "REGRESSION" if regression else "NO_REGRESSION"

        # --- Build evidence ---
        evidence = ComparisonEvidence(
            inference_result=inference_result,
            evidence_class="COMPARISON_DIFFERENCE",
            delta=delta,
            relative_delta=relative_change,
            confidence_interval=ci,
            sample_count=n,
            median_baseline=median_baseline,
            median_candidate=median_candidate,
            mad_baseline=mad_baseline,
            mad_candidate=mad_candidate,
            effect_size=effect_size,
            equivalence_margin=margin_used,
            assumptions=(
                "Observations are independent.",
                "Bootstrap assumes representative sampling.",
                "TOST assumes approximate normality for t-test approximation.",
            ),
            limitations=(
                "Bootstrap CI may be unreliable with small n (< 100).",
                "BCa correction requires n >= 3.",
                "Effect size (Cohen's d) assumes equal variances for independent samples.",
                "TOST is sensitive to the equivalence margin choice.",
            ),
            supporting_metrics={
                "cohens_d": effect_size,
                "cliff_delta": (
                    float(cliff_delta(before_arr, after_arr)) if effect_size is not None else None
                ),
                "tost": _sanitize_floats(tost_result),
            },
        )

        metrics.append(
            MetricComparison(
                name=name,
                baseline=before_mean,
                candidate=after_mean,
                delta=delta,
                relative_change=relative_change,
                direction=rule.direction,
                threshold=rule.threshold,
                threshold_kind="relative_change_proportion",
                status=status,
                evidence=evidence,
            )
        )

    status = (
        "REGRESSION"
        if any(metric.status == "REGRESSION" for metric in metrics)
        else "NO_REGRESSION"
    )
    return ComparisonResult(
        status,
        tuple(metrics),
        (),
        tuple(warnings) if warnings else (),
    )


@dataclass(frozen=True)
class RunMetadata:
    """Explicit run identity used only for comparability, never as a metric."""

    model: str
    version: str
    quantization: str
    concurrency: str
    allow_difference: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        for field, value in asdict(self).items():
            if field == "allow_difference":
                continue  # validated below
            if not isinstance(value, str) or not value.strip():
                raise AnomalyLabError("run metadata values must be non-empty strings")
        known = frozenset(k for k in asdict(self) if k != "allow_difference")
        for dim in self.allow_difference:
            if not isinstance(dim, str) or not dim.strip():
                raise AnomalyLabError("allow_difference values must be non-empty strings")
            if dim not in known:
                raise AnomalyLabError(f"allow_difference references unknown dimension: {dim!r}")


@dataclass(frozen=True)
class MetricRule:
    """Threshold is a relative-change proportion: 0.10 means 10%."""

    name: str
    direction: Literal["higher", "lower"]
    threshold: float

    def __post_init__(self) -> None:
        if (
            not isinstance(self.name, str)
            or not self.name.strip()
            or self.direction not in ("higher", "lower")
            or isinstance(self.threshold, bool)
            or not isinstance(self.threshold, (int, float))
            or not math.isfinite(self.threshold)
            or self.threshold < 0
        ):
            raise AnomalyLabError(
                "metric requires a name, direction higher/lower, and finite "
                "non-negative relative threshold"
            )
