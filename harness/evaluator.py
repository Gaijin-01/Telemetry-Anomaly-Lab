"""Scenario harness evaluator: SEMANTIC correctness checks beyond exit codes.

The evaluator verifies that an experiment result matches the expected evidence
contract defined in a scenario TOML — including evidence class, confidence
bounds, supporting metric tolerances, and assumption coverage.  Exit code
checks are a separate mechanical layer; this module handles meaning.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from anomaly_lab.evidence import EvidenceClass


# ─────────────────────────────────────────────────────────────────────────────
# Result types
# ─────────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True, kw_only=True)
class SemanticCheck:
    """One semantic assertion about a result."""

    field: str  # dot-path into the result dict, e.g. "evidence_class" or "supporting_metrics.f1"
    expected: Any
    actual: Any
    passed: bool
    tolerance: float | None  # relative tolerance for numeric comparisons
    note: str


@dataclass(frozen=True, kw_only=True)
class EvaluationOutcome:
    scenario: str
    passed: bool
    checks: list[SemanticCheck]
    errors: list[str]
    warnings: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "scenario": self.scenario,
            "passed": self.passed,
            "checks": [
                {
                    "field": c.field,
                    "expected": c.expected,
                    "actual": c.actual,
                    "passed": c.passed,
                    "tolerance": c.tolerance,
                    "note": c.note,
                }
                for c in self.checks
            ],
            "errors": self.errors,
            "warnings": self.warnings,
        }


# ─────────────────────────────────────────────────────────────────────────────
# Numeric helpers
# ─────────────────────────────────────────────────────────────────────────────


def _float_equal(a: Any, b: Any, rel_tol: float = 1e-5) -> bool:
    """Return True if two values are numerically equal within relative tolerance."""
    try:
        a_f, b_f = float(a), float(b)
    except (TypeError, ValueError):
        return a == b
    if math.isnan(a_f) and math.isnan(b_f):
        return True
    if math.isinf(a_f) and math.isinf(b_f) and (a_f > 0) == (b_f > 0):
        return True
    if a_f == b_f:
        return True
    diff = abs(a_f - b_f)
    scale = max(1.0, abs(a_f), abs(b_f))
    return diff <= rel_tol * scale


def _get_nested(data: dict, path: str) -> Any:
    """Navigate a dot-separated key path into a dict."""
    key: str
    for key in path.split("."):
        if not isinstance(data, dict):
            return KeyError(key)
        data = data.get(key)  # type: ignore
    return data


# ─────────────────────────────────────────────────────────────────────────────
# Evidence-class-level semantic checks
# ─────────────────────────────────────────────────────────────────────────────


_EVIDENCE_CLASS_HIERARCHY = {
    EvidenceClass.POINT_ANOMALY.value: {"POINT_ANOMALY", "LOCAL_OUTLIER", "GLOBAL_OUTLIER"},
    EvidenceClass.LOCAL_OUTLIER.value: {"LOCAL_OUTLIER", "GLOBAL_OUTLIER"},
    EvidenceClass.GLOBAL_OUTLIER.value: {"GLOBAL_OUTLIER"},
    EvidenceClass.DENSITY_NOISE.value: {"DENSITY_NOISE"},
    EvidenceClass.REGIME_SHIFT.value: {"REGIME_SHIFT"},
    EvidenceClass.DISTRIBUTION_SHIFT.value: {"DISTRIBUTION_SHIFT"},
    EvidenceClass.VARIANCE_COLLAPSE.value: {"VARIANCE_COLLAPSE"},
    EvidenceClass.COMPARISON_DIFFERENCE.value: {"COMPARISON_DIFFERENCE", "COMPARISON_EQUIVALENT"},
    EvidenceClass.COMPARISON_EQUIVALENT.value: {"COMPARISON_EQUIVALENT"},
    EvidenceClass.INCONCLUSIVE.value: {"INCONCLUSIVE"},
    EvidenceClass.NOT_APPLICABLE.value: {"NOT_APPLICABLE"},
}


def _check_evidence_class(
    actual: str,
    expected: str,
    allow_superset: bool = True,
) -> SemanticCheck:
    """Check evidence class with optional superset relaxation."""
    if actual == expected:
        return SemanticCheck(
            field="evidence_class",
            expected=expected,
            actual=actual,
            passed=True,
            tolerance=None,
            note="exact match",
        )
    if allow_superset and expected in _EVIDENCE_CLASS_HIERARCHY:
        allowed = _EVIDENCE_CLASS_HIERARCHY[expected]
        if actual in allowed:
            return SemanticCheck(
                field="evidence_class",
                expected=f"{expected} (or subset)",
                actual=actual,
                passed=True,
                tolerance=None,
                note=f"within-OK superset {allowed}",
            )
    return SemanticCheck(
        field="evidence_class",
        expected=expected,
        actual=actual,
        passed=False,
        tolerance=None,
        note="mismatch",
    )


_STATUS_TO_EVIDENCE_CLASS = {
    "NOT_COMPARABLE": EvidenceClass.NOT_APPLICABLE.value,
    "INSUFFICIENT_SAMPLE": EvidenceClass.INCONCLUSIVE.value,
    "REGRESSION": EvidenceClass.COMPARISON_DIFFERENCE.value,
    "NO_REGRESSION": EvidenceClass.COMPARISON_DIFFERENCE.value,
}


def _resolve_evidence_class(result_payload: dict[str, Any]) -> str:
    """Resolve the evidence class from a result payload.

    For DetectionResult payloads, evidence_class is at the top level.
    For ComparisonResult payloads, evidence_class is absent and must be
    derived from the status field.
    """
    ec = result_payload.get("evidence_class", "")
    if ec:
        return ec
    # Comparison result — derive from status
    status = result_payload.get("status", "")
    return _STATUS_TO_EVIDENCE_CLASS.get(status, "")


# ─────────────────────────────────────────────────────────────────────────────
# Main evaluator
# ─────────────────────────────────────────────────────────────────────────────


def evaluate_scenario(
    scenario_name: str,
    scenario_spec: dict[str, Any],
    result_payload: dict[str, Any],
) -> EvaluationOutcome:
    """Evaluate a result_payload against the contracts defined in scenario_spec.

    Parameters
    ----------
    scenario_name : str
        Human-readable scenario identifier (used in reporting).
    scenario_spec : dict[str, Any]
        Parsed scenario TOML as a dict.  Keys include:
        - expected_evidence_class : str
        - expected_confidence_min : float | None
        - expected_confidence_max : float | None
        - required_evidence_fields : list[str]  (dot-path fields that must be present)
        - tolerance : float  (relative tolerance for numeric fields)
        - expected_detector : str | None
        - required_assumptions : list[str]  (substrings that must appear)
        - forbidden_assumptions : list[str]  (substrings that must NOT appear)
        - expected_semantic_tags : list[str]  (for semantics checks)
        - anomaly_count_min / anomaly_count_max : int | None
    result_payload : dict[str, Any]
        The JSON result produced by running an experiment.

    Returns
    -------
    EvaluationOutcome
        Structured pass/fail with per-check details and error/warning lists.
    """
    checks: list[SemanticCheck] = []
    errors: list[str] = []
    warnings: list[str] = []

    # ── Evidence class ──────────────────────────────────────────────────────
    ec_spec = scenario_spec.get("expected_evidence_class")
    if ec_spec:
        actual_ec = _resolve_evidence_class(result_payload)
        check = _check_evidence_class(actual_ec, ec_spec)
        checks.append(check)
        if not check.passed:
            errors.append(
                f"evidence_class mismatch: expected {ec_spec!r}, got {actual_ec!r}"
            )

    # ── Confidence bounds ──────────────────────────────────────────────────
    conf_min = scenario_spec.get("expected_confidence_min")
    conf_max = scenario_spec.get("expected_confidence_max")
    if conf_min is not None or conf_max is not None:
        actual_conf = result_payload.get("confidence")
        if actual_conf is None:
            errors.append("confidence is None but bounds were specified in scenario")
            checks.append(
                SemanticCheck(
                    field="confidence",
                    expected=f"[{conf_min}, {conf_max}]",
                    actual=None,
                    passed=False,
                    tolerance=None,
                    note="missing",
                )
            )
        else:
            passed = True
            if conf_min is not None and actual_conf < conf_min:
                passed = False
            if conf_max is not None and actual_conf > conf_max:
                passed = False
            checks.append(
                SemanticCheck(
                    field="confidence",
                    expected=f"[{conf_min}, {conf_max}]",
                    actual=actual_conf,
                    passed=passed,
                    tolerance=None,
                    note="within bounds" if passed else "out of bounds",
                )
            )
            if not passed:
                errors.append(
                    f"confidence {actual_conf} outside specified range "
                    f"[{conf_min}, {conf_max}]"
                )

    # ── Required evidence fields (must be present and non-null) ─────────────
    tolerance = scenario_spec.get("tolerance", 1e-5)
    for field_path in scenario_spec.get("required_evidence_fields", []):
        val = _get_nested(result_payload, field_path)
        present = val is not None and not (isinstance(val, float) and math.isnan(val))
        checks.append(
            SemanticCheck(
                field=field_path,
                expected="present and non-null",
                actual=val,
                passed=present,
                tolerance=None,
                note="" if present else "missing",
            )
        )
        if not present:
            errors.append(f"required field '{field_path}' is missing or null")

    # ── Numeric supporting-metric tolerances ──────────────────────────────
    for metric_spec in scenario_spec.get("supporting_metrics", []):
        m_name = metric_spec.get("name")
        m_expected = metric_spec.get("expected")
        m_tol = metric_spec.get("tolerance", tolerance)
        if m_name is None or m_expected is None:
            continue
        actual_val = result_payload.get("supporting_metrics", {}).get(m_name)
        if actual_val is None:
            passed = False
            note = "missing"
        else:
            passed = _float_equal(actual_val, m_expected, rel_tol=m_tol)
            note = "ok" if passed else f"diff={abs(float(actual_val) - float(m_expected))}"
        checks.append(
            SemanticCheck(
                field=f"supporting_metrics.{m_name}",
                expected=m_expected,
                actual=actual_val,
                passed=passed,
                tolerance=m_tol,
                note=note,
            )
        )
        if not passed:
            errors.append(
                f"supporting_metrics.{m_name} = {actual_val!r}, "
                f"expected ≈ {m_expected!r} (rel_tol={m_tol})"
            )

    # ── Detector name ───────────────────────────────────────────────────────
    expected_detector = scenario_spec.get("expected_detector")
    if expected_detector:
        actual_detector = result_payload.get("detector", "")
        passed = actual_detector == expected_detector
        checks.append(
            SemanticCheck(
                field="detector",
                expected=expected_detector,
                actual=actual_detector,
                passed=passed,
                tolerance=None,
                note="" if passed else "mismatch",
            )
        )
        if not passed:
            errors.append(
                f"detector mismatch: expected {expected_detector!r}, "
                f"got {actual_detector!r}"
            )

    # ── Required assumptions ───────────────────────────────────────────────
    for required in scenario_spec.get("required_assumptions", []):
        assumptions = result_payload.get("assumptions", [])
        found = any(required in a for a in assumptions)
        checks.append(
            SemanticCheck(
                field="assumptions",
                expected=f"contains {required!r}",
                actual=list(assumptions),
                passed=found,
                tolerance=None,
                note="" if found else "missing assumption",
            )
        )
        if not found:
            errors.append(f"assumption {required!r} not found in result")

    # ── Forbidden assumptions ─────────────────────────────────────────────
    for forbidden in scenario_spec.get("forbidden_assumptions", []):
        assumptions = result_payload.get("assumptions", [])
        found = any(forbidden in a for a in assumptions)
        checks.append(
            SemanticCheck(
                field="assumptions",
                expected=f"NOT contains {forbidden!r}",
                actual=list(assumptions),
                passed=not found,
                tolerance=None,
                note="" if not found else "forbidden assumption present",
            )
        )
        if found:
            errors.append(f"forbidden assumption {forbidden!r} found in result")

    # ── Anomaly count range ────────────────────────────────────────────────
    rows = result_payload.get("rows", [])
    anomaly_count = sum(1 for r in rows if r.get("is_anomaly"))
    for bound_key, bound_val in [
        ("anomaly_count_min", scenario_spec.get("anomaly_count_min")),
        ("anomaly_count_max", scenario_spec.get("anomaly_count_max")),
    ]:
        if bound_val is None:
            continue
        key = "anomaly_count"
        passed = True
        if bound_key.endswith("_min") and anomaly_count < bound_val:
            passed = False
        if bound_key.endswith("_max") and anomaly_count > bound_val:
            passed = False
        checks.append(
            SemanticCheck(
                field=key,
                expected=f"{bound_key}={bound_val}",
                actual=anomaly_count,
                passed=passed,
                tolerance=None,
                note="" if passed else f"out of range [{scenario_spec.get('anomaly_count_min')}, {scenario_spec.get('anomaly_count_max')}]",
            )
        )
        if not passed:
            errors.append(
                f"anomaly_count={anomaly_count} out of range "
                f"[{scenario_spec.get('anomaly_count_min')}, "
                f"{scenario_spec.get('anomaly_count_max')}]"
            )

    # ── Semantic tags ───────────────────────────────────────────────────────
    for tag in scenario_spec.get("expected_semantic_tags", []):
        tags = result_payload.get("semantic_tags", [])
        found = tag in tags
        checks.append(
            SemanticCheck(
                field="semantic_tags",
                expected=tag,
                actual=tags,
                passed=found,
                tolerance=None,
                note="" if found else "tag not found",
            )
        )
        if not found:
            warnings.append(f"semantic tag {tag!r} not found (may be optional)")

    all_passed = len(errors) == 0
    return EvaluationOutcome(
        scenario=scenario_name,
        passed=all_passed,
        checks=checks,
        errors=errors,
        warnings=warnings,
    )
