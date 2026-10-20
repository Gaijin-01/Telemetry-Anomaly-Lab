"""validate_comparison skill: check that two DetectionResult objects are
safe to compare — same dataset semantics, compatible metadata, no null scores.

No LLM. No autonomous loops.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from anomaly_lab.results import DetectionResult


@dataclass(frozen=True, kw_only=True)
class ValidationIssue:
    field: str
    code: str
    message: str


@dataclass(frozen=True, kw_only=True)
class ComparisonValidationResult:
    is_valid: bool
    issues: tuple[ValidationIssue, ...]
    warnings: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "is_valid": self.is_valid,
            "issues": [
                {"field": i.field, "code": i.code, "message": i.message} for i in self.issues
            ],
            "warnings": list(self.warnings),
        }


def validate_comparison(
    baseline: DetectionResult,
    candidate: DetectionResult,
) -> ComparisonValidationResult:
    """Validate that baseline and candidate can be compared.

    Checks
    ------
    - Both have the same detector name
    - Both have non-null scores arrays
    - Dataset semantics match (or both are None)
    - Same number of rows
    - Same feature names
    - No NaN/inf in scores
    """
    issues: list[ValidationIssue] = []
    warnings: list[str] = []

    # Detector must match
    if baseline.detector != candidate.detector:
        issues.append(
            ValidationIssue(
                field="detector",
                code="DETECTOR_MISMATCH",
                message=(
                    f"baseline detector={baseline.detector!r} "
                    f"!= candidate detector={candidate.detector!r}"
                ),
            )
        )

    # Scores must be present
    if baseline.scores is None:
        issues.append(
            ValidationIssue(
                field="scores",
                code="BASELINE_SCORES_NULL",
                message="baseline scores are None",
            )
        )
    if candidate.scores is None:
        issues.append(
            ValidationIssue(
                field="scores",
                code="CANDIDATE_SCORES_NULL",
                message="candidate scores are None",
            )
        )

    # Row count must match
    if len(baseline.row_indices) != len(candidate.row_indices):
        issues.append(
            ValidationIssue(
                field="row_indices",
                code="ROW_COUNT_MISMATCH",
                message=(
                    f"baseline rows={len(baseline.row_indices)} "
                    f"!= candidate rows={len(candidate.row_indices)}"
                ),
            )
        )

    # Feature names must match
    if baseline.features != candidate.features:
        issues.append(
            ValidationIssue(
                field="features",
                code="FEATURE_MISMATCH",
                message=(
                    f"baseline features={baseline.features!r} "
                    f"!= candidate features={candidate.features!r}"
                ),
            )
        )

    # Dataset semantics
    if baseline.dataset_semantics != candidate.dataset_semantics:
        if baseline.dataset_semantics is None or candidate.dataset_semantics is None:
            warnings.append("one_side_dataset_semantics_missing")
        else:
            issues.append(
                ValidationIssue(
                    field="dataset_semantics",
                    code="SEMANTICS_MISMATCH",
                    message=(
                    f"baseline semantics={baseline.dataset_semantics!r} "
                    f"!= candidate semantics={candidate.dataset_semantics!r}"
                ),
                )
            )

    # Score quality: no NaN/inf
    if baseline.scores is not None:
        for i, s in enumerate(baseline.scores):
            if s is not None and not isinstance(s, (int, float)):
                issues.append(
                    ValidationIssue(
                        field="scores",
                        code="BASELINE_SCORE_NON_NUMERIC",
                        message=f"baseline score[{i}] is {type(s).__name__}",
                    )
                )
                break
            elif s is not None:
                import math

                if math.isnan(s) or math.isinf(s):
                    issues.append(
                        ValidationIssue(
                            field="scores",
                            code="BASELINE_SCORE_NOT_FINITE",
                            message=f"baseline score[{i}] is not finite: {s}",
                        )
                    )
                    break

    if candidate.scores is not None:
        import math

        for i, s in enumerate(candidate.scores):
            if s is not None and not isinstance(s, (int, float)):
                issues.append(
                    ValidationIssue(
                        field="scores",
                        code="CANDIDATE_SCORE_NON_NUMERIC",
                        message=f"candidate score[{i}] is {type(s).__name__}",
                    )
                )
                break
            elif s is not None:
                if math.isnan(s) or math.isinf(s):
                    issues.append(
                        ValidationIssue(
                            field="scores",
                            code="CANDIDATE_SCORE_NOT_FINITE",
                            message=f"candidate score[{i}] is not finite: {s}",
                        )
                    )
                    break

    return ComparisonValidationResult(
        is_valid=len(issues) == 0,
        issues=tuple(issues),
        warnings=tuple(warnings),
    )
