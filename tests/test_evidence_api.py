"""Tests for the evidence API: taxonomy, applicability, comparability, and verdicts."""

from __future__ import annotations

from anomaly_lab.evidence import (
    CausalityStatus,
    ComparisonApplicability,
    DetectionApplicability,
    EvidenceApplicability,
    EvidenceClass,
    EvidenceRecord,
    EvidenceVerdict,
    VerdictDecision,
    VerdictReason,
    check_comparability,
    compute_comparison_verdict,
    compute_detection_verdict,
    get_applicability,
    is_comparable,
)
from anomaly_lab.evidence.taxonomy import (
    DEFAULT_APPLICABILITY_RULES,
)
from anomaly_lab.results import DetectionResult

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_detection_result(
    detector: str = "isolation-forest",
    is_anomaly: tuple[bool, ...] = (True, False, False),
    scores: tuple[float | None, ...] | None = None,
    clusters: tuple[int, ...] | None = None,
    parameters: dict | None = None,
    failed: bool = False,
) -> DetectionResult:
    """Build a minimal DetectionResult for testing."""
    n = len(is_anomaly)
    if scores is None:
        scores = (0.9, 0.1, 0.2)[:n]
    return DetectionResult(
        detector=detector,
        row_indices=tuple(range(n)),
        is_anomaly=is_anomaly,
        features=("x",),
        constant_features=(),
        parameters=parameters or {"contamination": 0.1},
        scores=scores,
        clusters=clusters,
        metadata={"failed": failed} if failed else None,
    )


def make_evidence_record(
    evidence_class: str = EvidenceClass.POINT_ANOMALY.value,
    confidence: float | None = 0.8,
    detector: str = "isolation-forest",
    failed: bool = False,
    causality_status: CausalityStatus = CausalityStatus.NOT_ESTABLISHED,
    **extra_fields,
) -> EvidenceRecord:
    """Build a minimal EvidenceRecord for testing."""
    kwargs = dict(
        evidence_class=evidence_class,
        confidence=confidence,
        detector=detector,
        score=0.9,
        threshold=0.5,
        supporting_metrics={"anomaly_count": 1.0},
        assumptions=("iid_data",),
        limitations=("no_ground_truth",),
        interpretation="Test evidence.",
    )
    # Add extended fields if provided
    if "causality_status" not in kwargs:
        kwargs["causality_status"] = causality_status
    if "applicability" not in kwargs:
        kwargs["applicability"] = DetectionApplicability(
            applicable=True,
            decision=VerdictDecision.ANOMALY_DETECTION,
            domain_constraints=(),
        )
    if "verdict" not in kwargs:
        kwargs["verdict"] = None
    kwargs.update(extra_fields)
    return EvidenceRecord(**kwargs)


# ---------------------------------------------------------------------------
# Taxonomy: EvidenceClass enum
# ---------------------------------------------------------------------------


class TestEvidenceClassValues:
    """EvidenceClass enum contains all required taxonomy labels."""

    def test_has_point_anomaly(self):
        assert EvidenceClass.POINT_ANOMALY.value == "POINT_ANOMALY"

    def test_has_local_outlier(self):
        assert EvidenceClass.LOCAL_OUTLIER.value == "LOCAL_OUTLIER"

    def test_has_global_outlier(self):
        assert EvidenceClass.GLOBAL_OUTLIER.value == "GLOBAL_OUTLIER"

    def test_has_density_noise(self):
        assert EvidenceClass.DENSITY_NOISE.value == "DENSITY_NOISE"

    def test_has_regime_shift(self):
        assert EvidenceClass.REGIME_SHIFT.value == "REGIME_SHIFT"

    def test_has_distribution_shift(self):
        assert EvidenceClass.DISTRIBUTION_SHIFT.value == "DISTRIBUTION_SHIFT"

    def test_has_variance_collapse(self):
        assert EvidenceClass.VARIANCE_COLLAPSE.value == "VARIANCE_COLLAPSE"

    def test_has_comparison_difference(self):
        assert EvidenceClass.COMPARISON_DIFFERENCE.value == "COMPARISON_DIFFERENCE"

    def test_has_comparison_equivalent(self):
        assert EvidenceClass.COMPARISON_EQUIVALENT.value == "COMPARISON_EQUIVALENT"

    def test_has_inconclusive(self):
        assert EvidenceClass.INCONCLUSIVE.value == "INCONCLUSIVE"

    def test_has_not_applicable(self):
        assert EvidenceClass.NOT_APPLICABLE.value == "NOT_APPLICABLE"

    # v0.5 additions
    def test_has_improvement(self):
        assert EvidenceClass.IMPROVEMENT.value == "IMPROVEMENT"

    def test_has_regression_evidence_class(self):
        assert EvidenceClass.REGRESSION.value == "REGRESSION"

    def test_has_neutral(self):
        assert EvidenceClass.NEUTRAL.value == "NEUTRAL"

    def test_has_insufficient_evidence(self):
        assert EvidenceClass.INSUFFICIENT_EVIDENCE.value == "INSUFFICIENT_EVIDENCE"

    def test_has_incomparable(self):
        assert EvidenceClass.INCOMPARABLE.value == "INCOMPARABLE"


# ---------------------------------------------------------------------------
# Taxonomy: CausalityStatus enum
# ---------------------------------------------------------------------------


class TestCausalityStatus:
    """CausalityStatus defaults to NOT_ESTABLISHED unless explicitly set."""

    def test_not_established_exists(self):
        assert CausalityStatus.NOT_ESTABLISHED.value == "NOT_ESTABLISHED"

    def test_established_exists(self):
        assert CausalityStatus.ESTABLISHED.value == "ESTABLISHED"

    def test_indeterminate_exists(self):
        assert CausalityStatus.INDETERMINATE.value == "INDETERMINATE"

    def test_default_is_not_established(self):
        """New EvidenceRecord instances default to NOT_ESTABLISHED."""
        rec = make_evidence_record()
        assert rec.causality_status == CausalityStatus.NOT_ESTABLISHED

    def test_established_explicitly_set(self):
        rec = make_evidence_record(causality_status=CausalityStatus.ESTABLISHED)
        assert rec.causality_status == CausalityStatus.ESTABLISHED


# ---------------------------------------------------------------------------
# Taxonomy: EvidenceVerdict enum
# ---------------------------------------------------------------------------


class TestEvidenceVerdict:
    """EvidenceVerdict values for pass/fail/inconclusive."""

    def test_pass_exists(self):
        assert EvidenceVerdict.PASS.value == "PASS"

    def test_fail_exists(self):
        assert EvidenceVerdict.FAIL.value == "FAIL"

    def test_inconclusive_exists(self):
        assert EvidenceVerdict.INCONCLUSIVE.value == "INCONCLUSIVE"

    def test_cannot_pass_exists(self):
        assert EvidenceVerdict.CANNOT_PASS.value == "CANNOT_PASS"


# ---------------------------------------------------------------------------
# Taxonomy: EvidenceRecord extended fields
# ---------------------------------------------------------------------------


class TestEvidenceRecordExtendedFields:
    """EvidenceRecord carries causality_status, applicability, and verdict."""

    def test_evidence_record_has_causality_status_field(self):
        rec = make_evidence_record()
        assert hasattr(rec, "causality_status")
        assert rec.causality_status == CausalityStatus.NOT_ESTABLISHED

    def test_evidence_record_has_applicability_field(self):
        rec = make_evidence_record()
        assert hasattr(rec, "applicability")
        assert isinstance(rec.applicability, EvidenceApplicability)

    def test_evidence_record_has_verdict_field(self):
        rec = make_evidence_record(verdict=None)
        assert hasattr(rec, "verdict")
        assert rec.verdict is None

    def test_evidence_record_to_dict_includes_new_fields(self):
        rec = make_evidence_record(
            causality_status=CausalityStatus.ESTABLISHED,
            applicability=DetectionApplicability(
                applicable=True,
                decision=VerdictDecision.ANOMALY_DETECTION,
                domain_constraints=(),
            ),
            verdict=None,
        )
        d = rec.to_dict()
        assert "causality_status" in d
        assert "applicability" in d
        assert "verdict" in d

    def test_backward_compatible_without_new_fields(self):
        """Old EvidenceRecord construction still works."""
        rec = EvidenceRecord(
            evidence_class=EvidenceClass.POINT_ANOMALY.value,
            confidence=0.8,
            detector="isolation-forest",
            score=0.9,
            threshold=0.5,
            supporting_metrics={},
            assumptions=(),
            limitations=(),
            interpretation="test",
        )
        assert rec.evidence_class == EvidenceClass.POINT_ANOMALY.value
        # New fields have defaults
        assert rec.causality_status == CausalityStatus.NOT_ESTABLISHED


# ---------------------------------------------------------------------------
# Applicability
# ---------------------------------------------------------------------------


class TestDetectionApplicability:
    """DetectionApplicability carries detector-specific applicability info."""

    def test_fields(self):
        app = DetectionApplicability(
            applicable=True,
            decision=VerdictDecision.ANOMALY_DETECTION,
            domain_constraints=("low_sample_size",),
        )
        assert app.applicable is True
        assert app.decision == VerdictDecision.ANOMALY_DETECTION
        assert app.domain_constraints == ("low_sample_size",)

    def test_to_dict(self):
        app = DetectionApplicability(
            applicable=True,
            decision=VerdictDecision.ANOMALY_DETECTION,
            domain_constraints=(),
        )
        d = app.to_dict()
        assert d["applicable"] is True
        assert d["decision"] == "ANOMALY_DETECTION"
        assert d["domain_constraints"] == []


class TestComparisonApplicability:
    """ComparisonApplicability carries comparison-specific applicability info."""

    def test_fields(self):
        app = ComparisonApplicability(
            applicable=True,
            decision=VerdictDecision.REGRESSION_TEST,
            domain_constraints=("unequal_variance",),
            requires_paired_structure=True,
        )
        assert app.applicable is True
        assert app.decision == VerdictDecision.REGRESSION_TEST
        assert app.domain_constraints == ("unequal_variance",)
        assert app.requires_paired_structure is True

    def test_to_dict(self):
        app = ComparisonApplicability(
            applicable=False,
            decision=VerdictDecision.REGRESSION_TEST,
            domain_constraints=(),
            requires_paired_structure=False,
        )
        d = app.to_dict()
        assert d["applicable"] is False
        assert d["requires_paired_structure"] is False


class TestGetApplicability:
    """get_applicability returns correct EvidenceApplicability per evidence_class."""

    def test_point_anomaly_applicable_to_anomaly_detection(self):
        rec = make_evidence_record(
            evidence_class=EvidenceClass.POINT_ANOMALY.value,
            applicability=DetectionApplicability(
                applicable=True,
                decision=VerdictDecision.ANOMALY_DETECTION,
                domain_constraints=(),
            ),
        )
        app = get_applicability(rec)
        assert app.applicable is True
        assert app.decision == VerdictDecision.ANOMALY_DETECTION

    def test_inconclusive_not_applicable(self):
        rec = make_evidence_record(
            evidence_class=EvidenceClass.INCONCLUSIVE.value,
            confidence=0.0,
            applicability=DetectionApplicability(
                applicable=False,
                decision=VerdictDecision.ANOMALY_DETECTION,
                domain_constraints=("insufficient_evidence",),
            ),
        )
        app = get_applicability(rec)
        assert app.applicable is False

    def test_not_applicable_class(self):
        rec = make_evidence_record(
            evidence_class=EvidenceClass.NOT_APPLICABLE.value,
            applicability=DetectionApplicability(
                applicable=False,
                decision=VerdictDecision.ANOMALY_DETECTION,
                domain_constraints=(),
            ),
        )
        app = get_applicability(rec)
        assert app.applicable is False

    def test_improvement_applicable_to_regression_test(self):
        rec = make_evidence_record(
            evidence_class=EvidenceClass.IMPROVEMENT.value,
            applicability=ComparisonApplicability(
                applicable=True,
                decision=VerdictDecision.REGRESSION_TEST,
                domain_constraints=(),
            ),
        )
        app = get_applicability(rec)
        assert app.applicable is True
        assert app.decision == VerdictDecision.REGRESSION_TEST

    def test_regression_class_applicable_to_regression_test(self):
        rec = make_evidence_record(
            evidence_class=EvidenceClass.REGRESSION.value,
            applicability=ComparisonApplicability(
                applicable=True,
                decision=VerdictDecision.REGRESSION_TEST,
                domain_constraints=(),
            ),
        )
        app = get_applicability(rec)
        assert app.applicable is True
        assert app.decision == VerdictDecision.REGRESSION_TEST


# ---------------------------------------------------------------------------
# Comparability
# ---------------------------------------------------------------------------


class TestCheckComparability:
    """check_comparability returns a dict describing comparability between records."""

    def test_same_detector_comparable(self):
        rec1 = make_evidence_record(detector="isolation-forest")
        rec2 = make_evidence_record(detector="isolation-forest")
        result = check_comparability(rec1, rec2)
        assert result.comparable is True
        assert result.same_detector is True
        assert result.same_semantics is True
        assert result.reason == ""

    def test_different_detectors_same_family_comparable(self):
        # Both statistical family
        rec1 = make_evidence_record(detector="robust-zscore")
        rec2 = make_evidence_record(detector="isolation-forest")
        result = check_comparability(rec1, rec2)
        assert result.comparable is True
        assert result.same_detector is False
        assert "Different detectors" in result.reason

    def test_incomparable_different_semantics(self):
        rec1 = make_evidence_record(
            detector="isolation-forest",
            applicability=DetectionApplicability(
                applicable=True,
                decision=VerdictDecision.ANOMALY_DETECTION,
                domain_constraints=(),
            ),
        )
        rec2 = make_evidence_record(
            detector="isolation-forest",
            applicability=DetectionApplicability(
                applicable=False,
                decision=VerdictDecision.REGRESSION_TEST,
                domain_constraints=("different_semantics",),
            ),
        )
        result = check_comparability(rec1, rec2)
        assert result.comparable is False
        assert result.reason != ""

    def test_incomparable_class(self):
        rec1 = make_evidence_record(evidence_class=EvidenceClass.INCOMPARABLE.value)
        rec2 = make_evidence_record(evidence_class=EvidenceClass.POINT_ANOMALY.value)
        result = check_comparability(rec1, rec2)
        assert result.comparable is False


class TestIsComparable:
    """is_comparable is a convenience wrapper."""

    def test_same_detector_is_comparable(self):
        rec1 = make_evidence_record(detector="isolation-forest")
        rec2 = make_evidence_record(detector="isolation-forest")
        assert is_comparable(rec1, rec2) is True

    def test_incomparable_returns_false(self):
        rec1 = make_evidence_record(evidence_class=EvidenceClass.INCOMPARABLE.value)
        rec2 = make_evidence_record(evidence_class=EvidenceClass.POINT_ANOMALY.value)
        assert is_comparable(rec1, rec2) is False


# ---------------------------------------------------------------------------
# Verdicts: FAILED_EXECUTION_CANNOT_PASS invariant
# ---------------------------------------------------------------------------


class TestFailedExecutionCannotPass:
    """FAILED_EXECUTION results always yield CANNOT_PASS verdict regardless of evidence."""

    def test_failed_detection_yields_cannot_pass(self):
        # Build an EvidenceRecord that represents a failed detection execution
        rec = make_evidence_record(
            evidence_class=EvidenceClass.INCONCLUSIVE.value,
            confidence=0.0,
            limitations=("execution_failed",),
            supporting_metrics={"execution_failed": True},
        )
        verdict = compute_detection_verdict(rec)
        assert verdict.verdict == EvidenceVerdict.CANNOT_PASS
        assert verdict.reason_code in (
            VerdictReason.FAILED_EXECUTION,
            VerdictReason.EXECUTION_FAILED,
        )

    def test_normal_detection_passes_point_anomaly(self):
        rec = make_evidence_record(
            evidence_class=EvidenceClass.POINT_ANOMALY.value,
            confidence=0.8,
            supporting_metrics={"anomaly_count": 1.0},
        )
        verdict = compute_detection_verdict(rec)
        assert verdict.verdict == EvidenceVerdict.PASS

    def test_no_anomaly_detection_inconclusive(self):
        rec = make_evidence_record(
            evidence_class=EvidenceClass.INCONCLUSIVE.value,
            confidence=0.0,
            supporting_metrics={},
        )
        verdict = compute_detection_verdict(rec)
        assert verdict.verdict == EvidenceVerdict.INCONCLUSIVE


class TestComputeComparisonVerdict:
    """compute_comparison_verdict derives verdict from comparison EvidenceRecord."""

    def test_improvement_yields_pass(self):
        rec = make_evidence_record(
            evidence_class=EvidenceClass.IMPROVEMENT.value,
            confidence=0.85,
            applicability=ComparisonApplicability(
                applicable=True,
                decision=VerdictDecision.REGRESSION_TEST,
                domain_constraints=(),
            ),
            causality_status=CausalityStatus.NOT_ESTABLISHED,
        )
        verdict = compute_comparison_verdict(rec)
        assert verdict.verdict == EvidenceVerdict.PASS

    def test_regression_yields_fail(self):
        rec = make_evidence_record(
            evidence_class=EvidenceClass.REGRESSION.value,
            confidence=0.9,
            applicability=ComparisonApplicability(
                applicable=True,
                decision=VerdictDecision.REGRESSION_TEST,
                domain_constraints=(),
            ),
            causality_status=CausalityStatus.NOT_ESTABLISHED,
        )
        verdict = compute_comparison_verdict(rec)
        assert verdict.verdict == EvidenceVerdict.FAIL

    def test_neutral_yields_inconclusive(self):
        rec = make_evidence_record(
            evidence_class=EvidenceClass.NEUTRAL.value,
            confidence=0.5,
            applicability=ComparisonApplicability(
                applicable=True,
                decision=VerdictDecision.REGRESSION_TEST,
                domain_constraints=(),
            ),
            causality_status=CausalityStatus.NOT_ESTABLISHED,
        )
        verdict = compute_comparison_verdict(rec)
        assert verdict.verdict == EvidenceVerdict.INCONCLUSIVE

    def test_insufficient_evidence_yields_cannot_pass(self):
        rec = make_evidence_record(
            evidence_class=EvidenceClass.INSUFFICIENT_EVIDENCE.value,
            confidence=0.0,
            applicability=ComparisonApplicability(
                applicable=False,
                decision=VerdictDecision.REGRESSION_TEST,
                domain_constraints=("insufficient_samples",),
            ),
            causality_status=CausalityStatus.NOT_ESTABLISHED,
        )
        verdict = compute_comparison_verdict(rec)
        assert verdict.verdict == EvidenceVerdict.CANNOT_PASS

    def test_comparison_incomparable_yields_cannot_pass(self):
        rec = make_evidence_record(
            evidence_class=EvidenceClass.INCOMPARABLE.value,
            confidence=None,
            applicability=ComparisonApplicability(
                applicable=False,
                decision=VerdictDecision.REGRESSION_TEST,
                domain_constraints=("metadata_mismatch",),
            ),
            causality_status=CausalityStatus.NOT_ESTABLISHED,
        )
        verdict = compute_comparison_verdict(rec)
        assert verdict.verdict == EvidenceVerdict.CANNOT_PASS

    def test_not_applicable_yields_cannot_pass(self):
        rec = make_evidence_record(
            evidence_class=EvidenceClass.NOT_APPLICABLE.value,
            confidence=None,
            applicability=ComparisonApplicability(
                applicable=False,
                decision=VerdictDecision.REGRESSION_TEST,
                domain_constraints=(),
            ),
            causality_status=CausalityStatus.NOT_ESTABLISHED,
        )
        verdict = compute_comparison_verdict(rec)
        assert verdict.verdict == EvidenceVerdict.CANNOT_PASS


# ---------------------------------------------------------------------------
# Default applicability rules
# ---------------------------------------------------------------------------


class TestDefaultApplicabilityRules:
    """DEFAULT_APPLICABILITY_RULES is a non-empty TOML-serialisable dict."""

    def test_is_dict(self):
        assert isinstance(DEFAULT_APPLICABILITY_RULES, dict)

    def test_has_version_key(self):
        assert "version" in DEFAULT_APPLICABILITY_RULES

    def test_version_is_semver_like(self):
        v = DEFAULT_APPLICABILITY_RULES["version"]
        assert isinstance(v, str)
        parts = v.split(".")
        assert len(parts) == 3
        assert all(p.isdigit() for p in parts)


# ---------------------------------------------------------------------------
# Integration: classify_detection_evidence produces extended EvidenceRecord
# ---------------------------------------------------------------------------


class TestClassifyDetectionEvidenceExtended:
    """classify_detection_evidence (existing function) returns EvidenceRecord
    with new fields filled by the verdict/applicability pipeline."""

    def test_produces_evidence_record_with_applicability(self):
        from anomaly_lab.evidence import classify_detection_evidence

        result = make_detection_result(
            detector="isolation-forest",
            is_anomaly=(True, False),
            scores=(0.9, 0.1),
        )
        rec = classify_detection_evidence(result)
        assert isinstance(rec, EvidenceRecord)
        assert hasattr(rec, "causality_status")
        assert hasattr(rec, "applicability")
        assert isinstance(rec.applicability, EvidenceApplicability)

    def test_produces_evidence_record_with_verdict(self):
        from anomaly_lab.evidence import classify_detection_evidence

        result = make_detection_result(
            detector="isolation-forest",
            is_anomaly=(True, False),
            scores=(0.9, 0.1),
        )
        rec = classify_detection_evidence(result)
        assert hasattr(rec, "verdict")
        assert rec.verdict is not None
        assert isinstance(rec.verdict.verdict, EvidenceVerdict)

    def test_dbscan_noise_has_noise_evidence_class(self):
        from anomaly_lab.evidence import classify_detection_evidence

        result = make_detection_result(
            detector="dbscan",
            is_anomaly=(False, False, False),
            clusters=(-1, 0, 0),
            scores=(0.3, 0.5, 0.6),
        )
        rec = classify_detection_evidence(result)
        assert rec.evidence_class == EvidenceClass.DENSITY_NOISE.value


# ---------------------------------------------------------------------------
# CausalityStatus is NOT_ESTABLISHED by default in EvidenceRecord
# ---------------------------------------------------------------------------


class TestCausalityStatusDefault:
    """Every EvidenceRecord defaults causality_status to NOT_ESTABLISHED."""

    def test_classify_detection_evidence_defaults_to_not_established(self):
        from anomaly_lab.evidence import classify_detection_evidence

        result = make_detection_result(
            detector="robust-zscore",
            is_anomaly=(True, False),
            scores=(4.0, 0.1),
        )
        rec = classify_detection_evidence(result)
        assert rec.causality_status == CausalityStatus.NOT_ESTABLISHED
