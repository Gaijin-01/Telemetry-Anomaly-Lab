"""Tests for skills boundary enforcement.

SKILL_OUTPUT_CANNOT_OVERRIDE_CORE_EVIDENCE invariant:
  Skills may reorganise facts but never alter verdict/evidence_class.

These tests verify that each skill respects the boundary:
- inspect_data: does not alter evidence
- select_analysis_plan: does not alter evidence
- validate_comparison: does not alter evidence
- run_experiment: does not alter evidence
- summarize_evidence: does not alter individual EvidenceRecord evidence_class
- export_report: does not alter evidence
"""

import numpy as np
import pandas as pd
import pytest

from anomaly_lab.evidence.taxonomy import EvidenceClass, EvidenceRecord
from anomaly_lab.orchestration.guardrails import check_skill_output
from anomaly_lab.skills.export_report import ExportedReport, ReportMetadata, export_report
from anomaly_lab.skills.inspect_data import DataInspectionResult, inspect_data
from anomaly_lab.skills.run_experiment import ExperimentRunResult, run_experiment
from anomaly_lab.skills.select_analysis_plan import AnalysisPlan, select_analysis_plan
from anomaly_lab.skills.summarize_evidence import EvidenceSummary, summarize_evidence
from anomaly_lab.skills.validate_comparison import ComparisonValidationResult, validate_comparison

# -----------------------------------------------------------------------
# Fixtures
# -----------------------------------------------------------------------


@pytest.fixture
def sample_df() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    return pd.DataFrame(
        {
            "value": rng.normal(size=100),
            "id": range(100),
        }
    )


@pytest.fixture
def sample_records() -> list[EvidenceRecord]:
    return [
        EvidenceRecord(
            evidence_class=EvidenceClass.POINT_ANOMALY.value,
            confidence=0.9,
            detector="isolation-forest",
            score=0.85,
            threshold=0.5,
            supporting_metrics={"anomaly_count": 3, "total_points": 100},
            assumptions=("contamination_reasonable",),
            limitations=("not_root_cause",),
            interpretation="Point anomaly detected.",
        ),
        EvidenceRecord(
            evidence_class=EvidenceClass.LOCAL_OUTLIER.value,
            confidence=0.7,
            detector="robust-zscore",
            score=4.1,
            threshold=3.5,
            supporting_metrics={"anomaly_count": 2, "total_points": 100},
            assumptions=("symmetric_heavy_tail",),
            limitations=("not_root_cause",),
            interpretation="Local outlier detected.",
        ),
    ]


# -----------------------------------------------------------------------
# inspect_data
# -----------------------------------------------------------------------


class TestInspectDataBoundary:
    def test_returns_datainspectionresult(self, sample_df):
        result = inspect_data(sample_df)
        assert isinstance(result, DataInspectionResult)

    def test_result_is_frozen(self, sample_df):
        result = inspect_data(sample_df)
        with pytest.raises(AttributeError):
            result.row_count = 999  # type: ignore[misc]

    def test_does_not_touch_evidence(self, sample_df):
        # inspect_data has no access to EvidenceRecord
        result = inspect_data(sample_df)
        assert not hasattr(result, "evidence_class")
        assert not hasattr(result, "verdict")

    def test_empty_df_handled(self):
        result = inspect_data(pd.DataFrame())
        assert result.is_empty is True
        assert result.warnings == ("empty_dataframe",)

    def test_constant_column_detected(self):
        df = pd.DataFrame({"constant": [1.0] * 50, "varying": np.random.randn(50)})
        result = inspect_data(df)
        assert "constant" in result.constant_columns


# -----------------------------------------------------------------------
# select_analysis_plan
# -----------------------------------------------------------------------


class TestSelectAnalysisPlanBoundary:
    def test_returns_analysisplan(self):
        inspection = DataInspectionResult(
            row_count=200,
            column_count=2,
            numeric_columns=("a",),
            non_numeric_columns=(),
            null_fraction={},
            constant_columns=(),
            duplicate_row_count=0,
            finite_violations={},
            warnings=(),
            is_empty=False,
        )
        plan = select_analysis_plan(inspection, "point_anomaly")
        assert isinstance(plan, AnalysisPlan)

    def test_plan_is_frozen(self):
        inspection = DataInspectionResult(
            row_count=200,
            column_count=2,
            numeric_columns=("a",),
            non_numeric_columns=(),
            null_fraction={},
            constant_columns=(),
            duplicate_row_count=0,
            finite_violations={},
            warnings=(),
            is_empty=False,
        )
        plan = select_analysis_plan(inspection, "exploratory")
        with pytest.raises(AttributeError):
            plan.intent = "something_else"  # type: ignore[misc]

    def test_small_sample_restricts_detectors(self):
        inspection = DataInspectionResult(
            row_count=8,
            column_count=1,
            numeric_columns=("a",),
            non_numeric_columns=(),
            null_fraction={},
            constant_columns=(),
            duplicate_row_count=0,
            finite_violations={},
            warnings=(),
            is_empty=False,
        )
        plan = select_analysis_plan(inspection, "point_anomaly")
        assert "isolation-forest" not in plan.detectors
        assert "robust-zscore" in plan.detectors

    def test_comparison_intent_enables_spearman(self):
        inspection = DataInspectionResult(
            row_count=200,
            column_count=1,
            numeric_columns=("a",),
            non_numeric_columns=(),
            null_fraction={},
            constant_columns=(),
            duplicate_row_count=0,
            finite_violations={},
            warnings=(),
            is_empty=False,
        )
        plan = select_analysis_plan(inspection, "comparison")
        assert plan.correlation_method == "spearman"

    def test_large_sample_enables_regime_detection(self):
        inspection = DataInspectionResult(
            row_count=600,
            column_count=1,
            numeric_columns=("a",),
            non_numeric_columns=(),
            null_fraction={},
            constant_columns=(),
            duplicate_row_count=0,
            finite_violations={},
            warnings=(),
            is_empty=False,
        )
        plan = select_analysis_plan(inspection, "point_anomaly")
        assert plan.regime_detection is True


# -----------------------------------------------------------------------
# validate_comparison
# -----------------------------------------------------------------------


class TestValidateComparisonBoundary:
    def test_returns_validation_result(self):
        from anomaly_lab.results import DetectionResult

        dr1 = DetectionResult(
            detector="robust-zscore",
            row_indices=tuple(range(10)),
            is_anomaly=tuple(False for _ in range(10)),
            features=("x",),
            constant_features=(),
            parameters={},
            scores=tuple([0.1] * 10),
        )
        dr2 = DetectionResult(
            detector="robust-zscore",
            row_indices=tuple(range(10)),
            is_anomaly=tuple(False for _ in range(10)),
            features=("x",),
            constant_features=(),
            parameters={},
            scores=tuple([0.2] * 10),
        )
        result = validate_comparison(dr1, dr2)
        assert isinstance(result, ComparisonValidationResult)

    def test_detector_mismatch_flagged(self):
        from anomaly_lab.results import DetectionResult

        dr1 = DetectionResult(
            detector="robust-zscore",
            row_indices=tuple(range(10)),
            is_anomaly=tuple(False for _ in range(10)),
            features=("x",),
            constant_features=(),
            parameters={},
            scores=tuple([0.1] * 10),
        )
        dr2 = DetectionResult(
            detector="isolation-forest",
            row_indices=tuple(range(10)),
            is_anomaly=tuple(False for _ in range(10)),
            features=("x",),
            constant_features=(),
            parameters={},
            scores=tuple([0.2] * 10),
        )
        result = validate_comparison(dr1, dr2)
        assert result.is_valid is False
        codes = [i.code for i in result.issues]
        assert "DETECTOR_MISMATCH" in codes

    def test_row_count_mismatch_flagged(self):
        from anomaly_lab.results import DetectionResult

        dr1 = DetectionResult(
            detector="robust-zscore",
            row_indices=tuple(range(10)),
            is_anomaly=tuple(False for _ in range(10)),
            features=("x",),
            constant_features=(),
            parameters={},
            scores=tuple([0.1] * 10),
        )
        dr2 = DetectionResult(
            detector="robust-zscore",
            row_indices=tuple(range(20)),
            is_anomaly=tuple(False for _ in range(20)),
            features=("x",),
            constant_features=(),
            parameters={},
            scores=tuple([0.2] * 20),
        )
        result = validate_comparison(dr1, dr2)
        assert result.is_valid is False
        codes = [i.code for i in result.issues]
        assert "ROW_COUNT_MISMATCH" in codes

    def test_valid_comparison_passes(self):
        from anomaly_lab.results import DetectionResult

        dr1 = DetectionResult(
            detector="robust-zscore",
            row_indices=tuple(range(10)),
            is_anomaly=tuple(False for _ in range(10)),
            features=("x",),
            constant_features=(),
            parameters={},
            scores=tuple([0.1] * 10),
        )
        dr2 = DetectionResult(
            detector="robust-zscore",
            row_indices=tuple(range(10)),
            is_anomaly=tuple(False for _ in range(10)),
            features=("x",),
            constant_features=(),
            parameters={},
            scores=tuple([0.2] * 10),
        )
        result = validate_comparison(dr1, dr2)
        assert result.is_valid is True
        assert result.issues == ()

    def test_result_is_frozen(self):
        from anomaly_lab.results import DetectionResult

        dr1 = DetectionResult(
            detector="robust-zscore",
            row_indices=tuple(range(10)),
            is_anomaly=tuple(False for _ in range(10)),
            features=("x",),
            constant_features=(),
            parameters={},
            scores=tuple([0.1] * 10),
        )
        dr2 = DetectionResult(
            detector="robust-zscore",
            row_indices=tuple(range(10)),
            is_anomaly=tuple(False for _ in range(10)),
            features=("x",),
            constant_features=(),
            parameters={},
            scores=tuple([0.2] * 10),
        )
        result = validate_comparison(dr1, dr2)
        with pytest.raises(AttributeError):
            result.is_valid = False  # type: ignore[misc]


# -----------------------------------------------------------------------
# run_experiment
# -----------------------------------------------------------------------


class TestRunExperimentBoundary:
    def test_returns_experimentrunresult(self, sample_df):
        result = run_experiment("robust-zscore", "value", sample_df)
        assert isinstance(result, ExperimentRunResult)

    def test_result_is_frozen(self, sample_df):
        result = run_experiment("robust-zscore", "value", sample_df)
        with pytest.raises(AttributeError):
            result.detector = "something_else"  # type: ignore[misc]

    def test_detection_result_is_present_on_success(self, sample_df):
        result = run_experiment("robust-zscore", "value", sample_df)
        assert result.error_message is None
        assert result.detection is not None

    def test_unknown_detector_raises(self, sample_df):
        with pytest.raises(ValueError, match="Unknown detector"):
            run_experiment("nonexistent-detector", "value", sample_df)

    def test_isolation_forest_runs(self, sample_df):
        result = run_experiment("isolation-forest", "value", sample_df)
        assert result.error_message is None
        assert result.detection is not None
        assert result.detection.detector == "isolation-forest"

    def test_dbscan_runs(self, sample_df):
        result = run_experiment("dbscan", "value", sample_df)
        assert result.error_message is None
        assert result.detection is not None
        assert result.detection.detector == "dbscan"

    def test_evidence_class_not_mutated_by_skill(self, sample_df, sample_records):
        # run_experiment doesn't touch EvidenceRecord
        result = run_experiment("robust-zscore", "value", sample_df)
        assert not hasattr(result, "evidence_class")


# -----------------------------------------------------------------------
# summarize_evidence
# -----------------------------------------------------------------------


class TestSummarizeEvidenceBoundary:
    def test_returns_evidencesummary(self, sample_records):
        summary = summarize_evidence(sample_records)
        assert isinstance(summary, EvidenceSummary)

    def test_summary_is_frozen(self, sample_records):
        summary = summarize_evidence(sample_records)
        with pytest.raises(AttributeError):
            summary.total_records = 999  # type: ignore[misc]

    def test_individual_records_not_mutated(self, sample_records):
        original_classes = [r.evidence_class for r in sample_records]
        _ = summarize_evidence(sample_records)
        # The records_by_class dict contains copies (to_dict), originals unchanged
        for orig, rec in zip(original_classes, sample_records, strict=False):
            assert rec.evidence_class == orig

    def test_consensus_via_majority_vote(self, sample_records):
        # With 2 records of different classes the vote is tied → consensus is None
        summary = summarize_evidence(sample_records)
        assert summary.consensus_evidence_class is None  # tie
        assert summary.evidence_class_counts[EvidenceClass.POINT_ANOMALY.value] == 1
        assert summary.evidence_class_counts[EvidenceClass.LOCAL_OUTLIER.value] == 1

    def test_empty_records_warning(self):
        summary = summarize_evidence([])
        assert summary.total_records == 0
        assert "no_records_to_summarize" in summary.warnings

    def test_aggregate_confidence(self, sample_records):
        summary = summarize_evidence(sample_records)
        assert summary.aggregate_confidence == pytest.approx((0.9 + 0.7) / 2)

    def test_guardrail_passes_for_legitimate_summary(self, sample_records):
        summary = summarize_evidence(sample_records)
        guardrail_result = check_skill_output(
            "summarize_evidence",
            summary.to_dict(),
            original_evidence=sample_records,
        )
        assert guardrail_result.passed is True

    def test_guardrail_fails_on_evidence_class_override(self, sample_records):
        # Simulate a bad skill that tampers with evidence_class of an EvidenceRecord instance
        bad_record = EvidenceRecord(
            evidence_class="TAMPERED_CLASS",  # different from original POINT_ANOMALY
            confidence=0.99,
            detector="test",
            score=1.0,
            threshold=0.5,
            supporting_metrics={},
            assumptions=(),
            limitations=(),
            interpretation="Tampered.",
            causality_status=EvidenceClass.NOT_APPLICABLE,  # type: ignore[arg-type]
        )
        guardrail_result = check_skill_output(
            "bad_skill",
            bad_record,
            original_evidence=sample_records,
        )
        assert guardrail_result.passed is False
        assert len(guardrail_result.violations) > 0
        assert any(v.attempted_value == "TAMPERED_CLASS" for v in guardrail_result.violations)


# -----------------------------------------------------------------------
# export_report
# -----------------------------------------------------------------------


class TestExportReportBoundary:
    def test_returns_exportedreport(self, sample_records):
        summary = summarize_evidence(sample_records)
        report = export_report(summary, skill_chain=("summarize_evidence",))
        assert isinstance(report, ExportedReport)

    def test_report_is_frozen(self, sample_records):
        summary = summarize_evidence(sample_records)
        report = export_report(summary)
        with pytest.raises(AttributeError):
            report.metadata = ReportMetadata(  # type: ignore[misc]
                dataset_name=None,
                feature=None,
                analyst_notes=None,
                tool_version="0.5.0",
                generated_at="",
            )

    def test_evidence_not_mutated(self, sample_records):
        summary = summarize_evidence(sample_records)
        report = export_report(summary)
        # The evidence_summary dict is a to_dict copy
        assert report.evidence_summary["total_records"] == len(sample_records)
        # Original records unchanged
        for rec in sample_records:
            assert rec.evidence_class in (
                EvidenceClass.POINT_ANOMALY.value,
                EvidenceClass.LOCAL_OUTLIER.value,
            )

    def test_skill_chain_recorded(self, sample_records):
        summary = summarize_evidence(sample_records)
        chain = ("inspect_data", "select_analysis_plan", "summarize_evidence")
        report = export_report(summary, skill_chain=chain)
        assert report.skill_chain == chain


# -----------------------------------------------------------------------
# SKILL_OUTPUT_CANNOT_OVERRIDE_CORE_EVIDENCE — general guardrail
# -----------------------------------------------------------------------


class TestSkillOutputCannotOverrideCoreEvidence:
    def test_none_original_evidence_passes(self):
        result = check_skill_output("any_skill", {"evidence_class": "ANY"}, original_evidence=None)
        assert result.passed is True

    def test_skill_output_dict_with_evidence_class_key_raises(self, sample_records):
        bad_output = {"evidence_class": "TAMPERED", "confidence": 0.99}
        guardrail_result = check_skill_output(
            "bad_skill",
            bad_output,
            original_evidence=sample_records,
        )
        assert guardrail_result.passed is False

    def test_skill_output_list_of_dicts_tampering_caught(self, sample_records):
        bad_list = [{"evidence_class": "TAMPERED", "confidence": 0.99}]
        guardrail_result = check_skill_output(
            "bad_skill",
            bad_list,
            original_evidence=sample_records,
        )
        assert guardrail_result.passed is False

    def test_evidence_record_instance_tampering_caught(self, sample_records):
        bad_record = EvidenceRecord(
            evidence_class="TAMPERED_CLASS",
            confidence=0.99,
            detector="test",
            score=1.0,
            threshold=0.5,
            supporting_metrics={},
            assumptions=(),
            limitations=(),
            interpretation="Tampered.",
        )
        guardrail_result = check_skill_output(
            "bad_skill",
            bad_record,
            original_evidence=sample_records,
        )
        assert guardrail_result.passed is False

    def test_legitimate_none_passes(self, sample_records):
        result = check_skill_output("some_skill", None, original_evidence=sample_records)
        assert result.passed is True

    def test_result_is_frozen(self, sample_records):
        result = check_skill_output("some_skill", {"foo": "bar"}, original_evidence=sample_records)
        with pytest.raises(AttributeError):
            result.passed = False  # type: ignore[misc]
