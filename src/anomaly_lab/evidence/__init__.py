"""Evidence API: taxonomy, applicability, comparability, and verdicts.

Public API
----------
Taxonomy (from .taxonomy):
    EvidenceClass, CausalityStatus, EvidenceStrength,
    EvidenceVerdict, VerdictDecision,
    VerdictReason, EvidenceRecord, EvidenceApplicability,
    DetectionApplicability, ComparisonApplicability,
    EvidenceVerdictResult, APPLICABILITY_RULES_VERSION, DEFAULT_APPLICABILITY_RULES

Applicability (from .applicability):
    get_applicability

Comparability (from .comparability):
    check_comparability, is_comparable, ComparabilityResult

Verdicts (from .verdicts):
    compute_detection_verdict, compute_comparison_verdict,
    attach_verdict, attach_applicability

Backward compatibility (from the original evidence module):
    classify_detection_evidence, classify_comparison_evidence
"""

from __future__ import annotations

# Backward compatibility: the original evidence.py module is shadowed by this
# package.  Use importlib to load it directly from its file path, bypassing
# the normal import system (which would resolve anomaly_lab.evidence to the
# package instead of the module).
import importlib.util
import os
import sys

# Applicability
from anomaly_lab.evidence.applicability import get_applicability

# Comparability
from anomaly_lab.evidence.comparability import (
    check_comparability,
    is_comparable,
)

# Taxonomy
from anomaly_lab.evidence.taxonomy import (
    APPLICABILITY_RULES_VERSION,
    DEFAULT_APPLICABILITY_RULES,
    CausalityStatus,
    ComparisonApplicability,
    DetectionApplicability,
    EvidenceApplicability,
    EvidenceClass,
    EvidenceRecord,
    EvidenceStrength,
    EvidenceVerdict,
    EvidenceVerdictResult,
    VerdictDecision,
    VerdictReason,
)

# Verdicts
from anomaly_lab.evidence.verdicts import (
    attach_applicability,
    attach_verdict,
    compute_comparison_verdict,
    compute_detection_verdict,
)

_OLD_MODULE_NAME = "anomaly_lab._evidence_legacy"
_spec = importlib.util.find_spec("anomaly_lab.evidence")
_evidence_file_path = os.path.join(os.path.dirname(__file__), "..", "evidence.py")
if _evidence_file_path and os.path.exists(_evidence_file_path):
    _spec = importlib.util.spec_from_file_location(_OLD_MODULE_NAME, _evidence_file_path)
    if _spec and _spec.loader:
        _old = importlib.util.module_from_spec(_spec)
        # Set it in sys.modules BEFORE loading to prevent circular dependency
        sys.modules[_OLD_MODULE_NAME] = _old
        _spec.loader.exec_module(_old)
        classify_detection_evidence = getattr(_old, "classify_detection_evidence", None)
        classify_comparison_evidence = getattr(_old, "classify_comparison_evidence", None)
    else:
        classify_detection_evidence = None
        classify_comparison_evidence = None
else:
    classify_detection_evidence = None
    classify_comparison_evidence = None

__all__ = [
    # Taxonomy
    "EvidenceClass",
    "CausalityStatus",
    "EvidenceVerdict",
    "VerdictDecision",
    "VerdictReason",
    "EvidenceRecord",
    "EvidenceStrength",
    "EvidenceApplicability",
    "DetectionApplicability",
    "ComparisonApplicability",
    "EvidenceVerdictResult",
    "APPLICABILITY_RULES_VERSION",
    "DEFAULT_APPLICABILITY_RULES",
    # Applicability
    "get_applicability",
    # Comparability
    "check_comparability",
    "is_comparable",
    # Verdicts
    "compute_detection_verdict",
    "compute_comparison_verdict",
    "attach_verdict",
    "attach_applicability",
    # Backward compat
    "classify_detection_evidence",
    "classify_comparison_evidence",
]
