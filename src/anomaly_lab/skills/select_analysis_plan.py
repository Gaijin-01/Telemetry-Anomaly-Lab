"""select_analysis_plan skill: choose an analysis recipe based on data shape
and declared intent.

No LLM. No autonomous loops. Pure deterministic selection.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from anomaly_lab.skills.inspect_data import DataInspectionResult

AnalysisIntent = Literal["point_anomaly", "regime_shift", "comparison", "exploratory"]


@dataclass(frozen=True, kw_only=True)
class AnalysisPlan:
    intent: str
    detectors: tuple[str, ...]
    regime_detection: bool
    correlation_method: str | None
    min_samples: int
    rationale: tuple[str, ...]

    def to_dict(self) -> dict:
        return {
            "intent": self.intent,
            "detectors": list(self.detectors),
            "regime_detection": self.regime_detection,
            "correlation_method": self.correlation_method,
            "min_samples": self.min_samples,
            "rationale": list(self.rationale),
        }


# Minimum reliable sample sizes per detector type
_MIN_SAMPLES = {
    "isolation-forest": 50,
    "dbscan": 20,
    "robust-zscore": 10,
}


def select_analysis_plan(
    inspection: DataInspectionResult,
    intent: AnalysisIntent,
) -> AnalysisPlan:
    """Select detectors and settings based on data inspection and declared intent.

    Rules
    -----
    - intent='point_anomaly': isolation-forest primary, robust-zscore secondary
    - intent='regime_shift': robust-zscore first, dbscan second
    - intent='comparison': robust-zscore + correlation check
    - intent='exploratory': all three detectors
    - Insufficient rows -> fewer detectors, never more
    - regime_detection added when intent in {regime_shift, exploratory}
      or when row_count > 500 (suggesting possible non-stationarity)
    """
    n = inspection.row_count
    rationale: list[str] = []

    if n < 10:
        rationale.append("very_small_sample: limiting to robust_zscore only")
        detectors = ("robust-zscore",)
        min_samples = _MIN_SAMPLES["robust-zscore"]
    elif n < 50:
        rationale.append("small_sample: excluding isolation_forest")
        detectors = ("robust-zscore", "dbscan")
        min_samples = _MIN_SAMPLES["robust-zscore"]
    else:
        if intent == "point_anomaly":
            rationale.append("point_anomaly_intent: isolation_forest primary")
            detectors = ("isolation-forest", "robust-zscore", "dbscan")
        elif intent == "regime_shift":
            rationale.append("regime_shift_intent: robust_zscore primary")
            detectors = ("robust-zscore", "dbscan", "isolation-forest")
        elif intent == "comparison":
            rationale.append("comparison_intent: robust_zscore + correlation")
            detectors = ("robust-zscore",)
        else:  # exploratory
            rationale.append("exploratory_intent: all detectors")
            detectors = ("isolation-forest", "robust-zscore", "dbscan")
        min_samples = _MIN_SAMPLES["isolation-forest"]

    regime_detection = intent in {"regime_shift", "exploratory"} or n > 500
    if regime_detection:
        rationale.append(
            "regime_detection_enabled" if regime_detection else "regime_detection_suppressed"
        )

    correlation_method = "spearman" if intent == "comparison" else None
    if correlation_method:
        rationale.append("spearman_correlation_for_comparison")

    return AnalysisPlan(
        intent=intent,
        detectors=tuple(detectors),
        regime_detection=regime_detection,
        correlation_method=correlation_method,
        min_samples=min_samples,
        rationale=tuple(rationale),
    )
