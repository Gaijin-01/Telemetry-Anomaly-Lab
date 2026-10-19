"""Detector Applicability Registry — contracts and preflight checks."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from anomaly_lab.semantics import DatasetProfile, DatasetSemantics


class ApplicabilityCode(Enum):
    APPLICABLE = "APPLICABLE"
    NOT_APPLICABLE = "NOT_APPLICABLE"
    INSUFFICIENT_DATA = "INSUFFICIENT_DATA"
    INVALID_INPUT = "INVALID_INPUT"


@dataclass(frozen=True, kw_only=True)
class DetectorContract:
    name: str
    family: str
    supported_semantics: frozenset[DatasetSemantics]
    min_samples: int
    constant_feature_behavior: str
    missing_value_policy: str
    scaling_required: bool
    temporal_order_required: bool
    stochastic: bool
    random_state_support: bool
    score_direction: str
    output_type: str
    score_normalization: str | None


# -------------------------------------------------------------------
# Predefined detector contracts
# -------------------------------------------------------------------

ROBUST_ZSCORE_CONTRACT = DetectorContract(
    name="robust-zscore",
    family="statistical",
    supported_semantics=frozenset(
        {
            DatasetSemantics.UNORDERED_TABULAR,
            DatasetSemantics.ORDERED_SERIES,
            DatasetSemantics.INDEPENDENT_EXPERIMENT,
        }
    ),
    min_samples=3,
    constant_feature_behavior="reject",
    missing_value_policy="reject",
    scaling_required=False,
    temporal_order_required=False,
    stochastic=False,
    random_state_support=False,
    score_direction="higher_is_more_anomalous",
    output_type="point_anomaly",
    score_normalization=None,
)

ISOLATION_FOREST_CONTRACT = DetectorContract(
    name="isolation-forest",
    family="ensemble",
    supported_semantics=frozenset(
        {
            DatasetSemantics.UNORDERED_TABULAR,
            DatasetSemantics.ORDERED_SERIES,
            DatasetSemantics.INDEPENDENT_EXPERIMENT,
        }
    ),
    min_samples=2,
    constant_feature_behavior="skip",
    missing_value_policy="skip",
    scaling_required=True,
    temporal_order_required=False,
    stochastic=True,
    random_state_support=True,
    score_direction="higher_is_more_anomalous",
    output_type="point_anomaly",
    score_normalization=None,
)

DBSCAN_CONTRACT = DetectorContract(
    name="dbscan",
    family="clustering",
    supported_semantics=frozenset(
        {
            DatasetSemantics.UNORDERED_TABULAR,
            DatasetSemantics.ORDERED_SERIES,
            DatasetSemantics.INDEPENDENT_EXPERIMENT,
        }
    ),
    min_samples=1,
    constant_feature_behavior="skip",
    missing_value_policy="skip",
    scaling_required=True,
    temporal_order_required=False,
    stochastic=False,
    random_state_support=False,
    score_direction="lower_is_more_anomalous",
    output_type="cluster/noise",
    score_normalization=None,
)

ALL_DETECTOR_CONTRACTS: dict[str, DetectorContract] = {
    "robust-zscore": ROBUST_ZSCORE_CONTRACT,
    "isolation-forest": ISOLATION_FOREST_CONTRACT,
    "dbscan": DBSCAN_CONTRACT,
}


# -------------------------------------------------------------------
# Preflight applicability check
# -------------------------------------------------------------------


def check_applicability(
    contract: DetectorContract,
    profile: DatasetProfile,
    n_samples: int,
    constant_features: tuple[str, ...],
    has_missing: bool,
) -> tuple[ApplicabilityCode, str]:
    """
    Check whether a detector contract applies to a given dataset profile.

    Returns (ApplicabilityCode, reason).
    """
    # Check semantics support
    if profile.semantics not in contract.supported_semantics:
        return (
            ApplicabilityCode.NOT_APPLICABLE,
            f"Detector '{contract.name}' does not support semantics "
            f"'{profile.semantics.value}'. "
            f"Supported: {', '.join(s.value for s in contract.supported_semantics)}",
        )

    # Check minimum sample count
    if n_samples < contract.min_samples:
        return (
            ApplicabilityCode.INSUFFICIENT_DATA,
            f"Dataset has {n_samples} samples but '{contract.name}' requires "
            f"at least {contract.min_samples}.",
        )

    # Check constant features
    if constant_features and contract.constant_feature_behavior == "reject":
        return (
            ApplicabilityCode.INVALID_INPUT,
            f"Dataset has constant features {list(constant_features)}; "
            f"detector '{contract.name}' rejects constant features.",
        )

    # Check missing values
    if has_missing and contract.missing_value_policy == "reject":
        return (
            ApplicabilityCode.INVALID_INPUT,
            f"Dataset contains missing values; detector '{contract.name}' "
            f"does not handle missing values.",
        )

    # Check temporal order requirement
    if contract.temporal_order_required and profile.ordering_field is None:
        return (
            ApplicabilityCode.NOT_APPLICABLE,
            f"Detector '{contract.name}' requires temporal ordering, but no "
            f"ordering field was detected.",
        )

    return (
        ApplicabilityCode.APPLICABLE,
        f"Detector '{contract.name}' is applicable to this dataset "
        f"(semantics={profile.semantics.value}, n_samples={n_samples}).",
    )
