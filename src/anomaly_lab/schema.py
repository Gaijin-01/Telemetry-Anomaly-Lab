class AnomalyLabError(ValueError):
    """Base error for invalid input or analysis configuration."""


class EmptyDataError(AnomalyLabError):
    """Input contains no rows."""


class InsufficientDataError(AnomalyLabError):
    """Input contains too few rows for the requested detector to operate reliably.

    Each detector has a minimum sample requirement above zero.
    """


class FeatureSelectionError(AnomalyLabError):
    """Feature selection is missing, invalid, or uninformative."""


class NonNumericFeatureError(AnomalyLabError):
    """Selected feature is not numeric."""


class NonFiniteValueError(AnomalyLabError):
    """Selected feature contains NaN or infinity."""


class InvalidDetectorConfigError(AnomalyLabError):
    """Detector configuration is invalid."""


class InputReadError(AnomalyLabError):
    """Input file cannot be read as a telemetry CSV."""


class ConfigValidationError(AnomalyLabError):
    """TOML configuration is invalid or refers to unknown fields."""


class RegimeDetectionError(AnomalyLabError):
    """Regime analysis precondition failed."""
