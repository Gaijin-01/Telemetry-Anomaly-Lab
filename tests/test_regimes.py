"""Tests for regimes.py — Regime-Aware Analysis with ruptures."""

import numpy as np
import pandas as pd
import pytest

from anomaly_lab.regimes import RegimeAnalysisResult, RegimeSegment, detect_regimes
from anomaly_lab.schema import RegimeDetectionError


class TestDetectRegimesBasics:
    """Basic regime detection tests."""

    def test_insufficient_data_raises(self):
        data = [1.0, 2.0, 3.0]
        frame = pd.DataFrame({"x": data})
        with pytest.raises(RegimeDetectionError, match="INSUFFICIENT_DATA"):
            detect_regimes(frame, "x")

    def test_missing_feature_raises(self):
        y_data = [1.0, 2.0, 3.0, 4.0, 5.0]
        frame = pd.DataFrame({"y": y_data})
        with pytest.raises(RegimeDetectionError, match="not found"):
            detect_regimes(frame, "nonexistent")

    @pytest.mark.requires_ruptures
    def test_nonfinite_values_raises(self):
        x_vals = [1.0, np.nan, 3.0, 4.0, 5.0]
        frame = pd.DataFrame({"x": x_vals})
        with pytest.raises(RegimeDetectionError, match="non-finite"):
            detect_regimes(frame, "x")

    @pytest.mark.requires_ruptures
    def test_unordered_ordering_field_raises(self):
        x_vals = [1.0, 2.0, 3.0, 1.0, 2.0]
        t_vals = [10.0, 10.0, 9.0, 8.0, 7.0]
        frame = pd.DataFrame({"x": x_vals, "t": t_vals})
        with pytest.raises(RegimeDetectionError, match="ORDERING_VIOLATION"):
            detect_regimes(frame, "x", ordering_field="t")

    @pytest.mark.requires_ruptures
    def test_unknown_method_raises(self):
        rng = np.random.default_rng(0)
        x_vals = rng.normal(size=20)
        frame = pd.DataFrame({"x": x_vals})
        with pytest.raises(RegimeDetectionError, match="method"):
            detect_regimes(frame, "x", method="unknown_method")

    @pytest.mark.requires_ruptures
    def test_unknown_model_raises(self):
        rng = np.random.default_rng(0)
        x_vals = rng.normal(size=20)
        frame = pd.DataFrame({"x": x_vals})
        with pytest.raises(RegimeDetectionError, match="model"):
            detect_regimes(frame, "x", model="unknown_model")

    def test_raises_without_ruptures(self):
        """When ruptures cannot be imported, error directs to pip install."""
        import builtins

        orig_import = builtins.__import__

        def fake_import(name, *args, **kwargs):
            if name == "ruptures":
                raise ImportError("No module named 'ruptures'")
            return orig_import(name, *args, **kwargs)

        import sys

        sys.modules.pop("ruptures", None)
        builtins.__import__ = fake_import
        try:
            rng = np.random.default_rng(0)
            x_vals = rng.normal(size=15)
            frame = pd.DataFrame({"x": x_vals})
            with pytest.raises(RegimeDetectionError, match="pip install"):
                detect_regimes(frame, "x")
        finally:
            builtins.__import__ = orig_import


@pytest.mark.requires_ruptures
class TestDetectRegimesResults:
    """Test RegimeAnalysisResult structure and fields."""

    @pytest.fixture
    def regime_result(self):
        """Synthetic series with two regimes: low then high values."""
        rng = np.random.default_rng(42)
        regime1 = rng.normal(loc=0.0, size=30)
        regime2 = rng.normal(loc=5.0, size=30)
        series = np.concatenate([regime1, regime2])
        frame = pd.DataFrame({"x": series})
        return detect_regimes(frame, "x", penalty=10.0)

    def test_result_is_regime_analysis_result(self, regime_result):
        assert isinstance(regime_result, RegimeAnalysisResult)

    def test_result_has_required_fields(self, regime_result):
        assert hasattr(regime_result, "feature")
        assert hasattr(regime_result, "breakpoints")
        assert hasattr(regime_result, "segments")
        assert hasattr(regime_result, "n_regimes")
        assert hasattr(regime_result, "ordering_field")
        assert hasattr(regime_result, "penalty")
        assert hasattr(regime_result, "method")
        assert hasattr(regime_result, "model")
        assert hasattr(regime_result, "interpretation")
        assert hasattr(regime_result, "assumptions")
        assert hasattr(regime_result, "limitations")

    def test_breakpoints_start_at_zero(self, regime_result):
        assert regime_result.breakpoints[0] == 0

    def test_breakpoints_end_at_length(self, regime_result):
        assert regime_result.breakpoints[-1] == 60

    def test_segments_are_regime_segments(self, regime_result):
        for seg in regime_result.segments:
            assert isinstance(seg, RegimeSegment)
            assert seg.start_index < seg.end_index
            assert seg.n_rows == seg.end_index - seg.start_index
            assert "x" in seg.median_values
            assert "x" in seg.mean_values

    def test_n_regimes_matches_segments(self, regime_result):
        assert regime_result.n_regimes == len(regime_result.segments)

    def test_to_dict_returns_dict(self, regime_result):
        d = regime_result.to_dict()
        assert isinstance(d, dict)
        assert d["feature"] == "x"
        assert d["method"] == "pelt"
        assert d["model"] == "l2"


class TestDetectRegimesOrdering:
    """Test ordering field handling."""

    def test_ordering_field_requires_strict_increasing(self):
        """When ordering_field is provided and not strictly increasing, raise."""
        x_vals = [1.0, 2.0, 3.0]
        t_vals = [1.0, 1.0, 2.0]  # not strictly increasing (1.0, 1.0, ...)
        frame = pd.DataFrame({"x": x_vals * 5, "t": t_vals * 5})
        with pytest.raises(RegimeDetectionError, match="ORDERING_VIOLATION"):
            detect_regimes(frame, "x", ordering_field="t")

    def test_missing_ordering_field_raises(self):
        x_vals = list(range(15))
        t_vals = list(range(15))
        frame = pd.DataFrame({"x": x_vals, "t": t_vals})
        with pytest.raises(RegimeDetectionError, match="not found"):
            detect_regimes(frame, "x", ordering_field="nonexistent")

    @pytest.mark.requires_ruptures
    def test_ordering_field_is_recorded_in_result(self):
        rng = np.random.default_rng(0)
        x_vals = rng.normal(size=20)
        t_vals = list(range(20))
        frame = pd.DataFrame({"x": x_vals, "t": t_vals})
        result = detect_regimes(frame, "x", ordering_field="t", penalty=5.0)
        assert result.ordering_field == "t"


@pytest.mark.requires_ruptures
class TestDetectRegimesMethods:
    """Test different segmentation methods."""

    @pytest.fixture
    def frame(self):
        rng = np.random.default_rng(7)
        regime1 = rng.normal(loc=0.0, size=25)
        regime2 = rng.normal(loc=4.0, size=25)
        return pd.DataFrame({"x": np.concatenate([regime1, regime2])})

    def test_pelt_method(self, frame):
        result = detect_regimes(frame, "x", method="pelt", penalty=5.0)
        assert result.method == "pelt"
        assert len(result.breakpoints) >= 2

    def test_bottomup_method(self, frame):
        result = detect_regimes(frame, "x", method="bottomup")
        assert result.method == "bottomup"

    def test_window_method(self, frame):
        result = detect_regimes(frame, "x", method="window")
        assert result.method == "window"


class TestRegimeSegmentDataclass:
    """Test RegimeSegment frozen dataclass."""

    def test_regime_segment_is_frozen(self):
        seg = RegimeSegment(
            start_index=0,
            end_index=10,
            start_timestamp="2024-01-01T00:00:00",
            end_timestamp="2024-01-01T00:00:10",
            median_values={"x": 1.5},
            mean_values={"x": 1.6},
            n_rows=10,
            segment_id=0,
        )
        with pytest.raises(AttributeError):
            seg.start_index = 5  # type: ignore

    def test_regime_segment_kw_only(self):
        seg = RegimeSegment(
            start_index=0,
            end_index=10,
            start_timestamp=None,
            end_timestamp=None,
            median_values={},
            mean_values={},
            n_rows=10,
            segment_id=0,
        )
        assert seg.n_rows == 10
