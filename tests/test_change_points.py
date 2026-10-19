import json

import numpy as np
import pandas as pd
import pytest

from anomaly_lab.change_points import detect_change_points
from anomaly_lab.schema import (
    FeatureSelectionError,
    InvalidDetectorConfigError,
    NonFiniteValueError,
)


def test_reports_deterministic_median_shift_candidate_and_serializes():
    frame = pd.DataFrame(
        {"time": pd.date_range("2026-01-01", periods=8), "latency": [1, 1, 1, 1, 5, 5, 5, 5]}
    )
    result = detect_change_points(frame, "latency", min_segment=2, threshold=2, order_by="time")
    assert result.candidates[0].split_position == 2
    assert any(candidate.split_position == 4 for candidate in result.candidates)
    candidate = next(candidate for candidate in result.candidates if candidate.split_position == 4)
    assert candidate.left_median == 1
    assert candidate.right_median == 5
    assert candidate.median_shift == 4
    assert result.retrospective is True
    assert json.dumps(result.to_dict(), allow_nan=False)


def test_no_shift_when_threshold_not_met():
    frame = pd.DataFrame({"x": [1, 1, 2, 2, 1, 1]})
    assert detect_change_points(frame, "x", min_segment=2, threshold=3).candidates == ()


def test_minimum_segment_length_is_enforced():
    frame = pd.DataFrame({"x": [0, 0, 0, 10]})
    result = detect_change_points(frame, "x", min_segment=2, threshold=1)
    assert all(
        c.split_position >= 2 and len(frame) - c.split_position >= 2 for c in result.candidates
    )
    with pytest.raises(InvalidDetectorConfigError):
        detect_change_points(frame, "x", min_segment=1, threshold=1)


def test_order_by_requires_strictly_increasing_unique_values():
    for times in ([2, 1, 3], [1, 1, 2], [1, None, 3]):
        frame = pd.DataFrame({"time": times, "x": [1, 2, 3]})
        with pytest.raises(FeatureSelectionError):
            detect_change_points(frame, "x", min_segment=2, threshold=0, order_by="time")


def test_rejects_nonfinite_values_without_mutating_input():
    frame = pd.DataFrame({"x": [1.0, np.inf, 4.0]})
    before = frame.copy(deep=True)
    with pytest.raises(NonFiniteValueError):
        detect_change_points(frame, "x", min_segment=2, threshold=0)
    pd.testing.assert_frame_equal(frame, before)
