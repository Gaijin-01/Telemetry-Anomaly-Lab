import pandas as pd
import pytest

from anomaly_lab.compare import MetricRule, RunMetadata, compare_runs
from anomaly_lab.schema import AnomalyLabError


def metadata():
    return RunMetadata(model="m", version="1", quantization="q4", concurrency="1")


def test_insufficient_sample_rejects_overflowing_relative_change():
    with pytest.raises(AnomalyLabError, match="finite"):
        compare_runs(
            pd.DataFrame({"x": [1e-308, 1e-308]}),
            pd.DataFrame({"x": [1e308, 1e308]}),
            metadata(),
            metadata(),
            ["x"],
            [MetricRule("x", "higher", 0.1)],
            min_samples=3,
        )
