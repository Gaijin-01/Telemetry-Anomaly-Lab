import json

import numpy as np
import pytest

from anomaly_lab.results import DetectionResult


def test_result_serializes_indices_and_numpy_scalars_to_json():
    result = DetectionResult(
        detector="example",
        row_indices=(np.int64(5), 5),
        is_anomaly=(False, True),
        features=("x",),
        constant_features=(),
        parameters={"seed": np.int64(4)},
        scores=(0.1, 1.2),
        metadata={"scale": "standard"},
    )
    payload = result.to_dict()
    encoded = json.dumps(payload, allow_nan=False)
    assert json.loads(encoded)["rows"][1]["index"] == 5


def test_result_rejects_empty_or_inconsistent_rows_and_nonfinite_scores():
    with pytest.raises(ValueError, match="zero rows"):
        DetectionResult("x", (), (), (), (), {})
    with pytest.raises(ValueError, match="length"):
        DetectionResult("x", (1,), (), (), (), {})
    with pytest.raises(ValueError, match="finite"):
        DetectionResult("x", (1,), (True,), (), (), {}, scores=(float("inf"),))
