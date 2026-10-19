"""Tests for v0.3: comparison policy, CI gating, TOML config, change-point ranking."""

from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).parent.parent
EXAMPLES = REPO_ROOT / "examples"
sys_path_insert = str(REPO_ROOT / "src")


class TestCompareAllowDifference:
    """Phase C: allow_difference lets controlled mismatches through."""

    def test_version_mismatch_blocks_by_default(self):
        import sys

        sys.path.insert(0, sys_path_insert)
        from anomaly_lab.compare import MetricRule, RunMetadata, compare_runs
        from anomaly_lab.ingest import read_csv

        bm = RunMetadata(model="qwen", version="v1", quantization="q4", concurrency="4")
        cm = RunMetadata(model="qwen", version="v2", quantization="q4", concurrency="4")
        csv_a = EXAMPLES / "synthetic_telemetry.csv"
        result = compare_runs(
            read_csv(csv_a),
            read_csv(csv_a),
            bm,
            cm,
            ["p95_s"],
            [MetricRule("p95_s", "lower", 0.10)],
        )
        assert result.status == "NOT_COMPARABLE"
        assert "version" in result.mismatches

    def test_version_mismatch_allowed_via_allow_difference(self):
        import sys

        sys.path.insert(0, sys_path_insert)
        from anomaly_lab.compare import MetricRule, RunMetadata, compare_runs
        from anomaly_lab.ingest import read_csv

        bm = RunMetadata(
            model="qwen",
            version="v1",
            quantization="q4",
            concurrency="4",
            allow_difference=("version",),
        )
        cm = RunMetadata(
            model="qwen",
            version="v2",
            quantization="q4",
            concurrency="4",
            allow_difference=("version",),
        )
        csv_a = EXAMPLES / "synthetic_telemetry.csv"
        result = compare_runs(
            read_csv(csv_a),
            read_csv(csv_a),
            bm,
            cm,
            ["p95_s"],
            [MetricRule("p95_s", "lower", 0.10)],
        )
        assert result.status == "NO_REGRESSION"
        assert "version" not in result.mismatches

    def test_unauthorized_second_mismatch_still_blocks(self):
        import sys

        sys.path.insert(0, sys_path_insert)
        from anomaly_lab.compare import MetricRule, RunMetadata, compare_runs
        from anomaly_lab.ingest import read_csv

        bm = RunMetadata(
            model="qwen",
            version="v1",
            quantization="q4",
            concurrency="4",
            allow_difference=("version",),
        )
        cm = RunMetadata(
            model="qwen",
            version="v2",
            quantization="q8",
            concurrency="4",
            allow_difference=("version",),
        )
        csv_a = EXAMPLES / "synthetic_telemetry.csv"
        result = compare_runs(
            read_csv(csv_a),
            read_csv(csv_a),
            bm,
            cm,
            ["p95_s"],
            [MetricRule("p95_s", "lower", 0.10)],
        )
        assert result.status == "NOT_COMPARABLE"
        assert "quantization" in result.mismatches

    def test_allow_difference_unknown_dimension_raises(self):
        import sys

        sys.path.insert(0, sys_path_insert)
        from anomaly_lab.compare import RunMetadata
        from anomaly_lab.schema import AnomalyLabError

        with pytest.raises(AnomalyLabError):
            RunMetadata(
                model="qwen",
                version="v1",
                quantization="q4",
                concurrency="4",
                allow_difference=("nonexistent",),
            )


class TestCIGates:
    """Phase C: --fail-on gates return exit 1 without mutating JSON output."""

    def test_default_regression_exit_0(self):
        import sys

        sys.path.insert(0, sys_path_insert)
        import tempfile

        from anomaly_lab.cli import main

        csv_a = str(EXAMPLES / "synthetic_telemetry.csv")
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as f:
            out = f.name
        try:
            rc = main(
                [
                    "compare",
                    csv_a,
                    csv_a,
                    "--features",
                    "p95_s",
                    "--baseline-metadata",
                    "model=qwen",
                    "--baseline-metadata",
                    "version=v1",
                    "--baseline-metadata",
                    "quantization=q4",
                    "--baseline-metadata",
                    "concurrency=4",
                    "--candidate-metadata",
                    "model=qwen",
                    "--candidate-metadata",
                    "version=v1",
                    "--candidate-metadata",
                    "quantization=q4",
                    "--candidate-metadata",
                    "concurrency=4",
                    "--metric",
                    "p95_s:lower:0.10",
                    "--output-json",
                    out,
                ]
            )
            assert rc == 0
        finally:
            Path(out).unlink(missing_ok=True)

    def test_invalid_config_exit_2(self):
        import sys

        sys.path.insert(0, sys_path_insert)
        from anomaly_lab.cli import main

        rc = main(
            [
                "compare",
                str(EXAMPLES / "synthetic_telemetry.csv"),
                str(EXAMPLES / "synthetic_telemetry.csv"),
                "--features",
                "p95_s",
                "--baseline-metadata",
                "model=qwen",
                # missing required keys
                "--candidate-metadata",
                "model=qwen",
                "--candidate-metadata",
                "version=v1",
                "--candidate-metadata",
                "quantization=q4",
                "--candidate-metadata",
                "concurrency=4",
                "--metric",
                "p95_s:lower:0.10",
            ]
        )
        assert rc == 2


class TestTOMLConfig:
    """Phase C: TOML config round-trips through the compare API."""

    def _toml_file(self, lines):
        import tempfile

        with tempfile.NamedTemporaryFile(
            suffix=".toml", delete=False, mode="w", dir=str(EXAMPLES)
        ) as f:
            f.write("\n".join(lines) + "\n")
            return Path(f.name)

    def test_malformed_toml_raises(self):
        import sys

        sys.path.insert(0, sys_path_insert)
        from anomaly_lab.cli import _load_config

        p = self._toml_file(
            [
                "command = compare",
                "baseline = b.csv",
                "candidate = c.csv",
                "features = ['x']",
                "baseline_metadata = {model='q',version='v',quantization='q',concurrency='4'}",
            ]
        )
        try:
            with pytest.raises(Exception, match="TOML syntax error"):
                _load_config(p)
        finally:
            p.unlink(missing_ok=True)

    def test_unknown_top_level_key_raises(self):
        import sys

        sys.path.insert(0, sys_path_insert)
        from anomaly_lab.cli import _load_config

        p = self._toml_file(
            [
                "command = 'compare'",
                "baseline = 'b.csv'",
                "candidate = 'c.csv'",
                "features = ['p95_s']",
                "baseline_metadata = {model='q',version='v',quantization='q',concurrency='4'}",
                "candidate_metadata = {model='q',version='v',quantization='q',concurrency='4'}",
                "metrics = [{name='p95_s',direction='lower',threshold=0.1}]",
                "unknown_field = 'bad'",
            ]
        )
        try:
            with pytest.raises(Exception, match="unknown field"):
                _load_config(p)
        finally:
            p.unlink(missing_ok=True)

    def test_valid_toml_minimal(self):
        import sys

        sys.path.insert(0, sys_path_insert)
        from anomaly_lab.cli import _load_config

        p = self._toml_file(
            [
                'command = "compare"',
                'baseline = "synthetic_telemetry.csv"',
                'candidate = "synthetic_telemetry.csv"',
                'features = ["p95_s"]',
                'baseline_metadata = {model="qwen",version="v1",quantization="q4",concurrency="4"}',
                'candidate_metadata = {model="qwen",version="v1",quantization="q4",concurrency="4"}',
                "[[metrics]]",
                'name = "p95_s"',
                'direction = "lower"',
                "threshold = 0.1",
            ]
        )
        try:
            cfg = _load_config(p)
            assert cfg["command"] == "compare"
            assert cfg["baseline_metadata"]["version"] == "v1"
            assert len(cfg["metrics"]) == 1
        finally:
            p.unlink(missing_ok=True)

    def test_toml_gate_fail_on(self):
        import sys

        sys.path.insert(0, sys_path_insert)
        from anomaly_lab.cli import _load_config

        p = self._toml_file(
            [
                'command = "compare"',
                'baseline = "synthetic_telemetry.csv"',
                'candidate = "synthetic_telemetry.csv"',
                'features = ["p95_s"]',
                'baseline_metadata = {model="qwen",version="v1",quantization="q4",concurrency="4"}',
                'candidate_metadata = {model="qwen",version="v1",quantization="q4",concurrency="4"}',
                "[[metrics]]",
                'name = "p95_s"',
                'direction = "lower"',
                "threshold = 0.1",
                "[gate]",
                'fail_on = ["regression"]',
            ]
        )
        try:
            cfg = _load_config(p)
            assert cfg["gate"]["fail_on"] == ["regression"]
        finally:
            p.unlink(missing_ok=True)

    def test_toml_gate_invalid_value_raises(self):
        import sys

        sys.path.insert(0, sys_path_insert)
        from anomaly_lab.cli import _load_config

        p = self._toml_file(
            [
                'command = "compare"',
                'baseline = "synthetic_telemetry.csv"',
                'candidate = "synthetic_telemetry.csv"',
                'features = ["p95_s"]',
                'baseline_metadata = {model="qwen",version="v1",quantization="q4",concurrency="4"}',
                'candidate_metadata = {model="qwen",version="v1",quantization="q4",concurrency="4"}',
                "[[metrics]]",
                'name = "p95_s"',
                'direction = "lower"',
                "threshold = 0.1",
                "[gate]",
                'fail_on = ["not-a-real-gate"]',
            ]
        )
        try:
            with pytest.raises(Exception, match="unknown value"):
                _load_config(p)
        finally:
            p.unlink(missing_ok=True)


class TestChangePointRanking:
    """Phase D: improved ranking incorporates support/balance."""

    def test_balanced_split_ranks_ahead_of_edge_split_with_same_shift(self):
        import sys

        sys.path.insert(0, sys_path_insert)
        import pandas as pd

        from anomaly_lab.change_points import detect_change_points

        df = pd.DataFrame({"v": [1.0, 1.0, 1.0, 1.0, 5.0, 5.0, 5.0, 5.0]})
        result = detect_change_points(df, "v", min_segment=2, threshold=0.0)
        assert len(result.candidates) == 5
        top = result.candidates[0]
        assert top.split_position == 4
        assert top.median_shift == 4.0
        assert top.left_size == 4
        assert top.right_size == 4
        assert top.support == 4

    def test_to_dict_includes_ranking_fields(self):
        import sys

        sys.path.insert(0, sys_path_insert)
        import pandas as pd

        from anomaly_lab.change_points import detect_change_points

        df = pd.DataFrame({"v": [1.0, 1.0, 1.0, 1.0, 5.0, 5.0, 5.0, 5.0]})
        result = detect_change_points(df, "v", min_segment=2, threshold=0.0)
        d = result.to_dict()
        c = d["candidates"][0]
        assert "left_size" in c
        assert "right_size" in c
        assert "support" in c
        assert c["support"] == 4

    def test_deterministic_ordering(self):
        import sys

        sys.path.insert(0, sys_path_insert)
        import pandas as pd

        from anomaly_lab.change_points import detect_change_points

        df = pd.DataFrame({"v": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]})
        r1 = detect_change_points(df, "v", min_segment=2, threshold=0.0)
        r2 = detect_change_points(df, "v", min_segment=2, threshold=0.0)
        assert [c.split_position for c in r1.candidates] == [
            c.split_position for c in r2.candidates
        ]

    def test_insufficient_segment_returns_empty_candidates(self):
        import sys

        sys.path.insert(0, sys_path_insert)
        import pandas as pd

        from anomaly_lab.change_points import detect_change_points

        df = pd.DataFrame({"v": [1.0, 2.0, 3.0]})
        result = detect_change_points(df, "v", min_segment=5, threshold=0.0)
        assert result.candidates == ()

    def test_no_mutation_of_input(self):
        import sys

        sys.path.insert(0, sys_path_insert)
        import pandas as pd

        from anomaly_lab.change_points import detect_change_points

        df = pd.DataFrame({"v": [1.0, 2.0, 3.0, 4.0]})
        orig = df["v"].copy()
        detect_change_points(df, "v", min_segment=2, threshold=0.0)
        assert list(df["v"].values) == list(orig.values)
