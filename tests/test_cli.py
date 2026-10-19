import json

from anomaly_lab.cli import main


def test_cli_writes_machine_readable_result_for_valid_input(tmp_path, capsys):
    source = tmp_path / "input.csv"
    source.write_text("x\n1\n1\n1\n1\n1\n2\n")
    output = tmp_path / "result.json"
    assert (
        main(
            [
                "analyze",
                str(source),
                "--detector",
                "robust-zscore",
                "--features",
                "x",
                "--output-json",
                str(output),
            ]
        )
        == 0
    )
    assert "statistical observations" in capsys.readouterr().out
    payload = json.loads(output.read_text())
    assert payload["detector"] == "robust-zscore"
    assert any(row["is_anomaly"] for row in payload["rows"])


def test_cli_fails_cleanly_for_missing_feature_and_nonfinite_input(tmp_path, capsys):
    source = tmp_path / "input.csv"
    source.write_text("x\n1\n2\n")
    assert main(["analyze", str(source), "--detector", "dbscan", "--features", "missing"]) == 2
    captured = capsys.readouterr()
    assert "error:" in captured.err and "traceback" not in captured.err.lower()
    source.write_text("x\n1\nNaN\n")
    assert main(["analyze", str(source), "--detector", "robust-zscore", "--features", "x"]) == 2
    assert "non-finite" in capsys.readouterr().err


def test_cli_baseline_emits_baseline_and_residual_json(tmp_path):
    source = tmp_path / "input.csv"
    source.write_text("x\n1\n2\n5\n")
    output = tmp_path / "baseline.json"
    assert (
        main(
            [
                "baseline",
                str(source),
                "--features",
                "x",
                "--method",
                "rolling",
                "--window",
                "2",
                "--output-json",
                str(output),
            ]
        )
        == 0
    )
    payload = json.loads(output.read_text())
    assert payload["method"] == "rolling"
    assert payload["baseline"]["x"][2] == 1.5
    assert payload["residual"]["x"][2] == 3.5


def test_cli_change_points_emits_retrospective_candidates(tmp_path):
    source = tmp_path / "input.csv"
    source.write_text("x\n1\n1\n10\n10\n")
    output = tmp_path / "points.json"
    assert (
        main(
            [
                "change-points",
                str(source),
                "--feature",
                "x",
                "--min-segment",
                "2",
                "--threshold",
                "5",
                "--output-json",
                str(output),
            ]
        )
        == 0
    )
    payload = json.loads(output.read_text())
    assert payload["retrospective"] is True
    assert payload["candidates"][0]["split_position"] == 2


def test_cli_regimes_reports_dbscan_labels(tmp_path):
    source = tmp_path / "input.csv"
    source.write_text("x\n0\n0.1\n10\n10.1\n")
    output = tmp_path / "regimes.json"
    assert (
        main(
            [
                "regimes",
                str(source),
                "--features",
                "x",
                "--eps",
                "0.3",
                "--min-samples",
                "2",
                "--output-json",
                str(output),
            ]
        )
        == 0
    )
    payload = json.loads(output.read_text())
    assert payload["clusters"] == [0, 0, 1, 1]
    assert payload["metadata"]["cluster_id_semantics"] == "opaque; no semantic meaning"
