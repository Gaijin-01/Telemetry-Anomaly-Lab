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
