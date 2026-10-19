import pytest

from anomaly_lab.ingest import read_csv
from anomaly_lab.schema import InputReadError


def test_reads_local_csv(tmp_path):
    source = tmp_path / "data.csv"
    source.write_text("x\n1\n2\n")
    assert read_csv(source)["x"].tolist() == [1, 2]


def test_rejects_missing_empty_and_header_only_csv(tmp_path):
    with pytest.raises(InputReadError, match="does not exist"):
        read_csv(tmp_path / "missing.csv")
    empty = tmp_path / "empty.csv"
    empty.write_text("")
    with pytest.raises(InputReadError):
        read_csv(empty)
    header = tmp_path / "header.csv"
    header.write_text("x,y\n")
    with pytest.raises(InputReadError, match="no data rows"):
        read_csv(header)
