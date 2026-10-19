from pathlib import Path

import pandas as pd

from anomaly_lab.schema import InputReadError


def read_csv(path: str | Path) -> pd.DataFrame:
    source = Path(path)
    if not source.is_file():
        raise InputReadError(f"input CSV does not exist or is not a file: {source}")
    try:
        frame = pd.read_csv(source)
    except (OSError, UnicodeDecodeError, pd.errors.ParserError, pd.errors.EmptyDataError) as exc:
        raise InputReadError(f"cannot read CSV {source}: {exc}") from exc
    if frame.empty:
        raise InputReadError("input CSV has no data rows")
    if len(frame.columns) == 0:
        raise InputReadError("input CSV has no columns")
    return frame
