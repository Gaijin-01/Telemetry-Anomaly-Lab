"""inspect_data skill: surface-level data quality and structure inspection.

No LLM. No side-effects beyond returning a dict.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


@dataclass(frozen=True, kw_only=True)
class DataInspectionResult:
    row_count: int
    column_count: int
    numeric_columns: tuple[str, ...]
    non_numeric_columns: tuple[str, ...]
    null_fraction: dict[str, float]
    constant_columns: tuple[str, ...]
    duplicate_row_count: int
    finite_violations: dict[str, int]  # col -> count of inf/nan
    warnings: tuple[str, ...]
    is_empty: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "row_count": self.row_count,
            "column_count": self.column_count,
            "numeric_columns": list(self.numeric_columns),
            "non_numeric_columns": list(self.non_numeric_columns),
            "null_fraction": dict(self.null_fraction),
            "constant_columns": list(self.constant_columns),
            "duplicate_row_count": self.duplicate_row_count,
            "finite_violations": dict(self.finite_violations),
            "warnings": list(self.warnings),
            "is_empty": self.is_empty,
        }


def inspect_data(df: pd.DataFrame) -> DataInspectionResult:
    """Inspect a DataFrame and return structural metadata.

    Does not modify data. Produces a frozen result that is safe to pass
    to subsequent skills.
    """
    warnings: list[str] = []

    if df is None or df.empty:
        return DataInspectionResult(
            row_count=0,
            column_count=0,
            numeric_columns=(),
            non_numeric_columns=(),
            null_fraction={},
            constant_columns=(),
            duplicate_row_count=0,
            finite_violations={},
            warnings=("empty_dataframe",),
            is_empty=True,
        )

    row_count, column_count = len(df), len(df.columns)

    # Numeric vs non-numeric columns
    numeric_cols = [
        c
        for c in df.columns
        if pd.api.types.is_numeric_dtype(df[c]) and not pd.api.types.is_object_dtype(df[c])
    ]
    non_numeric_cols = [c for c in df.columns if c not in numeric_cols]

    # Null fraction per column
    null_fraction = {
        c: float(df[c].isna().sum()) / row_count if row_count > 0 else 0.0 for c in df.columns
    }

    # Constant columns (all values identical, ignoring NA)
    constant_columns = tuple(c for c in df.columns if df[c].nunique(dropna=True) <= 1)

    # Duplicate rows
    duplicate_row_count = int(df.duplicated().sum())

    # Finite violations (inf or NaN in otherwise numeric cols)
    finite_violations: dict[str, int] = {}
    for c in numeric_cols:
        violations = (
            df[c]
            .apply(
                lambda v: (
                    not isinstance(v, (int, float)) or not np.isfinite(v) if pd.notna(v) else False
                )
            )
            .sum()
        )
        if violations:
            finite_violations[c] = int(violations)

    # Structural warnings
    if duplicate_row_count > 0:
        warnings.append(f"duplicate_rows: {duplicate_row_count}")
    if any(v > 0 for v in finite_violations.values()):
        warnings.append("non_finite_values_detected")
    high_null_cols = [c for c, f in null_fraction.items() if f > 0.5]
    if high_null_cols:
        warnings.append(f"high_null_fraction_columns: {high_null_cols}")

    return DataInspectionResult(
        row_count=row_count,
        column_count=column_count,
        numeric_columns=tuple(numeric_cols),
        non_numeric_columns=tuple(non_numeric_cols),
        null_fraction=null_fraction,
        constant_columns=constant_columns,
        duplicate_row_count=duplicate_row_count,
        finite_violations=finite_violations,
        warnings=tuple(warnings),
        is_empty=False,
    )
