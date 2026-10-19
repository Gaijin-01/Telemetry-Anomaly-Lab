"""Reproducibility and provenance tracking for anomaly-lab analyses."""

from __future__ import annotations

import hashlib
import sys
import uuid
from typing import Any

import pandas as pd


def collect_dependency_versions() -> dict[str, str | None]:
    """Return installed versions of key dependencies."""
    import importlib

    packages = ["numpy", "pandas", "scikit-learn", "scipy"]
    result: dict[str, str | None] = {}
    for pkg in packages:
        try:
            m = importlib.import_module(pkg)
            result[pkg] = getattr(m, "__version__", "unknown")
        except ImportError:
            result[pkg] = None
    return result


def compute_input_hash(frame: pd.DataFrame) -> str:
    """SHA256 of sorted column names + first 1000 rows."""
    # Step 1: reorder columns alphabetically for canonical representation
    sorted_frame = frame[sorted(frame.columns)]
    # Step 2: take first 1000 rows and sort them lexicographically
    snapshot = sorted_frame.head(1000)
    sorted_rows = snapshot.sort_values(list(sorted_frame.columns)).reset_index(drop=True)
    # Step 3: CSV with sorted columns and sorted rows
    rows_bytes = sorted_rows.to_csv(index=False, header=False).encode("utf-8")
    col_part = "|".join(sorted(frame.columns)).encode("utf-8")
    raw = col_part + b"||" + rows_bytes
    return hashlib.sha256(raw).hexdigest()


def generate_analysis_id() -> str:
    """UUID4-based analysis ID."""
    return str(uuid.uuid4())


def collect_environment_info() -> dict[str, Any]:
    """
    Collect environment information for reproducibility.

    Returns a dict with Python version, package versions, OS info, and platform.
    """
    import platform as _platform

    deps = collect_dependency_versions()
    return {
        "python_version": f"{sys.version_info.major}.{sys.version_info.minor}."
        f"{sys.version_info.micro}",
        "platform": _platform.platform(),
        "processor": _platform.processor(),
        "dependencies": deps,
    }


def build_reproducibility_metadata(
    analysis_id: str,
    input_hash: str | None,
    seed: int,
    parameters: dict[str, Any],
    paired: bool,
) -> dict[str, Any]:
    """Assemble full reproducibility block."""
    deps = collect_dependency_versions()
    return {
        "analysis_id": analysis_id,
        "input_hash": input_hash,
        "seed": seed,
        "paired": paired,
        "parameters": dict(parameters),
        "dependencies": deps,
    }
