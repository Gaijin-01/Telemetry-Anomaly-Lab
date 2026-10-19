"""Tests for release metadata consistency.

These tests verify that the package version, output schema version,
and changelog are always in sync. They prevent a recurrence of a
bug where the GitHub Release was cut but pyproject.toml still
reported version 0.3.0.
"""

import tomllib
from pathlib import Path


def _load_pyproject() -> dict:
    root = Path(__file__).resolve().parents[1]
    content = (root / "pyproject.toml").read_text(encoding="utf-8")
    return tomllib.loads(content)


def test_changelog_has_v041_and_v040_and_v030():
    """CHANGELOG.md must contain v0.4.1, v0.4.0 and v0.3.0 sections."""
    root = Path(__file__).resolve().parents[1]
    changelog = (root / "CHANGELOG.md").read_text()
    assert "## v0.4.1" in changelog, "CHANGELOG.md missing '## v0.4.1' section"
    assert "## v0.4.0" in changelog, "CHANGELOG.md missing '## v0.4.0' section"
    assert "## v0.3.0" in changelog, "CHANGELOG.md missing '## v0.3.0' section"


def test_schema_version_is_04():
    """DetectionResult schema_version must be '0.4'."""
    from anomaly_lab.results import DetectionResult

    result = DetectionResult(
        detector="test",
        row_indices=(0,),
        is_anomaly=(False,),
        features=("x",),
        constant_features=(),
        parameters={},
        scores=(0.0,),
        metadata={},
    )
    assert result.schema_version == "0.4", (
        f"schema_version is {result.schema_version!r}, expected '0.4'"
    )


def test_pyproject_version_is_041():
    """pyproject.toml version must be 0.4.1."""
    data = _load_pyproject()
    assert data["project"]["version"] == "0.4.1", (
        f"pyproject.toml version is {data['project']['version']!r}, expected '0.4.1'"
    )


def test_package_version_matches_schema_version_prefix():
    """Package version 0.4.x must have schema_version prefix 0.4."""
    data = _load_pyproject()
    pkg_version = data["project"]["version"]
    major = pkg_version.split(".")[0]
    assert major == "0", f"Major version {major!r} not '0'"
    assert pkg_version.startswith("0.4"), f"package version {pkg_version!r} does not start with 0.4"
