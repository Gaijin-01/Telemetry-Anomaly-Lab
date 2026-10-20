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


def test_changelog_has_v050_and_previous_sections():
    """CHANGELOG.md must contain v0.5.0 and all previous release sections."""
    root = Path(__file__).resolve().parents[1]
    changelog = (root / "CHANGELOG.md").read_text()
    assert "## v0.5.0" in changelog, "CHANGELOG.md missing '## v0.5.0' section"
    assert "## v0.4.2" in changelog, "CHANGELOG.md missing '## v0.4.2' section"
    assert "## v0.4.1" in changelog, "CHANGELOG.md missing '## v0.4.1' section"
    assert "## v0.4.0" in changelog, "CHANGELOG.md missing '## v0.4.0' section"
    assert "## v0.3.0" in changelog, "CHANGELOG.md missing '## v0.3.0' section"


def test_schema_version_is_05():
    """DetectionResult schema_version must be '0.5'."""
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
    assert result.schema_version == "0.5", (
        f"schema_version is {result.schema_version!r}, expected '0.5'"
    )


def test_pyproject_version_is_050():
    """pyproject.toml version must be 0.5.0."""
    data = _load_pyproject()
    assert data["project"]["version"] == "0.5.0", (
        f"pyproject.toml version is {data['project']['version']!r}, expected '0.5.0'"
    )


def test_package_version_matches_schema_version_prefix():
    """Package version 0.5.x must have schema_version prefix 0.5."""
    data = _load_pyproject()
    pkg_version = data["project"]["version"]
    major = pkg_version.split(".")[0]
    assert major == "0", f"Major version {major!r} not '0'"
    assert pkg_version.startswith("0.5"), f"package version {pkg_version!r} does not start with 0.5"

    from anomaly_lab.results import DetectionResult

    result = DetectionResult(
        detector="x",
        row_indices=(0,),
        is_anomaly=(False,),
        features=("x",),
        constant_features=(),
        parameters={},
        scores=(0.0,),
        metadata={},
    )
    schema_prefix = ".".join(result.schema_version.split(".")[:2])
    assert schema_prefix == "0.5", (
        f"schema_version {result.schema_version!r} does not share the package major.minor prefix"
    )


def test_no_toml_third_party_dependency():
    """The stdlib-only TOML invariant must hold: no 'toml' runtime dependency."""
    data = _load_pyproject()
    runtime_deps = data.get("project", {}).get("dependencies", [])
    offenders = [d for d in runtime_deps if d.split()[0].split("[")[0].strip() == "toml"]
    assert not offenders, f"third-party toml dependency present: {offenders}"


def test_stdlib_only_toml_writer_used():
    """runner.py must not import the third-party 'toml' module."""
    root = Path(__file__).resolve().parents[1]
    source = (root / "src" / "anomaly_lab" / "experiments" / "runner.py").read_text()
    assert "import toml" not in source, "runner.py must not import third-party 'toml'"
