from pathlib import Path


def test_project_version_is_v040():
    import tomllib

    project = tomllib.loads(Path("pyproject.toml").read_text())
    assert project["project"]["version"] == "0.4.0"


def test_public_release_scope_and_noncausal_contract():
    notes = Path("CHANGELOG.md").read_text().lower()
    for phrase in (
        "compare",
        "baseline",
        "change-points",
        "regimes",
        "metadata",
        "past-only",
        "ewma",
        "causality: not established",
    ):
        assert phrase in notes


def test_optional_real_hardware_data_is_not_claimed_as_shipped():
    notes = Path("CHANGELOG.md").read_text().lower()
    assert "real rtx 6000 data is not included" in notes
