"""conftest — shared pytest configuration for Telemetry Anomaly Lab."""

import pytest


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "requires_ruptures: skip test if ruptures is not installed",
    )


def pytest_collection_modifyitems(items):
    """Skip tests marked requires_ruptures when the package is unavailable."""
    try:
        import ruptures  # noqa: F401
    except Exception:
        skip_ruptures = pytest.mark.skip(reason="ruptures not installed")
        for item in items:
            if "requires_ruptures" in item.keywords:
                item.add_marker(skip_ruptures)
