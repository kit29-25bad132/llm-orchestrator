"""Phase 1 foundation tests.

These verify the package foundation only: that ``llm_orchestrator`` is
installed, imports cleanly, and exposes package identity without any provider or
orchestration APIs (none exist yet).
"""

from __future__ import annotations

from importlib.metadata import metadata

import llm_orchestrator

DISTRIBUTION_NAME = "llm-orchestrator"


def test_package_imports() -> None:
    assert llm_orchestrator is not None


def test_version_is_exposed() -> None:
    assert isinstance(llm_orchestrator.__version__, str)
    assert llm_orchestrator.__version__ != ""


def test_public_api_is_identity_only() -> None:
    # Phase 1 must not expose fabricated provider/orchestrator APIs.
    assert llm_orchestrator.__all__ == ["__version__"]


def test_distribution_is_installed() -> None:
    # Confirms `pip install -e .` registered the distribution rather than the
    # tests merely importing a stray source directory.
    meta = metadata(DISTRIBUTION_NAME)
    assert meta["Name"] == DISTRIBUTION_NAME
    assert ">=3.11" in (meta["Requires-Python"] or "")
