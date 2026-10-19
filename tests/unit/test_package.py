"""Package foundation and distribution identity tests.

These verify that ``llm_orchestrator`` is installed, imports cleanly, exposes
the frozen identity-only top-level API, ships its PEP 561 ``py.typed`` marker,
and registers real distribution metadata rather than importing a stray source
directory.
"""

from __future__ import annotations

from importlib.metadata import metadata
from pathlib import Path

import llm_orchestrator

DISTRIBUTION_NAME = "llm-orchestrator"


def test_package_imports() -> None:
    assert llm_orchestrator is not None


def test_version_is_exposed() -> None:
    assert isinstance(llm_orchestrator.__version__, str)
    assert llm_orchestrator.__version__ != ""


def test_public_api_is_identity_only() -> None:
    # The top-level API is frozen: identity only, no provider/orchestrator names.
    assert llm_orchestrator.__all__ == ["__version__"]


def test_distribution_is_installed() -> None:
    # Confirms `pip install -e .` registered the distribution rather than the
    # tests merely importing a stray source directory.
    meta = metadata(DISTRIBUTION_NAME)
    assert meta["Name"] == DISTRIBUTION_NAME
    assert ">=3.11" in (meta["Requires-Python"] or "")


def test_typing_marker_is_shipped() -> None:
    # PEP 561: the `Typing :: Typed` classifier must match the distributable
    # package, which has to carry a `py.typed` marker beside its `__init__`.
    marker = Path(llm_orchestrator.__file__).resolve().parent / "py.typed"
    assert marker.is_file(), "py.typed marker missing next to the installed package"
