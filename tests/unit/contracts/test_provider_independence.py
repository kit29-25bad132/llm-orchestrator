"""Architectural invariants for the contract layer.

These tests fail if the ``contracts``/``errors`` packages ever couple to a
concrete provider SDK, an orchestration framework, a web framework, a database,
or any application-specific code. Imports are inspected structurally (via
``ast``) rather than by fragile text matching.

The scan is scoped to the contract/error layers on purpose: later phases'
provider adapters may legitimately import provider SDKs, but nothing here ever
may.
"""

from __future__ import annotations

import ast
import re
from importlib.metadata import requires
from pathlib import Path

import pytest

DISTRIBUTION_NAME = "llm-orchestrator"
PACKAGE_ROOT = Path(__file__).resolve().parents[3] / "src" / "llm_orchestrator"
CONTRACT_LAYER_DIRS = (PACKAGE_ROOT / "contracts", PACKAGE_ROOT / "errors")

# Import roots that must never appear in the contract/error layers.
FORBIDDEN_IMPORT_ROOTS = {
    "google",
    "groq",
    "openai",
    "anthropic",
    "mistralai",
    "cohere",
    "langgraph",
    "langchain",
    "langchain_core",
    "fastapi",
    "starlette",
    "flask",
    "django",
    "supabase",
    "httpx",
    "requests",
    "aiohttp",
    "urllib3",
    "sqlalchemy",
    "redis",
    "boto3",
}

# Application coupling that must never appear.
FORBIDDEN_NAME_FRAGMENTS = ("reviewiq",)

# The contract layer is pure data: no I/O, no concurrency, no environment access.
FORBIDDEN_IO_ROOTS = {"socket", "ssl", "asyncio", "pathlib", "os", "shutil", "tempfile"}

SOURCE_FILES = sorted(path for directory in CONTRACT_LAYER_DIRS for path in directory.rglob("*.py"))


def _imported_module_roots(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    roots: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            roots.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            roots.add(node.module.split(".")[0])
    return roots


def test_contract_layer_source_is_present() -> None:
    assert SOURCE_FILES, "expected contracts/ and errors/ source files to exist"


@pytest.mark.parametrize("path", SOURCE_FILES, ids=lambda p: p.name)
def test_no_provider_or_framework_imports(path: Path) -> None:
    leaked = _imported_module_roots(path) & FORBIDDEN_IMPORT_ROOTS
    assert not leaked, f"{path.name} imports forbidden modules: {sorted(leaked)}"


@pytest.mark.parametrize("path", SOURCE_FILES, ids=lambda p: p.name)
def test_no_io_or_environment_access(path: Path) -> None:
    leaked = _imported_module_roots(path) & FORBIDDEN_IO_ROOTS
    assert not leaked, f"{path.name} performs I/O via {sorted(leaked)}"


def test_declared_runtime_dependencies_are_minimal() -> None:
    declared = requires(DISTRIBUTION_NAME) or []
    runtime: set[str] = set()
    for requirement in declared:
        if "extra ==" in requirement or "extra==" in requirement:
            continue
        name = re.split(r"[<>=!~;\[\s]", requirement, maxsplit=1)[0].strip().lower()
        runtime.add(name)
    # google-genai is the Phase 4 Gemini adapter dependency and groq is the
    # Phase 5 Groq adapter dependency; the contract layer itself never imports
    # either (asserted by the import scan above).
    assert runtime == {"pydantic", "google-genai", "groq"}, (
        f"unexpected runtime dependencies: {sorted(runtime)}"
    )


def test_no_application_specific_names_in_contract_layer() -> None:
    for path in SOURCE_FILES:
        text = path.read_text(encoding="utf-8").lower()
        for fragment in FORBIDDEN_NAME_FRAGMENTS:
            assert fragment not in text, f"{path.name} references application '{fragment}'"
