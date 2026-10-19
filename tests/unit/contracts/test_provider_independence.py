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


def _dependency_sets() -> tuple[set[str], dict[str, set[str]]]:
    """Parse installed metadata into ``(core, {extra: dependency names})``.

    Semantic and order-independent: requirements are compared as sets of
    normalized distribution names, never by list position or specifier text.
    """
    declared = requires(DISTRIBUTION_NAME) or []
    core: set[str] = set()
    by_extra: dict[str, set[str]] = {}
    for requirement in declared:
        name = re.split(r"[<>=!~;\[\s]", requirement, maxsplit=1)[0].strip().lower()
        marker = re.search(r"""extra\s*==\s*['"]([^'"]+)['"]""", requirement)
        if marker is None:
            core.add(name)
        else:
            by_extra.setdefault(marker.group(1), set()).add(name)
    return core, by_extra


def test_core_runtime_dependencies_are_minimal() -> None:
    # Phase 14 packaging contract: the default install needs only Pydantic —
    # provider SDKs and LangGraph are optional extras (asserted below), so a
    # consumer without any extra can still import contracts, errors, the
    # provider abstraction, reliability, routing, structured output, and config.
    core, _ = _dependency_sets()
    assert core == {"pydantic"}, f"unexpected core dependencies: {sorted(core)}"


def test_provider_and_orchestration_dependencies_are_extras() -> None:
    _, by_extra = _dependency_sets()
    # google-genai (Phase 4 adapter), groq (Phase 5 adapter), and langgraph
    # (Phase 8 orchestration) ship as extras; httpx is declared for the two
    # adapters because they import it directly for exception classification.
    assert by_extra.get("gemini") == {"google-genai", "httpx"}
    assert by_extra.get("groq") == {"groq", "httpx"}
    assert by_extra.get("langgraph") == {"langgraph"}


def test_all_extra_covers_every_optional_runtime_dependency() -> None:
    _, by_extra = _dependency_sets()
    optional = by_extra["gemini"] | by_extra["groq"] | by_extra["langgraph"]
    assert by_extra.get("all") == optional, (
        "the 'all' extra must install exactly the union of the optional "
        f"runtime dependencies: {sorted(optional)}"
    )


def test_dev_extra_is_the_development_toolchain_only() -> None:
    _, by_extra = _dependency_sets()
    assert by_extra.get("dev") == {"pytest", "ruff", "mypy", "build"}
    runtime_optionals = {"google-genai", "groq", "langgraph", "httpx"}
    assert not (by_extra["dev"] & runtime_optionals), (
        "dev must stay a toolchain extra; runtime capabilities come from "
        "'all' or the per-feature extras"
    )


def test_no_application_specific_names_in_contract_layer() -> None:
    for path in SOURCE_FILES:
        text = path.read_text(encoding="utf-8").lower()
        for fragment in FORBIDDEN_NAME_FRAGMENTS:
            assert fragment not in text, f"{path.name} references application '{fragment}'"
