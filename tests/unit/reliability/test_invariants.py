"""Architectural invariants for the reliability layer.

The retry engine must stay provider-agnostic: it may consume the generic
provider interface, contracts, and normalized errors, but it must never import
a vendor SDK, branch on a concrete provider, pull in orchestration frameworks,
or anticipate fallback/routing (later phases' concerns).

Checks operate on parsed code (via ``ast``), not raw text, so docstrings that
merely *mention* a provider cannot cause false positives.
"""

from __future__ import annotations

import ast
import re
import subprocess
import sys
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[3] / "src" / "llm_orchestrator"
RELIABILITY_DIR = PACKAGE_ROOT / "reliability"
RELIABILITY_MODULES = sorted(RELIABILITY_DIR.glob("*.py"))

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
    "tenacity",
    "backoff",
    "retrying",
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
    "os",
    "dotenv",
}

# Concrete providers the retry layer must never special-case.
CONCRETE_PROVIDER_PATTERN = re.compile(r"\b(gemini|groq|openai|anthropic|mistral)\b")

# Later-phase concerns that must not leak into the retry layer.
DEFERRED_CONCEPTS_PATTERN = re.compile(r"\b(fallback|rout\w*|langgraph|reviewiq)\b")


def _parse(path: Path) -> ast.Module:
    return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def _imported_module_roots(path: Path) -> set[str]:
    roots: set[str] = set()
    for node in ast.walk(_parse(path)):
        if isinstance(node, ast.Import):
            roots.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            roots.add(node.module.split(".")[0])
    return roots


def _relative_import_targets(path: Path) -> set[str]:
    """Top-level packages referenced via relative ``from .`` / ``from ..`` imports."""
    targets: set[str] = set()
    for node in ast.walk(_parse(path)):
        if isinstance(node, ast.ImportFrom) and node.level > 0 and node.module:
            targets.add(node.module.split(".")[0])
    return targets


def _code_only(path: Path) -> str:
    """Return the module's code with docstrings removed, then unparsed."""
    tree = _parse(path)
    for node in ast.walk(tree):
        if isinstance(node, ast.Module | ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef):
            body = node.body
            first = body[0] if body else None
            if (
                isinstance(first, ast.Expr)
                and isinstance(first.value, ast.Constant)
                and isinstance(first.value.value, str)
            ):
                node.body = body[1:] or [ast.Pass()]
    return ast.unparse(tree).lower()


def test_reliability_package_exists_with_retry_and_backoff_only() -> None:
    found = {path.name for path in RELIABILITY_MODULES}
    assert found == {"__init__.py", "backoff.py", "retry.py"}, (
        f"unexpected reliability modules: {sorted(found)}"
    )


def test_reliability_has_no_sdk_framework_or_env_imports() -> None:
    assert RELIABILITY_MODULES, "expected reliability modules to exist"
    for path in RELIABILITY_MODULES:
        leaked = _imported_module_roots(path) & FORBIDDEN_IMPORT_ROOTS
        assert not leaked, f"{path.name} imports forbidden modules: {sorted(leaked)}"


def test_reliability_only_depends_on_contracts_errors_and_providers() -> None:
    for path in RELIABILITY_MODULES:
        if path.name == "__init__.py":
            continue
        targets = _relative_import_targets(path)
        assert targets <= {"contracts", "errors", "providers", "backoff", "retry"}, (
            f"{path.name} depends on unexpected modules: {sorted(targets)}"
        )


def test_reliability_does_not_branch_on_concrete_providers() -> None:
    for path in RELIABILITY_MODULES:
        match = CONCRETE_PROVIDER_PATTERN.search(_code_only(path))
        assert match is None, f"{path.name} special-cases provider {match.group(0)!r}"


def test_reliability_contains_no_fallback_routing_or_app_concepts() -> None:
    for path in RELIABILITY_MODULES:
        match = DEFERRED_CONCEPTS_PATTERN.search(_code_only(path))
        assert match is None, f"{path.name} references deferred concept {match.group(0)!r}"


def test_reliability_has_no_module_level_mutable_state() -> None:
    for name, path in (
        ("retry", RELIABILITY_DIR / "retry.py"),
        ("backoff", RELIABILITY_DIR / "backoff.py"),
    ):
        for node in ast.walk(_parse(path)):
            if isinstance(node, ast.Assign | ast.AnnAssign) and node.col_offset == 0:
                value = node.value
                assert not isinstance(value, ast.List | ast.Dict | ast.Set | ast.Call), (
                    f"{name}.py has module-level mutable state"
                )


def test_importing_reliability_does_not_import_provider_sdks() -> None:
    code = (
        "import sys, llm_orchestrator.reliability; "
        "assert 'google.genai' not in sys.modules, 'Gemini SDK imported eagerly'; "
        "assert 'groq' not in sys.modules, 'Groq SDK imported eagerly'"
    )
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
