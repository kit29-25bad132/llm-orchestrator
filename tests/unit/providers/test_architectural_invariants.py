"""Architectural invariants for the provider abstraction.

These fail if the abstraction layer ever couples to a provider SDK, reads the
environment, branches on a concrete provider, or references an application.

Checks operate on parsed code (via ``ast``), not raw text, so docstrings and
comments that merely *mention* a provider or the environment cannot cause false
positives. Concrete provider modules (``gemini.py``, ``groq.py``) are
deliberately excluded from the SDK-import scan: they may legitimately import
vendor SDKs in later phases. The *abstraction* never may.
"""

from __future__ import annotations

import ast
import re
import subprocess
import sys
from importlib.metadata import requires
from pathlib import Path

import pytest

DISTRIBUTION_NAME = "llm-orchestrator"
PACKAGE_ROOT = Path(__file__).resolve().parents[3] / "src" / "llm_orchestrator"
PROVIDERS_DIR = PACKAGE_ROOT / "providers"

# The abstraction modules that must stay SDK-, framework-, and env-free.
ABSTRACTION_MODULES = (
    PROVIDERS_DIR / "__init__.py",
    PROVIDERS_DIR / "base.py",
    PROVIDERS_DIR / "registry.py",
)

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
    "os",
    "dotenv",
}

# Names of concrete providers that the abstraction must never special-case.
CONCRETE_PROVIDER_NAMES = ("gemini", "groq", "openai", "anthropic", "mistral")
CONCRETE_PROVIDER_PATTERN = re.compile(r"\b(" + "|".join(CONCRETE_PROVIDER_NAMES) + r")\b")

ENVIRONMENT_ACCESS_ATTRS = {"getenv", "environ"}

ALL_SOURCE_FILES = sorted(PACKAGE_ROOT.rglob("*.py"))


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


def _code_only(path: Path) -> str:
    """Return the module's code with docstrings removed, then unparsed.

    Comments are not part of the AST, so this excludes prose entirely while
    preserving identifiers and literal values that real code uses.
    """
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


def _attribute_names(path: Path) -> set[str]:
    return {node.attr for node in ast.walk(_parse(path)) if isinstance(node, ast.Attribute)}


@pytest.mark.parametrize("path", ABSTRACTION_MODULES, ids=lambda p: p.name)
def test_abstraction_has_no_sdk_or_env_imports(path: Path) -> None:
    leaked = _imported_module_roots(path) & FORBIDDEN_IMPORT_ROOTS
    assert not leaked, f"{path.name} imports forbidden modules: {sorted(leaked)}"


@pytest.mark.parametrize("path", ABSTRACTION_MODULES, ids=lambda p: p.name)
def test_abstraction_does_not_branch_on_concrete_providers(path: Path) -> None:
    match = CONCRETE_PROVIDER_PATTERN.search(_code_only(path))
    assert match is None, f"{path.name} special-cases concrete provider {match.group(0)!r}"


def test_no_environment_access_in_providers_package() -> None:
    for path in PROVIDERS_DIR.rglob("*.py"):
        leaked_imports = _imported_module_roots(path) & {"os", "dotenv"}
        assert not leaked_imports, f"{path.name} imports {sorted(leaked_imports)}"
        leaked_attrs = _attribute_names(path) & ENVIRONMENT_ACCESS_ATTRS
        assert not leaked_attrs, f"{path.name} accesses {sorted(leaked_attrs)}"


def test_no_langgraph_anywhere_in_the_library() -> None:
    for path in ALL_SOURCE_FILES:
        roots = _imported_module_roots(path)
        assert not (roots & {"langgraph", "langchain", "langchain_core"}), (
            f"{path.name} imports an orchestration framework"
        )


def test_gemini_and_groq_are_the_approved_concrete_providers() -> None:
    found = {path.name for path in PROVIDERS_DIR.rglob("*.py")} & {
        "gemini.py",
        "groq.py",
        "openai.py",
        "anthropic.py",
        "mistral.py",
    }
    assert found == {"gemini.py", "groq.py"}


def test_only_the_gemini_module_imports_the_sdk() -> None:
    for path in ALL_SOURCE_FILES:
        roots = _imported_module_roots(path)
        if path.name == "gemini.py":
            assert "google" in roots, "the Gemini adapter must import the SDK"
        else:
            assert "google" not in roots, f"{path.name} imports the Gemini SDK outside the adapter"


def test_only_the_groq_module_imports_the_sdk() -> None:
    for path in ALL_SOURCE_FILES:
        roots = _imported_module_roots(path)
        if path.name == "groq.py":
            assert "groq" in roots, "the Groq adapter must import the SDK"
        else:
            assert "groq" not in roots, f"{path.name} imports the Groq SDK outside the adapter"


def test_importing_the_library_does_not_import_provider_sdks() -> None:
    code = (
        "import sys, llm_orchestrator, llm_orchestrator.contracts, "
        "llm_orchestrator.errors, llm_orchestrator.providers; "
        "assert 'google.genai' not in sys.modules, 'Gemini SDK imported eagerly'; "
        "assert 'groq' not in sys.modules, 'Groq SDK imported eagerly'"
    )
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_no_credential_literals_in_source() -> None:
    pattern = re.compile(r"(AIza[0-9A-Za-z_\-]{20,}|sk-[A-Za-z0-9]{20,}|gsk_[A-Za-z0-9]{20,})")
    for path in ALL_SOURCE_FILES:
        assert not pattern.search(path.read_text(encoding="utf-8")), (
            f"{path.name} contains a credential-like literal"
        )


def test_no_application_coupling() -> None:
    for path in ALL_SOURCE_FILES:
        assert "reviewiq" not in _code_only(path), f"{path.name} references the application"


def test_declared_runtime_dependencies_still_minimal() -> None:
    declared = requires(DISTRIBUTION_NAME) or []
    runtime: set[str] = set()
    for requirement in declared:
        if "extra ==" in requirement or "extra==" in requirement:
            continue
        name = re.split(r"[<>=!~;\[\s]", requirement, maxsplit=1)[0].strip().lower()
        runtime.add(name)
    assert runtime == {"pydantic", "google-genai", "groq"}, (
        f"unexpected runtime dependencies: {sorted(runtime)}"
    )
