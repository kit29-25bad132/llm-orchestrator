"""Architectural invariants for the structured-output layer.

Validation must stay provider-agnostic, dependency-light, and pure local CPU
work: it may consume domain contracts, normalized errors, stdlib, and
Pydantic — nothing else. In particular it must never import a vendor SDK,
invoke providers, retry, sleep, touch the network, or evaluate payloads.

Checks operate on parsed code (via ``ast``), not raw text, so docstrings that
merely *mention* a deferred concept cannot cause false positives.
"""

from __future__ import annotations

import ast
import re
import subprocess
import sys
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[3] / "src" / "llm_orchestrator"
STRUCTURED_DIR = PACKAGE_ROOT / "structured"
STRUCTURED_MODULES = sorted(STRUCTURED_DIR.glob("*.py"))

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

# Concrete providers the validation layer must never special-case.
CONCRETE_PROVIDER_PATTERN = re.compile(r"\b(gemini|groq|openai|anthropic|mistral)\b")

# Later-phase concerns that must not leak into validation code. (Bare words
# like "structured"/"validation" are this layer's own legitimate vocabulary
# and are therefore not part of this pattern.)
DEFERRED_CONCEPTS_PATTERN = re.compile(
    r"\b(fallback|rout\w*|reviewiq|langgraph|agent|repair|reprompt|native)\b"
)

# Calls that would execute untrusted payloads or import machinery.
UNSAFE_CALLS = {"eval", "exec", "compile", "__import__"}


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


def _called_names(path: Path) -> set[str]:
    """Bare function names invoked anywhere in the module."""
    return {
        node.func.id
        for node in ast.walk(_parse(path))
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }


def test_structured_package_has_the_expected_modules() -> None:
    found = {path.name for path in STRUCTURED_MODULES}
    assert found == {"__init__.py", "pydantic_adapter.py"}, (
        f"unexpected structured modules: {sorted(found)}"
    )


def test_structured_has_no_sdk_framework_or_env_imports() -> None:
    assert STRUCTURED_MODULES, "expected structured modules to exist"
    for path in STRUCTURED_MODULES:
        leaked = _imported_module_roots(path) & FORBIDDEN_IMPORT_ROOTS
        assert not leaked, f"{path.name} imports forbidden modules: {sorted(leaked)}"


def test_structured_only_depends_on_contracts_and_errors() -> None:
    for path in STRUCTURED_MODULES:
        if path.name == "__init__.py":
            continue
        targets = _relative_import_targets(path)
        assert targets <= {"contracts", "errors"}, (
            f"{path.name} depends on unexpected modules: {sorted(targets)}"
        )


def test_structured_does_not_branch_on_concrete_providers() -> None:
    for path in STRUCTURED_MODULES:
        match = CONCRETE_PROVIDER_PATTERN.search(_code_only(path))
        assert match is None, f"{path.name} special-cases provider {match.group(0)!r}"


def test_structured_contains_no_deferred_concepts() -> None:
    for path in STRUCTURED_MODULES:
        match = DEFERRED_CONCEPTS_PATTERN.search(_code_only(path))
        assert match is None, f"{path.name} references deferred concept {match.group(0)!r}"


def test_structured_never_evaluates_payloads() -> None:
    for path in STRUCTURED_MODULES:
        leaked = _called_names(path) & UNSAFE_CALLS
        assert not leaked, f"{path.name} calls unsafe builtins: {sorted(leaked)}"


def test_structured_is_pure_local_work() -> None:
    for path in STRUCTURED_MODULES:
        tree = _parse(path)
        assert not any(isinstance(node, ast.For | ast.While) for node in ast.walk(tree)), (
            f"{path.name} contains a loop; validation must be straight-line"
        )
        code = _code_only(path)
        assert "sleep" not in code, f"{path.name} sleeps; validation must not wait"
        assert "retry" not in code, f"{path.name} retries; validation must not retry"


def _is_immutable_binding(value: ast.expr) -> bool:
    """Whether a module-level assignment is an immutable binding.

    Only ``TypeVar(...)`` (typing) and ``re.compile(...)`` (frozen pattern)
    calls are acceptable; anything else constructed at import time is
    suspect shared state.
    """
    if not isinstance(value, ast.Call):
        return True
    func = value.func
    if isinstance(func, ast.Name) and func.id == "TypeVar":
        return True
    return isinstance(func, ast.Attribute) and func.attr == "compile"


def test_structured_has_no_module_level_mutable_state() -> None:
    for path in STRUCTURED_MODULES:
        if path.name == "__init__.py":
            continue
        for node in ast.walk(_parse(path)):
            if isinstance(node, ast.Assign | ast.AnnAssign) and node.col_offset == 0:
                value = node.value
                assert not isinstance(value, ast.List | ast.Dict | ast.Set), (
                    f"{path.name} has module-level mutable state"
                )
                assert not isinstance(value, ast.Call) or _is_immutable_binding(value), (
                    f"{path.name} constructs shared state at import time"
                )


def test_importing_structured_does_not_import_provider_sdks() -> None:
    code = (
        "import sys, llm_orchestrator.structured; "
        "assert 'google.genai' not in sys.modules, 'Gemini SDK imported eagerly'; "
        "assert 'groq' not in sys.modules, 'Groq SDK imported eagerly'"
    )
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
