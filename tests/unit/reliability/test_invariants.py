"""Architectural invariants for the reliability layer.

The retry engine must stay provider-agnostic: it may consume the generic
provider interface, contracts, and normalized errors, but it must never import
a vendor SDK, branch on a concrete provider, pull in orchestration frameworks,
or anticipate fallback/routing. The fallback executor (Phase 11) owns
candidate transitions and may name fallback concepts, but it must never route,
retry, or touch vendors/frameworks itself.

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

# Fallback transitions belong to fallback.py only; nothing else may name them.
FALLBACK_CONCEPT_PATTERN = re.compile(r"\bfallback\b")

# Concerns that must not leak into any reliability module.
DEFERRED_CONCEPTS_PATTERN = re.compile(r"\b(rout\w*|langgraph|reviewiq)\b")


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


def test_reliability_package_exists_with_retry_backoff_and_fallback() -> None:
    found = {path.name for path in RELIABILITY_MODULES}
    assert found == {"__init__.py", "backoff.py", "retry.py", "fallback.py"}, (
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


def test_reliability_contains_no_routing_or_app_concepts() -> None:
    for path in RELIABILITY_MODULES:
        match = DEFERRED_CONCEPTS_PATTERN.search(_code_only(path))
        assert match is None, f"{path.name} references deferred concept {match.group(0)!r}"


def test_only_fallback_module_names_fallback() -> None:
    for path in RELIABILITY_MODULES:
        if path.name in {"__init__.py", "fallback.py"}:
            continue
        match = FALLBACK_CONCEPT_PATTERN.search(_code_only(path))
        assert match is None, f"{path.name} anticipates fallback {match.group(0)!r}"


def test_reliability_has_no_module_level_mutable_state() -> None:
    for name, path in (
        ("retry", RELIABILITY_DIR / "retry.py"),
        ("backoff", RELIABILITY_DIR / "backoff.py"),
        ("fallback", RELIABILITY_DIR / "fallback.py"),
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


FALLBACK_PATH = RELIABILITY_DIR / "fallback.py"

# Retry machinery that must live only in retry.py/backoff.py, never in fallback.py.
RETRY_CONCEPTS = ("sleep", "backoff", "jitter", "retry_after", "tenacity")


def _generate_call_targets(path: Path) -> list[str]:
    """Receivers of every ``.generate(...)`` call in ``path`` (unparsed)."""
    targets: list[str] = []
    for node in ast.walk(_parse(path)):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "generate"
        ):
            targets.append(ast.unparse(node.func.value))
    return targets


def test_fallback_owns_no_retry_machinery() -> None:
    """Fallback must call ``RetryEngine.generate`` instead of reimplementing retry."""
    code = _code_only(FALLBACK_PATH)
    for concept in RETRY_CONCEPTS:
        assert concept not in code, f"fallback.py reimplements retry via {concept!r}"
    loops = [node for node in ast.walk(_parse(FALLBACK_PATH)) if isinstance(node, ast.While)]
    assert not loops, "fallback.py must not run its own attempt loop"


def test_fallback_calls_retry_engine_and_nothing_else() -> None:
    """Exactly one retry entry point plus the recording pass-through; no SDK calls."""
    assert sorted(_generate_call_targets(FALLBACK_PATH)) == [
        "self._inner",
        "self._retry_engine",
    ]


def test_fallback_defines_no_candidate_selection() -> None:
    """Fallback consumes routing order; it must not reselect like a router."""
    defined = {
        node.name
        for node in ast.walk(_parse(FALLBACK_PATH))
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef)
    }
    assert "select" not in defined, "fallback.py must not duplicate ModelRouter.select"
    assert "generate" in defined, "the recording wrapper must define generate"


def test_fallback_catches_no_broad_exceptions() -> None:
    """Only normalized ``LLMError`` (and explicit cancellation) may be caught."""
    for node in ast.walk(_parse(FALLBACK_PATH)):
        if not isinstance(node, ast.ExceptHandler):
            continue
        assert node.type is not None, "fallback.py must not contain a bare except"
        name = ast.unparse(node.type)
        assert name not in {"Exception", "BaseException"}, (
            f"fallback.py catches broad {name} and would swallow bugs/cancellation"
        )
