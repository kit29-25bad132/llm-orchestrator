"""Architectural invariants for the orchestration layer.

LangGraph owns workflow state and transitions here — nothing else. These
tests fail if orchestration ever imports a provider SDK, reimplements retry
logic, anticipates fallback/routing/validation/observability, touches the
environment, references an application, or keeps module-level mutable state.

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
ORCHESTRATION_DIR = PACKAGE_ROOT / "orchestration"
ORCHESTRATION_MODULES = sorted(ORCHESTRATION_DIR.glob("*.py"))

# Only langgraph is allowed here: no vendor SDK, no other framework, no env.
FORBIDDEN_IMPORT_ROOTS = {
    "google",
    "groq",
    "openai",
    "anthropic",
    "mistralai",
    "cohere",
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

# Concrete providers the graph must never special-case.
CONCRETE_PROVIDER_PATTERN = re.compile(r"\b(gemini|groq|openai|anthropic|mistral)\b")

# Later-phase concerns that must not leak into orchestration code.
DEFERRED_CONCEPTS_PATTERN = re.compile(
    r"\b(fallback|rout\w*|structured|validat\w*|observ\w*|telemetry|metric|tracing"
    r"|reviewiq|agent|tool_loop|react)\b"
)


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


def test_orchestration_package_has_the_expected_modules() -> None:
    found = {path.name for path in ORCHESTRATION_MODULES}
    assert found == {"__init__.py", "state.py", "nodes.py", "graph.py"}, (
        f"unexpected orchestration modules: {sorted(found)}"
    )


def test_orchestration_has_no_sdk_framework_or_env_imports() -> None:
    assert ORCHESTRATION_MODULES, "expected orchestration modules to exist"
    for path in ORCHESTRATION_MODULES:
        leaked = _imported_module_roots(path) & FORBIDDEN_IMPORT_ROOTS
        assert not leaked, f"{path.name} imports forbidden modules: {sorted(leaked)}"


def test_orchestration_uses_langgraph() -> None:
    roots = _imported_module_roots(ORCHESTRATION_DIR / "graph.py")
    assert "langgraph" in roots, "the graph builder must actually use LangGraph"


def test_orchestration_only_depends_on_allowed_packages() -> None:
    for path in ORCHESTRATION_MODULES:
        if path.name == "__init__.py":
            continue
        targets = _relative_import_targets(path)
        assert targets <= {"contracts", "errors", "providers", "reliability", "state", "nodes"}, (
            f"{path.name} depends on unexpected modules: {sorted(targets)}"
        )


def test_orchestration_does_not_branch_on_concrete_providers() -> None:
    for path in ORCHESTRATION_MODULES:
        match = CONCRETE_PROVIDER_PATTERN.search(_code_only(path))
        assert match is None, f"{path.name} special-cases provider {match.group(0)!r}"


def test_orchestration_contains_no_deferred_concepts() -> None:
    for path in ORCHESTRATION_MODULES:
        match = DEFERRED_CONCEPTS_PATTERN.search(_code_only(path))
        assert match is None, f"{path.name} references deferred concept {match.group(0)!r}"


def test_orchestration_has_no_retry_loops_or_sleeps() -> None:
    for path in ORCHESTRATION_MODULES:
        tree = _parse(path)
        assert not any(isinstance(node, ast.For | ast.While) for node in ast.walk(tree)), (
            f"{path.name} contains a loop; retries belong to RetryEngine"
        )
        code = _code_only(path)
        assert "sleep" not in code, f"{path.name} sleeps; waiting belongs to RetryEngine"


def test_orchestration_has_no_module_level_mutable_state() -> None:
    # ``__init__.py`` holds only conventional metadata (``__all__``); the
    # workflow modules must not stash request state at module level.
    for path in (ORCHESTRATION_DIR / name for name in ("state.py", "nodes.py", "graph.py")):
        for node in ast.walk(_parse(path)):
            if isinstance(node, ast.Assign | ast.AnnAssign) and node.col_offset == 0:
                value = node.value
                assert not isinstance(value, ast.List | ast.Dict | ast.Set | ast.Call), (
                    f"{path.name} has module-level mutable state"
                )


# asyncio APIs that would let orchestration manage the event loop, sleep,
# spawn tasks, or enforce its own timeouts. Referencing the CancelledError
# *exception* is fine (cancellation translation); everything else belongs to
# RetryEngine and the providers.
FORBIDDEN_ASYNCIO_APIS = frozenset(
    {
        "sleep",
        "run",
        "run_until_complete",
        "run_forever",
        "wait_for",
        "wait",
        "gather",
        "create_task",
        "ensure_future",
        "to_thread",
        "run_in_executor",
        "shield",
        "timeout",
        "get_event_loop",
        "Task",
        "Event",
        "Lock",
        "Queue",
    }
)


def test_orchestration_never_blocks_sleeps_or_forces_timeouts() -> None:
    for path in ORCHESTRATION_MODULES:
        attrs = {node.attr for node in ast.walk(_parse(path)) if isinstance(node, ast.Attribute)}
        assert not (attrs & FORBIDDEN_ASYNCIO_APIS), (
            f"{path.name} manages the event loop or competes on waiting"
        )
        names = {node.id for node in ast.walk(_parse(path)) if isinstance(node, ast.Name)}
        assert not (names & FORBIDDEN_ASYNCIO_APIS), (
            f"{path.name} manages the event loop or competes on waiting"
        )
        for node in ast.walk(_parse(path)):
            if isinstance(node, ast.ImportFrom) and node.module == "asyncio":
                imported = {alias.name for alias in node.names}
                assert not (imported & FORBIDDEN_ASYNCIO_APIS), (
                    f"{path.name} imports loop-management APIs: {sorted(imported)}"
                )


def test_no_environment_access_in_orchestration_package() -> None:
    for path in ORCHESTRATION_MODULES:
        leaked_imports = _imported_module_roots(path) & {"os", "dotenv"}
        assert not leaked_imports, f"{path.name} imports {sorted(leaked_imports)}"
        attrs = {node.attr for node in ast.walk(_parse(path)) if isinstance(node, ast.Attribute)}
        assert not (attrs & {"getenv", "environ"}), f"{path.name} reads the environment"


def test_importing_orchestration_does_not_import_provider_sdks() -> None:
    code = (
        "import sys, llm_orchestrator.orchestration; "
        "assert 'google.genai' not in sys.modules, 'Gemini SDK imported eagerly'; "
        "assert 'groq' not in sys.modules, 'Groq SDK imported eagerly'"
    )
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
