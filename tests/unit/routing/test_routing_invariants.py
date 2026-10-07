"""Architectural invariants for the routing layer.

The router is a deterministic policy engine: it must stay provider-agnostic,
perform no I/O, and never retry, fall back, or invoke providers. Checks
operate on parsed code (via ``ast``), not raw text, so docstrings that merely
*discuss* retry/fallback cannot cause false positives — only real control
flow counts.
"""

from __future__ import annotations

import ast
import inspect
import re
import subprocess
import sys
from pathlib import Path

from llm_orchestrator.contracts import LLMRequest, Message, MessageRole, ModelConfig
from llm_orchestrator.routing import ModelRouter, StaticRoutingPolicy, select_model
from llm_orchestrator.routing import __all__ as ROUTING_ALL

PACKAGE_ROOT = Path(__file__).resolve().parents[3] / "src" / "llm_orchestrator"
ROUTING_DIR = PACKAGE_ROOT / "routing"
ROUTING_MODULES = sorted(ROUTING_DIR.glob("*.py"))

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
    "asyncio",
    "socket",
    "ssl",
    "os",
    "sys",
    "random",
    "time",
    "dotenv",
}

# Concrete vendors must never be named in routing source, not even in examples.
VENDOR_PATTERN = re.compile(r"\b(gemini|groq|openai|anthropic|mistral|cohere)\b")
APP_PATTERN = re.compile(r"\breviewiq\b")

# Control flow that belongs to retry/fallback/provider layers, never routing.
# Matched against code with docstrings stripped (string literals still count).
SEPARATION_PATTERN = re.compile(
    r"\b(retry|fallback|generate|aclose|sleep|shuffle|choice|perf_counter)\b"
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
    targets: set[str] = set()
    for node in ast.walk(_parse(path)):
        if isinstance(node, ast.ImportFrom) and node.level > 0 and node.module:
            targets.add(node.module.split(".")[0])
    return targets


def _code_only(path: Path) -> str:
    """Module code with docstrings removed, then unparsed and lowered."""
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


def _request() -> LLMRequest:
    return LLMRequest(messages=[Message(role=MessageRole.USER, content="hi")])


def test_routing_package_has_expected_modules() -> None:
    assert {path.name for path in ROUTING_MODULES} == {"__init__.py", "router.py", "static.py"}


def test_routing_public_api_is_minimal() -> None:
    assert sorted(ROUTING_ALL) == [
        "ModelRouter",
        "RoutingDecision",
        "SelectionReason",
        "StaticRoutingPolicy",
        "estimate_cost",
        "select_model",
    ]


def test_routing_has_no_sdk_framework_or_io_imports() -> None:
    assert ROUTING_MODULES, "expected routing modules to exist"
    for path in ROUTING_MODULES:
        leaked = _imported_module_roots(path) & FORBIDDEN_IMPORT_ROOTS
        assert not leaked, f"{path.name} imports forbidden modules: {sorted(leaked)}"


def test_routing_only_depends_on_contracts_and_errors() -> None:
    for path in ROUTING_MODULES:
        if path.name == "__init__.py":
            continue
        targets = _relative_import_targets(path)
        assert targets <= {"contracts", "errors", "static", "router"}, (
            f"{path.name} depends on unexpected modules: {sorted(targets)}"
        )


def test_routing_names_no_concrete_vendor_or_application() -> None:
    for path in ROUTING_MODULES:
        text = path.read_text(encoding="utf-8").lower()
        vendor = VENDOR_PATTERN.search(text)
        assert vendor is None, f"{path.name} names concrete vendor {vendor.group(0)!r}"
        app = APP_PATTERN.search(text)
        assert app is None, f"{path.name} references application {app.group(0)!r}"


def test_routing_code_performs_no_retry_fallback_or_provider_calls() -> None:
    for path in ROUTING_MODULES:
        if path.name == "__init__.py":
            continue
        match = SEPARATION_PATTERN.search(_code_only(path))
        assert match is None, f"{path.name} references deferred concept {match.group(0)!r}"


def test_routing_contains_no_await_expressions() -> None:
    for path in ROUTING_MODULES:
        if path.name == "__init__.py":
            continue
        awaits = [node for node in ast.walk(_parse(path)) if isinstance(node, ast.Await)]
        assert not awaits, f"{path.name} awaits something; routing must be synchronous"


def test_selection_entry_points_are_synchronous() -> None:
    assert not inspect.iscoroutinefunction(ModelRouter.select)
    assert not inspect.iscoroutinefunction(ModelRouter.explain)
    assert not inspect.iscoroutinefunction(select_model)
    assert not inspect.iscoroutinefunction(StaticRoutingPolicy.select)
    assert not inspect.iscoroutinefunction(StaticRoutingPolicy.ordered)


def test_router_accepts_no_provider_or_registry() -> None:
    """Routing inputs are request + candidates + policy; providers can never be invoked."""
    params = set(inspect.signature(ModelRouter.select).parameters)
    assert params == {"self", "request", "candidates", "policy", "context"}
    params = set(inspect.signature(select_model).parameters)
    assert params == {"request", "candidates", "policy", "context"}


def test_importing_routing_does_not_import_sdks_or_frameworks() -> None:
    code = (
        "import sys, llm_orchestrator.routing; "
        "assert 'google.genai' not in sys.modules, 'Gemini SDK imported eagerly'; "
        "assert 'groq' not in sys.modules, 'Groq SDK imported eagerly'; "
        "assert 'langgraph' not in sys.modules, 'LangGraph imported eagerly'"
    )
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


# --- selection properties (no new dependency; plain loops) ---


def test_select_returns_member_of_candidate_set() -> None:
    candidates = [
        ModelConfig(provider="b", model="m2"),
        ModelConfig(provider="a", model="m1", quality_tier=4),
        ModelConfig(provider="c", model="m3"),
    ]
    for policy in (
        StaticRoutingPolicy(),
        StaticRoutingPolicy(preferred_provider="c"),
        StaticRoutingPolicy(preferred_model="m2"),
    ):
        selected = select_model(_request(), candidates, policy)
        assert any(selected is member for member in candidates)


def test_selection_is_deterministic_across_repeated_calls() -> None:
    candidates = [
        ModelConfig(provider="b", model="m2", quality_tier=1),
        ModelConfig(provider="a", model="m1", quality_tier=1),
    ]
    policy = StaticRoutingPolicy(preferred_provider="b")
    first = select_model(_request(), candidates, policy)
    assert all(select_model(_request(), candidates, policy) == first for _ in range(10))


def test_adding_ineligible_candidates_changes_nothing() -> None:
    eligible = [ModelConfig(provider="b", model="m2"), ModelConfig(provider="a", model="m1")]
    ineligible = [
        ModelConfig(provider="z", model="zzz", enabled=False),
        ModelConfig(provider="z", model="zzz", quality_tier=999, enabled=False),
    ]
    policy = StaticRoutingPolicy()
    assert select_model(_request(), [*eligible, *ineligible], policy) == select_model(
        _request(), eligible, policy
    )
