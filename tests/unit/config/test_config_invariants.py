"""Architectural invariants for the configuration layer.

Configuration owns validated settings, environment mapping, and provider
construction — nothing else. These tests fail if ``config/`` ever reads the
environment at import time, initializes SDK clients at import time, keeps a
global settings instance, reimplements retry/fallback/routing control flow,
pulls in orchestration frameworks, or weakens the provider-SDK boundary.

Checks operate on parsed code (via ``ast``), not raw text, so docstrings that
merely *mention* a deferred concept cannot cause false positives. Policy
*types* (``RetryPolicy``, ``FallbackPolicy``, ``StaticRoutingPolicy``) are
legitimate settings fields; retry/fallback/routing *machinery* is not.
"""

from __future__ import annotations

import ast
import os
import subprocess
import sys
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[3] / "src" / "llm_orchestrator"
CONFIG_DIR = PACKAGE_ROOT / "config"
CONFIG_MODULES = sorted(CONFIG_DIR.glob("*.py"))

# Absolute imports that must never appear in config/: vendor SDKs, workflow
# frameworks, retry helpers, transports, and env-file loaders. ``os`` is
# deliberately allowed (the loader reads ``os.environ`` inside functions);
# module-level use is policed separately below.
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
    "dotenv",
    "yaml",
    "tomllib",
    "tomli",
}

# Retry/fallback/routing/orchestration machinery that must never live here.
# Policy *types* are excluded on purpose: settings legitimately reference
# ``RetryPolicy``/``FallbackPolicy``/``StaticRoutingPolicy`` as fields.
FORBIDDEN_CODE_IDENTIFIERS = frozenset(
    {
        "sleep",
        "backoff",
        "jitter",
        "retry_after",
        "RetryEngine",
        "FallbackExecutor",
        "ModelRouter",
        "select_model",
        "estimate_cost",
        "compute_backoff_delay",
        "build_orchestration_graph",
        "Orchestrator",
        "execute_model",
        "validate_structured_output",
        "get_settings",
        "global_settings",
        "_global_settings",
    }
)


def _parse(path: Path) -> ast.Module:
    return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def _imported_absolute_roots(path: Path) -> set[str]:
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


def _code_identifiers(path: Path) -> set[str]:
    """Names and attributes used by real code (docstrings stripped)."""
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
    found: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            found.add(node.id)
        elif isinstance(node, ast.Attribute):
            found.add(node.attr)
    return found


def _generate_call_targets(path: Path) -> list[str]:
    targets: list[str] = []
    for node in ast.walk(_parse(path)):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "generate"
        ):
            targets.append(ast.unparse(node.func.value))
    return targets


class _ScopeVisitor(ast.NodeVisitor):
    """Records whether ``os.environ``/``os.getenv`` is touched outside functions."""

    def __init__(self) -> None:
        self._depth = 0
        self.module_level_env_access = False

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._depth += 1
        self.generic_visit(node)
        self._depth -= 1

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._depth += 1
        self.generic_visit(node)
        self._depth -= 1

    def visit_Attribute(self, node: ast.Attribute) -> None:
        if (
            isinstance(node.value, ast.Name)
            and node.value.id == "os"
            and node.attr in {"environ", "getenv"}
            and self._depth == 0
        ):
            self.module_level_env_access = True
        self.generic_visit(node)


def test_config_package_has_the_expected_modules() -> None:
    found = {path.name for path in CONFIG_MODULES}
    assert found == {"__init__.py", "settings.py", "loader.py"}, (
        f"unexpected config modules: {sorted(found)}"
    )


def test_config_has_no_sdk_framework_or_env_loader_imports() -> None:
    assert CONFIG_MODULES, "expected config modules to exist"
    for path in CONFIG_MODULES:
        leaked = _imported_absolute_roots(path) & FORBIDDEN_IMPORT_ROOTS
        assert not leaked, f"{path.name} imports forbidden modules: {sorted(leaked)}"


def test_config_only_depends_on_allowed_internal_layers() -> None:
    for path in CONFIG_MODULES:
        targets = _relative_import_targets(path)
        assert targets <= {"contracts", "errors", "providers", "routing", "settings", "loader"}, (
            f"{path.name} depends on unexpected modules: {sorted(targets)}"
        )


def test_config_contains_no_retry_fallback_routing_orchestration_machinery() -> None:
    for path in CONFIG_MODULES:
        leaked = _code_identifiers(path) & FORBIDDEN_CODE_IDENTIFIERS
        assert not leaked, f"{path.name} references forbidden machinery: {sorted(leaked)}"


def test_config_never_calls_generate() -> None:
    for path in CONFIG_MODULES:
        assert _generate_call_targets(path) == [], f"{path.name} performs model calls"


def test_config_has_no_module_level_env_access() -> None:
    for path in CONFIG_MODULES:
        visitor = _ScopeVisitor()
        visitor.visit(_parse(path))
        assert not visitor.module_level_env_access, (
            f"{path.name} reads the environment at module level"
        )


def test_config_has_no_singleton_or_module_level_mutable_state() -> None:
    # ``__init__.py`` holds only conventional metadata (``__all__``); the
    # settings/loader modules must not stash request state at module level.
    for path in CONFIG_MODULES:
        if path.name == "__init__.py":
            continue
        for node in ast.walk(_parse(path)):
            if isinstance(node, ast.Assign | ast.AnnAssign) and node.col_offset == 0:
                value = node.value
                assert not isinstance(value, ast.List | ast.Dict | ast.Set | ast.Call), (
                    f"{path.name} has module-level mutable state"
                )
        for node in ast.walk(_parse(path)):
            if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
                assert node.name not in {"get_settings", "get_default_settings"}, (
                    f"{path.name} exposes a global-settings accessor"
                )


def test_config_has_no_loops_that_retry() -> None:
    for path in CONFIG_MODULES:
        for node in ast.walk(_parse(path)):
            assert not isinstance(node, ast.While), (
                f"{path.name} contains a while loop; control flow belongs elsewhere"
            )


def test_importing_config_reads_no_environment_and_needs_no_keys() -> None:
    code = (
        "import llm_orchestrator.config; "
        "from llm_orchestrator.config import OrchestratorSettings; "
        "assert OrchestratorSettings().default_timeout_s is None; "
        "print('config-import-ok')"
    )
    child_env = {
        "PATH": os.environ.get("PATH", ""),
        "SYSTEMROOT": os.environ.get("SYSTEMROOT", ""),
        "LLM_ORCH_DEFAULT_TIMEOUT_S": "not-a-number",
        "GEMINI_API_KEY": "AIzaFakeKeyThatMustNeverBeReadAtImport1234",
    }
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        env=child_env,
    )
    assert result.returncode == 0, result.stderr
    assert "config-import-ok" in result.stdout


def test_importing_config_does_not_import_provider_sdks() -> None:
    code = (
        "import sys, llm_orchestrator.config; "
        "assert 'google.genai' not in sys.modules, 'Gemini SDK imported eagerly'; "
        "assert 'groq' not in sys.modules, 'Groq SDK imported eagerly'"
    )
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
