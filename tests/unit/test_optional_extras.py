"""Optional-dependency behavior: lazy imports and actionable missing-extra errors.

Phase 14 moved the provider SDKs and LangGraph into optional extras
(``[gemini]``, ``[groq]``, ``[langgraph]``, ``[all]``). These tests prove,
offline and without credentials, that:

- every core subpackage imports with all optional dependencies unavailable;
- importing an adapter or the orchestration package without its extra raises
  the library's ``ConfigurationError`` naming the extra — never an opaque
  ``ImportError``;
- ``build_registry`` constructs only the configured provider's adapter, so a
  consumer who installs a single extra can build a registry for it.

Absence is simulated in a child interpreter by a ``sys.meta_path`` finder that
hides the optional import roots; the parent environment is untouched.
"""

from __future__ import annotations

import os
import subprocess
import sys

# Synthetic credentials for offline registry construction only. They are never
# used against a live API (no model call happens) and match the fake-key
# convention used by the configuration invariants.
_FAKE_GEMINI_KEY = "AIzaFakeKeyForOfflineRegistryBuild012345"
_FAKE_GROQ_KEY = "gsk_FakeKeyForOfflineRegistryBuild012345"

_CORE_IMPORT_CODE = """
import sys

import llm_orchestrator
import llm_orchestrator.config
import llm_orchestrator.contracts
import llm_orchestrator.errors
import llm_orchestrator.providers
import llm_orchestrator.reliability
import llm_orchestrator.routing
import llm_orchestrator.structured

for root in ("google", "groq", "langgraph"):
    assert root not in sys.modules, f"{root} imported by a core path"
print("core-imports-ok")
"""


def _run_child(
    code: str, *, block: frozenset[str] = frozenset()
) -> subprocess.CompletedProcess[str]:
    """Run ``code`` in a child interpreter with ``block`` roots unimportable."""
    prelude = (
        "import sys\n"
        f"_OPTIONAL_ROOTS = {set(block)!r}\n"
        "class _BlockOptional:\n"
        "    def find_spec(self, fullname, path=None, target=None):\n"
        "        if fullname.split('.')[0] in _OPTIONAL_ROOTS:\n"
        "            raise ModuleNotFoundError(f'No module named {fullname!r}', name=fullname)\n"
        "        return None\n"
        "sys.meta_path.insert(0, _BlockOptional())\n"
    )
    env = {
        "PATH": os.environ.get("PATH", ""),
        "SYSTEMROOT": os.environ.get("SYSTEMROOT", ""),
    }
    return subprocess.run(
        [sys.executable, "-c", prelude + code],
        capture_output=True,
        text=True,
        env=env,
    )


def test_core_subpackages_import_without_any_optional_dependency() -> None:
    result = _run_child(_CORE_IMPORT_CODE, block=frozenset({"google", "groq", "langgraph"}))
    assert result.returncode == 0, result.stderr
    assert "core-imports-ok" in result.stdout


def _expect_missing_extra(module: str, extra: str, block: frozenset[str]) -> None:
    code = f"""
try:
    import {module}
except Exception as exc:
    from llm_orchestrator.errors import ConfigurationError

    assert isinstance(exc, ConfigurationError), (
        f"expected ConfigurationError, got {{type(exc).__name__}}: {{exc}}"
    )
    assert "[{extra}]" in str(exc), f"message must name the extra: {{exc}}"
    assert "pip install" in str(exc), f"message must include an install command: {{exc}}"
    print("missing-extra-ok")
else:
    raise AssertionError("import unexpectedly succeeded without its extra")
"""
    result = _run_child(code, block=block)
    assert result.returncode == 0, result.stderr
    assert "missing-extra-ok" in result.stdout


def test_gemini_adapter_without_extra_raises_actionable_error() -> None:
    _expect_missing_extra("llm_orchestrator.providers.gemini", "gemini", frozenset({"google"}))


def test_groq_adapter_without_extra_raises_actionable_error() -> None:
    _expect_missing_extra("llm_orchestrator.providers.groq", "groq", frozenset({"groq"}))


def test_orchestration_without_extra_raises_actionable_error() -> None:
    _expect_missing_extra("llm_orchestrator.orchestration", "langgraph", frozenset({"langgraph"}))


def test_build_registry_gemini_only_never_imports_the_groq_adapter() -> None:
    # A consumer with only [gemini] installed must be able to build a Gemini
    # registry: the Groq adapter module must never be imported on that path.
    code = """
import sys

from llm_orchestrator.config import OrchestratorSettings, build_registry
from llm_orchestrator.contracts import ModelConfig, ProviderConfig

settings = OrchestratorSettings(
    providers=(ProviderConfig(name="gemini", credential_env_var="GEMINI_API_KEY"),),
    models=(ModelConfig(provider="gemini", model="test-model"),),
)
registry = build_registry(
    settings,
    env={"GEMINI_API_KEY": "AIzaFakeKeyForOfflineRegistryBuild012345"},
    client_factory=lambda name: object(),
)
assert registry.get("gemini") is not None
assert "llm_orchestrator.providers.groq" not in sys.modules, (
    "the Groq adapter was imported while building a Gemini-only registry"
)
print("gemini-only-ok")
"""
    result = _run_child(code, block=frozenset({"groq"}))
    assert result.returncode == 0, result.stderr
    assert "gemini-only-ok" in result.stdout


def test_build_registry_configured_groq_without_extra_is_actionable() -> None:
    code = """
from llm_orchestrator.config import OrchestratorSettings, build_registry
from llm_orchestrator.contracts import ModelConfig, ProviderConfig

settings = OrchestratorSettings(
    providers=(ProviderConfig(name="groq", credential_env_var="GROQ_API_KEY"),),
    models=(ModelConfig(provider="groq", model="test-model"),),
)
try:
    build_registry(
        settings,
        env={"GROQ_API_KEY": "gsk_FakeKeyForOfflineRegistryBuild012345"},
        client_factory=lambda name: object(),
    )
except Exception as exc:
    from llm_orchestrator.errors import ConfigurationError

    assert isinstance(exc, ConfigurationError), (
        f"expected ConfigurationError, got {type(exc).__name__}: {exc}"
    )
    assert "[groq]" in str(exc), f"message must name the extra: {exc}"
    print("groq-missing-extra-ok")
else:
    raise AssertionError("build_registry unexpectedly succeeded without the groq extra")
"""
    result = _run_child(code, block=frozenset({"groq"}))
    assert result.returncode == 0, result.stderr
    assert "groq-missing-extra-ok" in result.stdout


def test_provider_modules_import_normally_when_their_extra_is_present() -> None:
    # Positive control for the import guards: with extras installed (the dev
    # environment), the guarded modules still import and expose their public
    # names — the guards never break the normal path.
    from llm_orchestrator.orchestration import Orchestrator
    from llm_orchestrator.providers.gemini import GeminiProvider
    from llm_orchestrator.providers.groq import GroqProvider

    assert GeminiProvider.__name__ == "GeminiProvider"
    assert GroqProvider.__name__ == "GroqProvider"
    assert Orchestrator.__name__ == "Orchestrator"
