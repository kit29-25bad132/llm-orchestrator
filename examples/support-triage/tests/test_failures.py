"""Offline failure-path tests: credentials, config errors, extras, Orchestrator, hygiene.

Every failure asserted here comes from the library's documented public
behaviour; the tests never reach into private modules.
"""

from __future__ import annotations

import ast
import asyncio
import importlib.util
from collections.abc import Awaitable
from pathlib import Path
from typing import TypeVar

import pytest
from triage.app import triage_ticket_single
from triage.fakes import ScriptedProvider, classification_payload, make_response

from llm_orchestrator.config import OrchestratorSettings, build_registry
from llm_orchestrator.contracts import FallbackPolicy, ModelConfig, ProviderConfig, RetryPolicy
from llm_orchestrator.errors import (
    ConfigurationError,
    LLMError,
    ProviderNotRegisteredError,
    StructuredOutputValidationError,
)
from llm_orchestrator.providers import ProviderRegistry
from llm_orchestrator.routing import StaticRoutingPolicy

T = TypeVar("T")


def _has_module(name: str) -> bool:
    """``find_spec`` on a dotted name re-imports the parent package, which can
    itself raise ``ModuleNotFoundError`` when the parent is absent entirely."""
    try:
        return importlib.util.find_spec(name) is not None
    except ModuleNotFoundError:
        return False


_HAS_GROQ = _has_module("groq")
_HAS_GEMINI = _has_module("google.genai")
_HAS_LANGGRAPH = _has_module("langgraph")


def run(awaitable: Awaitable[T]) -> T:
    """Drive a coroutine to completion without an async test plugin."""
    return asyncio.run(awaitable)


def offline_settings() -> OrchestratorSettings:
    """Groq-preferred, Gemini-fallback settings with fast, seeded retries."""
    return OrchestratorSettings(
        models=(
            ModelConfig(provider="groq", model="groq-triage-model"),
            ModelConfig(provider="gemini", model="gemini-triage-model"),
        ),
        retry=RetryPolicy(max_attempts=3, initial_backoff_s=0.05),
        fallback=FallbackPolicy(max_fallbacks=1, cross_provider=True),
        routing=StaticRoutingPolicy(preferred_provider="groq"),
    )


@pytest.mark.skipif(
    not (_HAS_GROQ and _HAS_GEMINI), reason="needs the groq and gemini SDKs ([all] extra)"
)
def test_missing_credential_error_is_actionable_and_never_echoes_the_secret() -> None:
    settings = OrchestratorSettings(
        providers=(
            ProviderConfig(name="groq", credential_env_var="GROQ_API_KEY"),
            ProviderConfig(name="gemini", credential_env_var="GEMINI_API_KEY"),
        ),
    )
    sentinel = "gsk_sentinel_do_not_leak_123456789"

    # env, when supplied, is used exclusively (loader docstring) -> deterministic.
    with pytest.raises(ConfigurationError) as exc_info:
        build_registry(settings, env={"GROQ_API_KEY": sentinel})
    message = str(exc_info.value)
    assert "gemini" in message
    assert "GEMINI_API_KEY" in message
    assert sentinel not in message

    with pytest.raises(ConfigurationError) as exc_info:
        build_registry(settings, env={})
    assert "groq" in str(exc_info.value)
    assert "GROQ_API_KEY" in str(exc_info.value)


def test_unsupported_provider_fails_fast_listing_supported_ones() -> None:
    settings = OrchestratorSettings(
        providers=(ProviderConfig(name="openai", credential_env_var="OPENAI_API_KEY"),)
    )
    with pytest.raises(ConfigurationError) as exc_info:
        build_registry(settings, env={})
    message = str(exc_info.value)
    assert "openai" in message
    assert "gemini" in message
    assert "groq" in message


def test_unknown_provider_lookup_raises_public_error() -> None:
    registry = ProviderRegistry()
    with pytest.raises(ProviderNotRegisteredError) as exc_info:
        registry.get("openai")
    assert isinstance(exc_info.value, LLMError)
    assert "openai" in str(exc_info.value)


@pytest.mark.skipif(_HAS_GROQ, reason="groq extra is installed in this environment")
def test_missing_groq_extra_names_the_install_command() -> None:
    with pytest.raises(ConfigurationError) as exc_info:
        __import__("llm_orchestrator.providers.groq")
    message = str(exc_info.value)
    assert "[groq]" in message
    assert "pip install" in message


@pytest.mark.skipif(_HAS_LANGGRAPH, reason="langgraph extra is installed in this environment")
def test_missing_langgraph_extra_names_the_install_command() -> None:
    with pytest.raises(ConfigurationError) as exc_info:
        __import__("llm_orchestrator.orchestration")
    message = str(exc_info.value)
    assert "[langgraph]" in message
    assert "pip install" in message


@pytest.mark.skipif(not _HAS_LANGGRAPH, reason="langgraph extra ([all]) required")
def test_offline_orchestrator_run_classifies_ticket() -> None:
    groq = ScriptedProvider(
        "groq", [make_response("groq", "groq-triage-model", classification_payload())]
    )
    registry = ProviderRegistry()
    registry.register(groq)
    classification = run(
        triage_ticket_single(
            "Please reset my password", registry=registry, settings=offline_settings()
        )
    )
    assert classification.category == "billing"
    assert [call.model for call in groq.calls] == ["groq-triage-model"]


def test_consumer_package_imports_only_documented_public_modules() -> None:
    allowed = {
        "llm_orchestrator",
        "llm_orchestrator.config",
        "llm_orchestrator.contracts",
        "llm_orchestrator.errors",
        "llm_orchestrator.orchestration",
        "llm_orchestrator.providers",
        "llm_orchestrator.providers.gemini",
        "llm_orchestrator.providers.groq",
        "llm_orchestrator.reliability",
        "llm_orchestrator.routing",
        "llm_orchestrator.structured",
    }
    package_dir = Path(__file__).resolve().parents[1] / "triage"
    for path in sorted(package_dir.glob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                if node.level:  # relative import within the consumer package
                    continue
                names = [node.module or ""]
            else:
                continue
            for name in names:
                if name == "llm_orchestrator" or name.startswith("llm_orchestrator."):
                    assert name in allowed, f"{path.name} imports a non-public module {name!r}"
                root = name.split(".")[0]
                assert root not in {"support", "tests"}, (
                    f"{path.name} imports test-support module {name!r}"
                )


def test_invalid_structured_output_is_importable_for_the_cli_mapping() -> None:
    # The CLI maps this class to exit code 3; keep the contract visible here too.
    assert issubclass(StructuredOutputValidationError, LLMError)
