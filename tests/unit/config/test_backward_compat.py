"""Backward compatibility: existing public constructors work unchanged."""

from __future__ import annotations

from llm_orchestrator.contracts import (
    FallbackPolicy,
    LLMRequest,
    Message,
    MessageRole,
    ModelConfig,
    RetryPolicy,
)
from llm_orchestrator.orchestration import Orchestrator
from llm_orchestrator.providers import ProviderRegistry
from llm_orchestrator.providers.gemini import GeminiProvider
from llm_orchestrator.providers.groq import GroqProvider
from llm_orchestrator.reliability import FallbackExecutor, RetryEngine
from llm_orchestrator.routing import ModelRouter, StaticRoutingPolicy


def test_provider_constructors_unchanged() -> None:
    gemini = GeminiProvider(api_key="secret-test-key-abcdef123456")
    groq = GroqProvider(api_key="secret-test-key-abcdef123456")
    assert gemini.provider_id == "gemini"
    assert groq.provider_id == "groq"
    assert "secret-test-key" not in repr(gemini)
    assert "secret-test-key" not in repr(groq)


def test_engine_router_executor_orchestrator_constructors_unchanged() -> None:
    registry = ProviderRegistry()
    retry_engine = RetryEngine(RetryPolicy(max_attempts=2))
    assert retry_engine.policy.max_attempts == 2
    assert ModelRouter(StaticRoutingPolicy()).policy.preferred_provider is None
    assert FallbackExecutor(registry, retry_engine, FallbackPolicy()).policy.max_fallbacks == 0
    assert Orchestrator(registry, retry_engine).registry is registry


def test_existing_request_model_contracts_unchanged() -> None:
    request = LLMRequest(messages=[Message(role=MessageRole.USER, content="hi")])
    model = ModelConfig(provider="gemini", model="gemini-test-model")
    assert request.timeout_s is None
    assert model.provider == "gemini"
