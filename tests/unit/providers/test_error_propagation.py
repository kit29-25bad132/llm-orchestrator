"""Normalized errors must survive the provider boundary intact.

A provider raises a Phase 2 ``LLMError`` subtype; the caller must receive that
exact error (same instance, same category, same flags), never a generic
``Exception`` wrapper.
"""

from __future__ import annotations

import asyncio

import pytest

from llm_orchestrator.contracts import LLMRequest, Message, MessageRole, ModelConfig
from llm_orchestrator.errors import (
    AuthenticationError,
    ConfigurationError,
    ErrorCategory,
    LLMError,
    LLMTimeoutError,
    ProviderNotRegisteredError,
    RateLimitError,
)
from llm_orchestrator.providers import ProviderRegistry


def _request() -> LLMRequest:
    return LLMRequest(
        messages=[Message(role=MessageRole.USER, content="hi")],
        request_id="req-1",
    )


def _model() -> ModelConfig:
    return ModelConfig(provider="fake", model="fake-model-1")


@pytest.mark.parametrize(
    "error",
    [
        RateLimitError("throttled", retry_after_s=3.0),
        AuthenticationError("bad key"),
        LLMTimeoutError("too slow"),
    ],
)
def test_provider_error_propagates_unchanged(fake_provider, error: LLMError) -> None:
    fake_provider.error = error
    with pytest.raises(LLMError) as excinfo:
        asyncio.run(fake_provider.generate(_request(), _model()))
    assert excinfo.value is error


def test_error_category_and_flags_survive_the_boundary(fake_provider) -> None:
    fake_provider.error = RateLimitError("throttled")
    with pytest.raises(RateLimitError) as excinfo:
        asyncio.run(fake_provider.generate(_request(), _model()))
    assert excinfo.value.category is ErrorCategory.RATE_LIMIT
    assert excinfo.value.retryable is True
    assert excinfo.value.fallback_eligible is True


def test_errors_are_not_converted_to_generic_exceptions(fake_provider) -> None:
    fake_provider.error = AuthenticationError("nope")
    with pytest.raises(AuthenticationError):
        asyncio.run(fake_provider.generate(_request(), _model()))


def test_registry_error_is_part_of_the_llm_error_family() -> None:
    registry = ProviderRegistry()
    with pytest.raises(LLMError) as excinfo:
        registry.get("nope")
    assert isinstance(excinfo.value, ProviderNotRegisteredError)
    assert isinstance(excinfo.value, ConfigurationError)
    assert excinfo.value.category is ErrorCategory.CONFIGURATION
    assert excinfo.value.retryable is False
    assert excinfo.value.fallback_eligible is False
