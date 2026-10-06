"""Tests for the provider abstraction itself."""

from __future__ import annotations

import asyncio

from llm_orchestrator.contracts import (
    FinishReason,
    LLMRequest,
    LLMResponse,
    Message,
    MessageRole,
    ModelConfig,
)
from llm_orchestrator.providers import LLMProvider


def _request(content: str = "hello") -> LLMRequest:
    return LLMRequest(
        messages=[Message(role=MessageRole.USER, content=content)],
        request_id="req-1",
    )


def _model() -> ModelConfig:
    return ModelConfig(provider="fake", model="fake-model-1")


def test_concrete_provider_satisfies_the_protocol(fake_provider) -> None:
    assert isinstance(fake_provider, LLMProvider)


def test_partial_implementation_does_not_satisfy_the_protocol() -> None:
    class Partial:
        provider_id = "partial"

    assert not isinstance(Partial(), LLMProvider)
    assert not isinstance(object(), LLMProvider)


def test_generate_returns_a_phase2_response(fake_provider) -> None:
    response = asyncio.run(fake_provider.generate(_request(), _model()))
    assert isinstance(response, LLMResponse)


def test_generate_preserves_request_id_and_provider_provenance(fake_provider) -> None:
    response = asyncio.run(fake_provider.generate(_request(), _model()))
    assert response.request_id == "req-1"
    assert response.provider == "fake"
    assert response.model == "fake-model-1"
    assert response.finish_reason is FinishReason.STOP


def test_generate_echoes_request_content(fake_provider) -> None:
    response = asyncio.run(fake_provider.generate(_request("summarize this"), _model()))
    assert response.content == "echo:summarize this"


def test_generate_receives_the_target_model(fake_provider) -> None:
    model = ModelConfig(provider="fake", model="another-model")
    asyncio.run(fake_provider.generate(_request(), model))
    assert fake_provider.calls == [("req-1", "another-model")]


def test_generate_accepts_a_timeout_override(fake_provider) -> None:
    response = asyncio.run(fake_provider.generate(_request(), _model(), timeout_s=2.5))
    assert isinstance(response, LLMResponse)


def test_aclose_releases_resources(fake_provider) -> None:
    assert fake_provider.closed is False
    asyncio.run(fake_provider.aclose())
    assert fake_provider.closed is True
