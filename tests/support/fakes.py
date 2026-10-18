"""Deterministic test-only doubles and builders for the offline suite.

Everything here is test infrastructure: programmable providers, a recording
sleep, request/model/response builders, and scripted SDK clients for the real
Gemini/Groq adapters. No helper touches the network, sleeps for real time, or
requires credentials. Each helper is stateful per instance, so tests stay
isolated as long as they construct fresh helpers (never module-level ones).
"""

from __future__ import annotations

import asyncio
from collections.abc import Coroutine
from typing import Any, TypeVar

from llm_orchestrator.contracts import (
    FinishReason,
    LLMRequest,
    LLMResponse,
    Message,
    MessageRole,
    ModelConfig,
)
from llm_orchestrator.providers import ProviderRegistry

T = TypeVar("T")

Outcome = LLMResponse | BaseException


class ScriptedProvider:
    """A fake ``LLMProvider`` replaying a scripted outcome queue.

    Each ``generate`` call records ``{"request", "model", "timeout_s"}`` and
    replays the next outcome; when the queue drains, the final outcome repeats
    so ``max_attempts`` budgets larger than the script still terminate. This
    mirrors the doubles already used by the retry/fallback/orchestration
    suites, consolidated here for new tests.
    """

    def __init__(self, provider_id: str, outcomes: list[Outcome]) -> None:
        if not outcomes:
            raise ValueError("ScriptedProvider requires at least one outcome")
        self.provider_id = provider_id
        self._outcomes = list(outcomes)
        self.calls: list[dict[str, Any]] = []

    async def generate(
        self,
        request: LLMRequest,
        model: ModelConfig,
        *,
        timeout_s: float | None = None,
    ) -> LLMResponse:
        self.calls.append({"request": request, "model": model, "timeout_s": timeout_s})
        outcome = self._outcomes.pop(0) if len(self._outcomes) > 1 else self._outcomes[0]
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome

    async def aclose(self) -> None:
        """No-op: there is no underlying client to release."""


class RecordingSleep:
    """An injectable async sleep recording requested delays instead of waiting."""

    def __init__(self) -> None:
        self.delays: list[float] = []

    async def sleep(self, delay_s: float) -> None:
        self.delays.append(delay_s)


def make_request(content: str = "hi", *, request_id: str = "req-1") -> LLMRequest:
    """Build a minimal one-message request with a deterministic id."""
    return LLMRequest(
        messages=[Message(role=MessageRole.USER, content=content)],
        request_id=request_id,
    )


def make_model(provider: str, model: str = "m") -> ModelConfig:
    """Build a candidate reference for ``provider``/``model``."""
    return ModelConfig(provider=provider, model=model)


def make_response(
    provider: str,
    model: str = "m",
    content: str = "ok",
    *,
    request_id: str = "req-1",
) -> LLMResponse:
    """Build a minimal successful normalized response."""
    return LLMResponse(
        request_id=request_id,
        provider=provider,
        model=model,
        finish_reason=FinishReason.STOP,
        latency_ms=1.0,
        content=content,
    )


def make_registry(*providers: ScriptedProvider) -> ProviderRegistry:
    """Register scripted providers in a fresh, independent registry."""
    registry = ProviderRegistry()
    for provider in providers:
        registry.register(provider)
    return registry


def run_sync(coro: Coroutine[Any, Any, T]) -> T:
    """Drive one coroutine to completion (thin wrapper over ``asyncio.run``)."""
    return asyncio.run(coro)


def attempt_sequence(result: Any) -> list[tuple[str, str, int, bool]]:
    """Return ``(provider, model, attempt, succeeded)`` per recorded attempt."""
    return [
        (attempt.provider, attempt.model, attempt.attempt, attempt.succeeded)
        for attempt in result.attempts
    ]


class FakeGeminiSdkClient:
    """A scripted stand-in for the Google GenAI SDK client.

    Exposes only the ``aio.models.generate_content`` seam the real
    :class:`GeminiProvider` calls. Outcomes are real SDK response objects or
    exceptions so adapter normalization is genuinely exercised.
    """

    def __init__(self, outcomes: list[Any]) -> None:
        if not outcomes:
            raise ValueError("FakeGeminiSdkClient requires at least one outcome")
        self._outcomes = list(outcomes)
        self.calls: list[dict[str, Any]] = []
        self.aio = self._Aio(self)

    class _Models:
        def __init__(self, outer: FakeGeminiSdkClient) -> None:
            self._outer = outer

        async def generate_content(self, *, model: str, contents: Any, config: Any = None) -> Any:
            outer = self._outer
            outer.calls.append({"model": model, "contents": contents, "config": config})
            outcome = outer._outcomes.pop(0) if len(outer._outcomes) > 1 else outer._outcomes[0]
            if isinstance(outcome, BaseException):
                raise outcome
            return outcome

    class _Aio:
        def __init__(self, outer: FakeGeminiSdkClient) -> None:
            self.models = FakeGeminiSdkClient._Models(outer)

        async def aclose(self) -> None:
            """No-op: there is no real connection to close."""


class FakeGroqSdkClient:
    """A scripted stand-in for the Groq SDK client.

    Exposes only the ``chat.completions.create`` seam the real
    :class:`GroqProvider` calls. Outcomes are real SDK response objects or
    exceptions so adapter normalization is genuinely exercised.
    """

    def __init__(self, outcomes: list[Any]) -> None:
        if not outcomes:
            raise ValueError("FakeGroqSdkClient requires at least one outcome")
        self._outcomes = list(outcomes)
        self.calls: list[dict[str, Any]] = []
        self.chat = self._Chat(self)

    class _Completions:
        def __init__(self, outer: FakeGroqSdkClient) -> None:
            self._outer = outer

        async def create(self, **kwargs: Any) -> Any:
            outer = self._outer
            outer.calls.append(dict(kwargs))
            outcome = outer._outcomes.pop(0) if len(outer._outcomes) > 1 else outer._outcomes[0]
            if isinstance(outcome, BaseException):
                raise outcome
            return outcome

    class _Chat:
        def __init__(self, outer: FakeGroqSdkClient) -> None:
            self.completions = FakeGroqSdkClient._Completions(outer)

    async def close(self) -> None:
        """No-op: there is no real connection to close."""
