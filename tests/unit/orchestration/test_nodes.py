"""Unit tests for the execute_model node, called directly with fakes."""

from __future__ import annotations

import asyncio
import random
from typing import Any

import pytest

from llm_orchestrator.contracts import (
    ErrorInfo,
    FinishReason,
    JitterStrategy,
    LLMRequest,
    LLMResponse,
    Message,
    MessageRole,
    ModelConfig,
    OrchestrationResult,
    RetryPolicy,
)
from llm_orchestrator.errors import (
    AuthenticationError,
    ErrorCategory,
    InvalidRequestError,
    NetworkError,
    ProviderNotRegisteredError,
    RateLimitError,
)
from llm_orchestrator.orchestration import execute_model, initial_state
from llm_orchestrator.providers import ProviderRegistry
from llm_orchestrator.reliability import RetryEngine


class ScriptedProvider:
    """A fake ``LLMProvider`` replaying scripted outcomes (responses/errors)."""

    def __init__(self, provider_id: str, outcomes: list[LLMResponse | BaseException]) -> None:
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
        pass


def _success(content: str = "ok") -> LLMResponse:
    return LLMResponse(
        request_id="req-1",
        provider="fake",
        model="m",
        finish_reason=FinishReason.STOP,
        latency_ms=1.0,
        content=content,
    )


def _request(request_id: str = "req-1") -> LLMRequest:
    return LLMRequest(
        messages=[Message(role=MessageRole.USER, content="hi")], request_id=request_id
    )


def _model() -> ModelConfig:
    return ModelConfig(provider="fake", model="m")


def _registry(provider: ScriptedProvider | None) -> ProviderRegistry:
    registry = ProviderRegistry()
    if provider is not None:
        registry.register(provider)
    return registry


def _engine(policy: RetryPolicy) -> RetryEngine:
    async def instant_sleep(_: float) -> None:
        return None

    return RetryEngine(policy, sleep=instant_sleep, rng=random.Random(0))


def _run_node(
    provider: ScriptedProvider | None,
    *,
    timeout_s: float | None = None,
    policy: RetryPolicy | None = None,
) -> tuple[dict[str, Any], ScriptedProvider | None]:
    registry = _registry(provider)
    engine = _engine(policy or RetryPolicy(max_attempts=1, jitter=JitterStrategy.NONE))
    state = initial_state(_request(), _model(), timeout_s=timeout_s)
    update = asyncio.run(execute_model(state, registry=registry, retry_engine=engine))
    return update, provider


def test_successful_execution_returns_response_and_result() -> None:
    provider = ScriptedProvider("fake", [_success()])
    update, _ = _run_node(provider)
    assert update["response"] is not None
    assert update["response"].content == "ok"
    assert update["error"] is None
    result = update["result"]
    assert isinstance(result, OrchestrationResult)
    assert result.succeeded is True
    assert result.response is not None
    assert result.error is None
    assert result.request_id == "req-1"


def test_attempt_record_describes_the_terminal_success() -> None:
    provider = ScriptedProvider("fake", [_success()])
    update, _ = _run_node(provider)
    result = update["result"]
    assert len(result.attempts) == 1
    record = result.attempts[0]
    assert record.succeeded is True
    assert record.attempt == 1
    assert record.provider == "fake"
    assert record.model == "m"
    assert result.candidates_tried[0].provider == "fake"


def test_normalized_failure_returns_error_info_result() -> None:
    provider = ScriptedProvider("fake", [RateLimitError("throttled")])
    update, _ = _run_node(provider, policy=RetryPolicy(max_attempts=1, jitter=JitterStrategy.NONE))
    assert update["response"] is None
    error = update["error"]
    assert isinstance(error, RateLimitError)
    result = update["result"]
    assert result.succeeded is False
    assert result.response is None
    assert isinstance(result.error, ErrorInfo)
    assert result.error.category is ErrorCategory.RATE_LIMIT
    assert result.error.retryable is True
    assert result.attempts[0].succeeded is False
    assert result.attempts[0].error_category is ErrorCategory.RATE_LIMIT


def test_non_retryable_failure_makes_exactly_one_provider_call() -> None:
    provider = ScriptedProvider("fake", [AuthenticationError("bad key")])
    update, _ = _run_node(provider, policy=RetryPolicy(max_attempts=3, jitter=JitterStrategy.NONE))
    assert isinstance(update["error"], AuthenticationError)
    assert len(provider.calls) == 1


def test_retryable_failures_are_retried_through_the_engine() -> None:
    provider = ScriptedProvider(
        "fake", [NetworkError("down"), NetworkError("down"), _success("recovered")]
    )
    update, _ = _run_node(provider, policy=RetryPolicy(max_attempts=3, jitter=JitterStrategy.NONE))
    assert update["response"] is not None
    assert update["response"].content == "recovered"
    assert len(provider.calls) == 3
    assert update["result"].attempts[0].attempt == 3


def test_exhaustion_reports_the_final_attempt() -> None:
    provider = ScriptedProvider("fake", [NetworkError("down")])
    update, _ = _run_node(provider, policy=RetryPolicy(max_attempts=3, jitter=JitterStrategy.NONE))
    assert isinstance(update["error"], NetworkError)
    assert update["error"].attempt == 3
    assert update["result"].attempts[0].attempt == 3
    assert len(provider.calls) == 3


def test_unregistered_provider_is_a_normalized_failure() -> None:
    update, _ = _run_node(None)
    assert update["response"] is None
    assert isinstance(update["error"], ProviderNotRegisteredError)
    result = update["result"]
    assert result.succeeded is False
    assert result.error is not None
    assert result.error.category is ErrorCategory.CONFIGURATION
    assert result.attempts == ()


def test_unexpected_exceptions_propagate_instead_of_becoming_results() -> None:
    provider = ScriptedProvider("fake", [ValueError("programmer bug")])
    registry = _registry(provider)
    engine = _engine(RetryPolicy(max_attempts=1))
    state = initial_state(_request(), _model())
    with pytest.raises(ValueError, match="programmer bug"):
        asyncio.run(execute_model(state, registry=registry, retry_engine=engine))


def test_node_does_not_mutate_inputs_and_passes_timeout_through() -> None:
    provider = ScriptedProvider("fake", [_success()])
    registry = _registry(provider)
    engine = _engine(RetryPolicy(max_attempts=1))
    request = _request()
    before = request.model_dump()
    state = initial_state(request, _model(), timeout_s=7.5)
    asyncio.run(execute_model(state, registry=registry, retry_engine=engine))
    assert request.model_dump() == before
    assert provider.calls[0]["timeout_s"] == 7.5
    assert provider.calls[0]["request"] is request


def test_wrong_vendor_model_is_rejected_without_provider_call() -> None:
    provider = ScriptedProvider("other", [_success()])
    registry = _registry(provider)
    engine = _engine(RetryPolicy(max_attempts=1))
    state = initial_state(_request(), _model())
    update = asyncio.run(execute_model(state, registry=registry, retry_engine=engine))
    assert isinstance(update["error"], ProviderNotRegisteredError)
    assert provider.calls == []


def test_invalid_request_error_preserves_category() -> None:
    provider = ScriptedProvider("fake", [InvalidRequestError("bad shape")])
    update, _ = _run_node(provider)
    result = update["result"]
    assert result.error is not None
    assert result.error.category is ErrorCategory.INVALID_REQUEST
    assert result.error.retryable is False
    assert result.error.fallback_eligible is False
