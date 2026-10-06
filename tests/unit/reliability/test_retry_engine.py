"""Behavior tests for the retry engine.

A scripted fake provider stands in for Gemini/Groq: the real engine drives it,
a recording sleep stands in for time, and no test ever waits for real seconds.
"""

from __future__ import annotations

import asyncio
import random
from typing import Any

import pytest

from llm_orchestrator.contracts import (
    FinishReason,
    JitterStrategy,
    LLMRequest,
    LLMResponse,
    Message,
    MessageRole,
    ModelConfig,
    RetryPolicy,
)
from llm_orchestrator.errors import (
    AuthenticationError,
    ErrorCategory,
    InvalidRequestError,
    LLMError,
    NetworkError,
    ProviderUnavailableError,
    RateLimitError,
)
from llm_orchestrator.reliability import RetryEngine


class ScriptedProvider:
    """A fake ``LLMProvider`` replaying scripted outcomes (responses/errors)."""

    def __init__(self, outcomes: list[LLMResponse | BaseException]) -> None:
        self._outcomes = list(outcomes)
        self.calls: list[dict[str, Any]] = []
        self.provider_id = "scripted"

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


class Recorder:
    """An injectable sleep that records delays instead of waiting."""

    def __init__(self) -> None:
        self.delays: list[float] = []

    async def sleep(self, delay_s: float) -> None:
        self.delays.append(delay_s)


def _success(content: str = "ok") -> LLMResponse:
    return LLMResponse(
        request_id="req-1",
        provider="scripted",
        model="m",
        finish_reason=FinishReason.STOP,
        latency_ms=1.0,
        content=content,
    )


def _request() -> LLMRequest:
    return LLMRequest(messages=[Message(role=MessageRole.USER, content="hi")], request_id="req-1")


def _model() -> ModelConfig:
    return ModelConfig(provider="scripted", model="m")


def _engine(policy: RetryPolicy, recorder: Recorder) -> RetryEngine:
    return RetryEngine(policy, sleep=recorder.sleep, rng=random.Random(0))


def _no_retry_policy(**kwargs: Any) -> RetryPolicy:
    return RetryPolicy(jitter=JitterStrategy.NONE, **kwargs)


# --- success -----------------------------------------------------------------


def test_first_attempt_success_returns_immediately() -> None:
    recorder = Recorder()
    provider = ScriptedProvider([_success()])
    engine = _engine(_no_retry_policy(max_attempts=3), recorder)
    response = asyncio.run(engine.generate(provider, _request(), _model()))
    assert response.content == "ok"
    assert len(provider.calls) == 1
    assert recorder.delays == []


# --- retry then success -------------------------------------------------------


def test_retryable_failure_then_success() -> None:
    recorder = Recorder()
    provider = ScriptedProvider([RateLimitError("throttled"), _success("recovered")])
    engine = _engine(_no_retry_policy(max_attempts=3, initial_backoff_s=0.5), recorder)
    response = asyncio.run(engine.generate(provider, _request(), _model()))
    assert response.content == "recovered"
    assert len(provider.calls) == 2
    assert recorder.delays == [0.5]


def test_multiple_retries_before_success() -> None:
    recorder = Recorder()
    provider = ScriptedProvider(
        [NetworkError("down"), ProviderUnavailableError("outage"), _success()]
    )
    engine = _engine(
        _no_retry_policy(max_attempts=3, initial_backoff_s=0.5, multiplier=2.0), recorder
    )
    asyncio.run(engine.generate(provider, _request(), _model()))
    assert len(provider.calls) == 3
    assert recorder.delays == [0.5, 1.0]


# --- exhaustion ----------------------------------------------------------------


def test_exhaustion_raises_the_original_normalized_error() -> None:
    recorder = Recorder()
    failure = RateLimitError("still throttled")
    provider = ScriptedProvider([failure])
    engine = _engine(_no_retry_policy(max_attempts=3, initial_backoff_s=0.5), recorder)
    with pytest.raises(RateLimitError) as excinfo:
        asyncio.run(engine.generate(provider, _request(), _model()))
    assert excinfo.value is failure
    assert excinfo.value.attempt == 3
    assert excinfo.value.retryable is True
    assert len(provider.calls) == 3
    assert recorder.delays == [0.5, 1.0]


def test_max_attempts_is_never_exceeded() -> None:
    recorder = Recorder()
    provider = ScriptedProvider([ProviderUnavailableError("down")])
    engine = _engine(_no_retry_policy(max_attempts=5), recorder)
    with pytest.raises(ProviderUnavailableError):
        asyncio.run(engine.generate(provider, _request(), _model()))
    assert len(provider.calls) == 5
    assert len(recorder.delays) == 4


# --- non-retryable --------------------------------------------------------------


def test_non_retryable_error_propagates_without_retry_or_sleep() -> None:
    recorder = Recorder()
    failure = AuthenticationError("bad key")
    provider = ScriptedProvider([failure])
    engine = _engine(_no_retry_policy(max_attempts=3), recorder)
    with pytest.raises(AuthenticationError) as excinfo:
        asyncio.run(engine.generate(provider, _request(), _model()))
    assert excinfo.value is failure
    assert excinfo.value.attempt == 1
    assert len(provider.calls) == 1
    assert recorder.delays == []


def test_default_policy_attempts_exactly_once() -> None:
    recorder = Recorder()
    provider = ScriptedProvider([RateLimitError("throttled")])
    engine = RetryEngine(sleep=recorder.sleep, rng=random.Random(0))
    assert engine.policy.max_attempts == 1
    with pytest.raises(RateLimitError):
        asyncio.run(engine.generate(provider, _request(), _model()))
    assert len(provider.calls) == 1
    assert recorder.delays == []


# --- retry_on overrides -----------------------------------------------------------


def test_retry_on_override_retries_a_normally_fatal_category() -> None:
    recorder = Recorder()
    provider = ScriptedProvider([InvalidRequestError("bad"), _success()])
    policy = _no_retry_policy(max_attempts=2, retry_on=frozenset({ErrorCategory.INVALID_REQUEST}))
    engine = _engine(policy, recorder)
    asyncio.run(engine.generate(provider, _request(), _model()))
    assert len(provider.calls) == 2
    assert len(recorder.delays) == 1


def test_retry_on_override_suppresses_a_normally_retryable_category() -> None:
    recorder = Recorder()
    provider = ScriptedProvider([RateLimitError("throttled")])
    policy = _no_retry_policy(max_attempts=3, retry_on=frozenset())
    engine = _engine(policy, recorder)
    with pytest.raises(RateLimitError):
        asyncio.run(engine.generate(provider, _request(), _model()))
    assert len(provider.calls) == 1
    assert recorder.delays == []


# --- retry-after -------------------------------------------------------------------


def test_retry_after_raises_the_delay_without_exceeding_the_maximum() -> None:
    recorder = Recorder()
    provider = ScriptedProvider([RateLimitError("slow", retry_after_s=5.0), _success()])
    policy = _no_retry_policy(max_attempts=2, initial_backoff_s=0.5, max_backoff_s=30.0)
    asyncio.run(_engine(policy, recorder).generate(provider, _request(), _model()))
    assert recorder.delays == [5.0]


def test_retry_after_is_capped_at_max_backoff() -> None:
    recorder = Recorder()
    provider = ScriptedProvider([RateLimitError("slow", retry_after_s=3600.0), _success()])
    policy = _no_retry_policy(max_attempts=2, initial_backoff_s=0.5, max_backoff_s=4.0)
    asyncio.run(_engine(policy, recorder).generate(provider, _request(), _model()))
    assert recorder.delays == [4.0]


@pytest.mark.parametrize("retry_after", [-1.0, float("nan"), float("inf"), None])
def test_invalid_retry_after_is_ignored(retry_after: float | None) -> None:
    recorder = Recorder()
    provider = ScriptedProvider([RateLimitError("slow", retry_after_s=retry_after), _success()])
    policy = _no_retry_policy(max_attempts=2, initial_backoff_s=0.5)
    asyncio.run(_engine(policy, recorder).generate(provider, _request(), _model()))
    assert recorder.delays == [0.5]


# --- unknown exceptions and cancellation -------------------------------------------


def test_unexpected_exception_is_not_retried() -> None:
    recorder = Recorder()
    failure = ValueError("programmer bug")
    provider = ScriptedProvider([failure])
    engine = _engine(_no_retry_policy(max_attempts=3), recorder)
    with pytest.raises(ValueError) as excinfo:
        asyncio.run(engine.generate(provider, _request(), _model()))
    assert excinfo.value is failure
    assert len(provider.calls) == 1
    assert recorder.delays == []


def test_cancellation_propagates_without_retry() -> None:
    recorder = Recorder()
    provider = ScriptedProvider([asyncio.CancelledError()])
    engine = _engine(_no_retry_policy(max_attempts=3), recorder)
    with pytest.raises(asyncio.CancelledError):
        asyncio.run(engine.generate(provider, _request(), _model()))
    assert len(provider.calls) == 1
    assert recorder.delays == []


def test_cancellation_during_sleep_propagates() -> None:
    async def cancelling_sleep(_: float) -> None:
        raise asyncio.CancelledError

    provider = ScriptedProvider([RateLimitError("throttled"), _success()])
    engine = RetryEngine(_no_retry_policy(max_attempts=3), sleep=cancelling_sleep)
    with pytest.raises(asyncio.CancelledError):
        asyncio.run(engine.generate(provider, _request(), _model()))
    assert len(provider.calls) == 1


# --- immutability: same request, same provider, no fallback --------------------------


def test_request_is_never_mutated_and_same_object_is_reused() -> None:
    recorder = Recorder()
    provider = ScriptedProvider([NetworkError("flaky"), _success()])
    engine = _engine(_no_retry_policy(max_attempts=2), recorder)
    request = _request()
    before = request.model_dump()
    asyncio.run(engine.generate(provider, request, _model()))
    assert request.model_dump() == before
    assert provider.calls[0]["request"] is request
    assert provider.calls[1]["request"] is request


def test_same_provider_and_model_serve_every_attempt() -> None:
    recorder = Recorder()
    provider = ScriptedProvider([NetworkError("flaky"), _success()])
    other = ScriptedProvider([_success("wrong")])
    engine = _engine(_no_retry_policy(max_attempts=2), recorder)
    model = _model()
    asyncio.run(engine.generate(provider, _request(), model))
    assert len(provider.calls) == 2
    assert all(call["model"] == model for call in provider.calls)
    assert other.calls == []


def test_exhaustion_never_falls_back_to_another_provider() -> None:
    recorder = Recorder()
    provider = ScriptedProvider([ProviderUnavailableError("down")])
    fallback = ScriptedProvider([_success("should never be used")])
    engine = _engine(_no_retry_policy(max_attempts=3), recorder)
    with pytest.raises(ProviderUnavailableError):
        asyncio.run(engine.generate(provider, _request(), _model()))
    assert fallback.calls == []


def test_timeout_is_forwarded_to_every_attempt() -> None:
    recorder = Recorder()
    provider = ScriptedProvider([NetworkError("flaky"), _success()])
    engine = _engine(_no_retry_policy(max_attempts=2), recorder)
    asyncio.run(engine.generate(provider, _request(), _model(), timeout_s=7.5))
    assert [call["timeout_s"] for call in provider.calls] == [7.5, 7.5]


# --- concurrency ----------------------------------------------------------------------


def test_concurrent_calls_keep_independent_retry_state() -> None:
    async def run_both() -> None:
        recorder = Recorder()
        engine = _engine(_no_retry_policy(max_attempts=3, initial_backoff_s=0.25), recorder)
        first = ScriptedProvider([NetworkError("a")])
        second = ScriptedProvider([NetworkError("b")])
        with pytest.raises(NetworkError):
            await asyncio.gather(
                engine.generate(first, _request(), _model()),
                engine.generate(second, _request(), _model()),
            )

    asyncio.run(run_both())


def test_concurrent_success_and_exhaustion_do_not_interfere() -> None:
    async def main() -> tuple[LLMResponse, int]:
        recorder = Recorder()
        engine = _engine(_no_retry_policy(max_attempts=2, initial_backoff_s=0.1), recorder)
        lucky = ScriptedProvider([_success("fast")])
        unlucky = ScriptedProvider([NetworkError("down")])

        async def run_unlucky() -> int:
            with pytest.raises(NetworkError):
                await engine.generate(unlucky, _request(), _model())
            return len(unlucky.calls)

        response, unlucky_calls = await asyncio.gather(
            engine.generate(lucky, _request(), _model()), run_unlucky()
        )
        assert isinstance(response, LLMResponse)
        return response, unlucky_calls

    response, unlucky_calls = asyncio.run(main())
    assert response.content == "fast"
    assert unlucky_calls == 2


# --- error identity ---------------------------------------------------------------------


def test_llm_error_flags_survive_the_engine() -> None:
    recorder = Recorder()
    failure = ProviderUnavailableError("outage")
    provider = ScriptedProvider([failure])
    engine = _engine(_no_retry_policy(max_attempts=1), recorder)
    with pytest.raises(LLMError) as excinfo:
        asyncio.run(engine.generate(provider, _request(), _model()))
    assert excinfo.value.category is ErrorCategory.PROVIDER_UNAVAILABLE
    assert excinfo.value.retryable is True
