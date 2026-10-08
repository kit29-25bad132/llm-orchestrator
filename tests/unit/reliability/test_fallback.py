"""Behavior tests for candidate-level fallback execution.

A scripted fake provider stands in for real vendors; the real
``RetryEngine`` drives retries; ``asyncio.run`` drives the executor. No test
touches the network or waits on real time.
"""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from typing import Any

import pytest

from llm_orchestrator.contracts import (
    CandidateRef,
    FallbackPolicy,
    FinishReason,
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
    AuthorizationError,
    ConfigurationError,
    ErrorCategory,
    InvalidRequestError,
    LLMError,
    LLMTimeoutError,
    MalformedResponseError,
    ModelUnavailableError,
    NetworkError,
    ProviderUnavailableError,
    RateLimitError,
    StructuredOutputValidationError,
    UnknownProviderError,
)
from llm_orchestrator.providers import ProviderRegistry
from llm_orchestrator.reliability import FallbackExecutor, RetryEngine


class ScriptedProvider:
    """A fake ``LLMProvider`` replaying scripted outcomes per call."""

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


class Recorder:
    """An injectable sleep that records delays instead of waiting."""

    def __init__(self) -> None:
        self.delays: list[float] = []

    async def sleep(self, delay_s: float) -> None:
        self.delays.append(delay_s)


def _request() -> LLMRequest:
    return LLMRequest(messages=[Message(role=MessageRole.USER, content="hi")])


def _model(provider: str, model: str = "m") -> ModelConfig:
    return ModelConfig(provider=provider, model=model)


def _ok(provider: str, model: str = "m", content: str = "ok") -> LLMResponse:
    return LLMResponse(
        request_id="req-1",
        provider=provider,
        model=model,
        finish_reason=FinishReason.STOP,
        latency_ms=1.0,
        content=content,
    )


def _registry(*providers: ScriptedProvider) -> ProviderRegistry:
    registry = ProviderRegistry()
    for provider in providers:
        registry.register(provider)
    return registry


def _run(
    executor: FallbackExecutor,
    request: LLMRequest,
    candidates: Sequence[ModelConfig],
    **kwargs: Any,
) -> OrchestrationResult:
    return asyncio.run(executor.run(request, candidates, **kwargs))


ALLOW = FallbackPolicy(max_fallbacks=5)


# --- basic fallback ---


def test_first_candidate_success_no_fallback() -> None:
    first = ScriptedProvider("pa", [_ok("pa")])
    second = ScriptedProvider("pb", [_ok("pb")])
    executor = FallbackExecutor(_registry(first, second), policy=ALLOW)
    result = _run(executor, _request(), [_model("pa"), _model("pb")])
    assert result.succeeded is True
    assert result.response is not None and result.response.content == "ok"
    assert result.candidates_tried == (CandidateRef(provider="pa", model="m"),)
    assert second.calls == []


def test_first_fails_second_succeeds() -> None:
    first = ScriptedProvider("pa", [ProviderUnavailableError("down")])
    second = ScriptedProvider("pb", [_ok("pb")])
    executor = FallbackExecutor(_registry(first, second), policy=ALLOW)
    result = _run(executor, _request(), [_model("pa"), _model("pb")])
    assert result.succeeded is True
    assert result.response is not None and result.response.provider == "pb"
    assert result.candidates_tried == (
        CandidateRef(provider="pa", model="m"),
        CandidateRef(provider="pb", model="m"),
    )
    assert result.fallback_count == 1


def test_first_two_fail_third_succeeds() -> None:
    providers = [
        ScriptedProvider("pa", [NetworkError("blip")]),
        ScriptedProvider("pb", [LLMTimeoutError("slow")]),
        ScriptedProvider("pc", [_ok("pc")]),
    ]
    executor = FallbackExecutor(_registry(*providers), policy=ALLOW)
    result = _run(executor, _request(), [_model("pa"), _model("pb"), _model("pc")])
    assert result.succeeded is True
    assert result.response is not None and result.response.provider == "pc"
    assert len(result.candidates_tried) == 3
    assert result.fallback_count == 2


def test_all_candidates_fail_terminal_result() -> None:
    first = ScriptedProvider("pa", [NetworkError("blip-a")])
    second = ScriptedProvider("pb", [NetworkError("blip-b")])
    executor = FallbackExecutor(_registry(first, second), policy=ALLOW)
    result = _run(executor, _request(), [_model("pa"), _model("pb")])
    assert result.succeeded is False
    assert result.response is None
    assert result.error is not None
    assert result.error.category is ErrorCategory.NETWORK
    assert result.error.message == "blip-b"
    assert len(result.candidates_tried) == 2
    assert len(result.attempts) == 2


def test_no_candidates_raises_configuration_error() -> None:
    executor = FallbackExecutor(_registry(), policy=ALLOW)
    with pytest.raises(ConfigurationError):
        _run(executor, _request(), [])


def test_only_initial_candidate() -> None:
    provider = ScriptedProvider("pa", [_ok("pa")])
    executor = FallbackExecutor(_registry(provider))
    result = _run(executor, _request(), [_model("pa")])
    assert result.succeeded is True
    assert result.fallback_count == 0


def test_default_policy_disables_fallback() -> None:
    first = ScriptedProvider("pa", [NetworkError("blip")])
    second = ScriptedProvider("pb", [_ok("pb")])
    executor = FallbackExecutor(_registry(first, second))
    assert executor.policy == FallbackPolicy()
    result = _run(executor, _request(), [_model("pa"), _model("pb")])
    assert result.succeeded is False
    assert result.candidates_tried == (CandidateRef(provider="pa", model="m"),)
    assert second.calls == []


def test_per_call_policy_overrides_default() -> None:
    first = ScriptedProvider("pa", [NetworkError("blip")])
    second = ScriptedProvider("pb", [_ok("pb")])
    executor = FallbackExecutor(_registry(first, second))
    result = _run(executor, _request(), [_model("pa"), _model("pb")], policy=ALLOW)
    assert result.succeeded is True


def test_non_model_candidate_raises_configuration_error() -> None:
    executor = FallbackExecutor(_registry(), policy=ALLOW)
    with pytest.raises(ConfigurationError):
        _run(executor, _request(), ["nope"])  # type: ignore[list-item]


# --- eligibility across the taxonomy ---


@pytest.mark.parametrize(
    ("error", "advances"),
    [
        (AuthenticationError("bad key"), True),
        (AuthorizationError("denied"), True),
        (RateLimitError("throttled"), True),
        (LLMTimeoutError("slow"), True),
        (NetworkError("blip"), True),
        (ProviderUnavailableError("down"), True),
        (ModelUnavailableError("gone"), True),
        (MalformedResponseError("garbage"), True),
        (StructuredOutputValidationError("mismatch"), True),
        (UnknownProviderError("mystery"), True),
        (InvalidRequestError("bad params"), False),
        (ConfigurationError("misconfigured"), False),
    ],
)
def test_taxonomy_eligibility(error: LLMError, advances: bool) -> None:
    first = ScriptedProvider("pa", [error])
    second = ScriptedProvider("pb", [_ok("pb")])
    executor = FallbackExecutor(_registry(first, second), policy=ALLOW)
    result = _run(executor, _request(), [_model("pa"), _model("pb")])
    assert (result.succeeded is True) is advances
    assert (len(second.calls) > 0) is advances


def test_mixed_eligibility_stops_at_terminal_error() -> None:
    first = ScriptedProvider("pa", [ModelUnavailableError("gone")])
    second = ScriptedProvider("pb", [InvalidRequestError("bad params")])
    third = ScriptedProvider("pc", [_ok("pc")])
    executor = FallbackExecutor(_registry(first, second, third), policy=ALLOW)
    result = _run(executor, _request(), [_model("pa"), _model("pb"), _model("pc")])
    assert result.succeeded is False
    assert result.error is not None
    assert result.error.category is ErrorCategory.INVALID_REQUEST
    assert len(result.candidates_tried) == 2
    assert third.calls == []


def test_fallback_on_override_restricts_transitions() -> None:
    first = ScriptedProvider("pa", [ModelUnavailableError("gone")])
    second = ScriptedProvider("pb", [_ok("pb")])
    executor = FallbackExecutor(_registry(first, second))
    policy = FallbackPolicy(max_fallbacks=5, fallback_on=frozenset({ErrorCategory.TIMEOUT}))
    result = _run(executor, _request(), [_model("pa"), _model("pb")], policy=policy)
    assert result.succeeded is False
    assert second.calls == []


def test_fallback_on_override_allows_listed_category() -> None:
    first = ScriptedProvider("pa", [InvalidRequestError("bad params")])
    second = ScriptedProvider("pb", [_ok("pb")])
    executor = FallbackExecutor(_registry(first, second))
    policy = FallbackPolicy(max_fallbacks=5, fallback_on=frozenset({ErrorCategory.INVALID_REQUEST}))
    result = _run(executor, _request(), [_model("pa"), _model("pb")], policy=policy)
    assert result.succeeded is True


def test_excluding_validation_failures_stops() -> None:
    first = ScriptedProvider("pa", [StructuredOutputValidationError("mismatch")])
    second = ScriptedProvider("pb", [_ok("pb")])
    executor = FallbackExecutor(_registry(first, second))
    policy = FallbackPolicy(max_fallbacks=5, include_validation_failures=False)
    result = _run(executor, _request(), [_model("pa"), _model("pb")], policy=policy)
    assert result.succeeded is False
    assert second.calls == []


def test_missing_provider_is_terminal_configuration_error() -> None:
    present = ScriptedProvider("pb", [_ok("pb")])
    executor = FallbackExecutor(_registry(present), policy=ALLOW)
    result = _run(executor, _request(), [_model("ghost"), _model("pb")])
    assert result.succeeded is False
    assert result.error is not None
    assert result.error.category is ErrorCategory.CONFIGURATION
    assert result.attempts == ()
    assert len(result.candidates_tried) == 1
    assert present.calls == []


# --- retry interaction ---


def test_retry_exhausts_candidate_before_fallback() -> None:
    first = ScriptedProvider("pa", [NetworkError("blip-1"), NetworkError("blip-2"), _ok("pa")])
    second = ScriptedProvider("pb", [_ok("pb")])
    engine = RetryEngine(RetryPolicy(max_attempts=3, initial_backoff_s=0.0))
    executor = FallbackExecutor(_registry(first, second), engine, ALLOW)
    result = _run(executor, _request(), [_model("pa"), _model("pb")])
    assert result.succeeded is True
    assert result.response is not None and result.response.provider == "pa"
    assert len(first.calls) == 3
    assert second.calls == []
    assert result.fallback_count == 0
    assert [a.attempt for a in result.attempts] == [1, 2, 3]


def test_each_candidate_gets_its_own_retry_budget() -> None:
    first = ScriptedProvider("pa", [NetworkError("down")])
    second = ScriptedProvider("pb", [_ok("pb")])
    engine = RetryEngine(RetryPolicy(max_attempts=2, initial_backoff_s=0.0))
    executor = FallbackExecutor(_registry(first, second), engine, ALLOW)
    result = _run(executor, _request(), [_model("pa"), _model("pb")])
    assert result.succeeded is True
    assert len(first.calls) == 2
    assert [(a.provider, a.attempt) for a in result.attempts] == [
        ("pa", 1),
        ("pa", 2),
        ("pb", 1),
    ]
    assert result.retry_count == 1
    assert result.fallback_count == 1


def test_retry_after_honored_before_fallback() -> None:
    sleep = Recorder()
    first = ScriptedProvider(
        "pa", [RateLimitError("slow down", retry_after_s=2.0), NetworkError("down")]
    )
    second = ScriptedProvider("pb", [_ok("pb")])
    engine = RetryEngine(RetryPolicy(max_attempts=2, initial_backoff_s=0.0), sleep=sleep.sleep)
    executor = FallbackExecutor(_registry(first, second), engine, ALLOW)
    result = _run(executor, _request(), [_model("pa"), _model("pb")])
    assert result.succeeded is True
    assert sleep.delays == [2.0]
    assert len(first.calls) == 2


def test_transient_then_success_never_falls_back() -> None:
    first = ScriptedProvider("pa", [RateLimitError("busy"), _ok("pa")])
    second = ScriptedProvider("pb", [_ok("pb")])
    engine = RetryEngine(RetryPolicy(max_attempts=2, initial_backoff_s=0.0))
    executor = FallbackExecutor(_registry(first, second), engine, ALLOW)
    result = _run(executor, _request(), [_model("pa"), _model("pb")])
    assert result.succeeded is True
    assert second.calls == []
    assert result.fallback_count == 0


def test_fallback_transitions_perform_no_waits() -> None:
    """Candidate transitions are immediate; only the retry engine ever sleeps."""
    sleep = Recorder()
    providers = [
        ScriptedProvider("pa", [NetworkError("a")]),
        ScriptedProvider("pb", [NetworkError("b")]),
        ScriptedProvider("pc", [NetworkError("c")]),
    ]
    engine = RetryEngine(RetryPolicy(max_attempts=1), sleep=sleep.sleep)
    executor = FallbackExecutor(_registry(*providers), engine, ALLOW)
    result = _run(executor, _request(), [_model("pa"), _model("pb"), _model("pc")])
    assert result.succeeded is False
    assert len(result.candidates_tried) == 3
    assert sleep.delays == []


# --- candidate ordering ---


def test_deterministic_fallback_order() -> None:
    providers = [
        ScriptedProvider("pa", [NetworkError("a")]),
        ScriptedProvider("pb", [NetworkError("b")]),
        ScriptedProvider("pc", [NetworkError("c")]),
    ]
    executor = FallbackExecutor(_registry(*providers), policy=ALLOW)
    result = _run(executor, _request(), [_model("pa"), _model("pb"), _model("pc")])
    assert [c.provider for c in result.candidates_tried] == ["pa", "pb", "pc"]


def test_duplicate_candidates_execute_once() -> None:
    first = ScriptedProvider("pa", [NetworkError("a")])
    second = ScriptedProvider("pb", [_ok("pb")])
    executor = FallbackExecutor(_registry(first, second), policy=ALLOW)
    candidates = [_model("pa"), _model("pa"), _model("pb"), _model("pb")]
    result = _run(executor, _request(), candidates)
    assert result.succeeded is True
    assert result.candidates_tried == (
        CandidateRef(provider="pa", model="m"),
        CandidateRef(provider="pb", model="m"),
    )
    assert len(first.calls) == 1


def test_explicit_initial_attempted_first() -> None:
    first = ScriptedProvider("pa", [_ok("pa")])
    second = ScriptedProvider("pb", [_ok("pb")])
    executor = FallbackExecutor(_registry(first, second), policy=ALLOW)
    result = _run(executor, _request(), [_model("pb"), _model("pa")], initial=_model("pa"))
    assert result.succeeded is True
    assert result.candidates_tried == (CandidateRef(provider="pa", model="m"),)
    assert second.calls == []


def test_success_stops_immediately() -> None:
    first = ScriptedProvider("pa", [NetworkError("a")])
    second = ScriptedProvider("pb", [_ok("pb")])
    third = ScriptedProvider("pc", [_ok("pc")])
    executor = FallbackExecutor(_registry(first, second, third), policy=ALLOW)
    result = _run(executor, _request(), [_model("pa"), _model("pb"), _model("pc")])
    assert result.succeeded is True
    assert third.calls == []


def test_policy_candidates_act_as_allowlist_in_routing_order() -> None:
    providers = [
        ScriptedProvider("pa", [NetworkError("a")]),
        ScriptedProvider("pb", [NetworkError("b")]),
        ScriptedProvider("pc", [_ok("pc")]),
    ]
    executor = FallbackExecutor(_registry(*providers), policy=ALLOW)
    policy = FallbackPolicy(
        max_fallbacks=5,
        candidates=(
            CandidateRef(provider="pc", model="m"),
            CandidateRef(provider="pa", model="m"),
        ),
    )
    result = _run(executor, _request(), [_model("pa"), _model("pb"), _model("pc")], policy=policy)
    assert result.succeeded is True
    # Routing order preserved (pa before pc); pb excluded by the allowlist.
    assert [c.provider for c in result.candidates_tried] == ["pa", "pc"]


def test_cross_provider_false_stays_on_initial_provider() -> None:
    same_provider = ScriptedProvider("pa", [NetworkError("a"), _ok("pa")])
    other = ScriptedProvider("pb", [_ok("pb")])
    executor = FallbackExecutor(_registry(same_provider, other))
    policy = FallbackPolicy(max_fallbacks=5, cross_provider=False)
    result = _run(
        executor,
        _request(),
        [_model("pa", "m1"), _model("pb", "m2"), _model("pa", "m3")],
        policy=policy,
    )
    assert result.succeeded is True
    assert [c.model for c in result.candidates_tried] == ["m1", "m3"]
    assert other.calls == []


# --- policy limits ---


def test_max_fallbacks_zero_runs_initial_only() -> None:
    first = ScriptedProvider("pa", [NetworkError("a")])
    second = ScriptedProvider("pb", [_ok("pb")])
    executor = FallbackExecutor(_registry(first, second))
    policy = FallbackPolicy(max_fallbacks=0)
    result = _run(executor, _request(), [_model("pa"), _model("pb")], policy=policy)
    assert result.succeeded is False
    assert len(result.candidates_tried) == 1
    assert second.calls == []


def test_max_fallbacks_one() -> None:
    providers = [
        ScriptedProvider("pa", [NetworkError("a")]),
        ScriptedProvider("pb", [NetworkError("b")]),
        ScriptedProvider("pc", [_ok("pc")]),
    ]
    executor = FallbackExecutor(_registry(*providers))
    policy = FallbackPolicy(max_fallbacks=1)
    result = _run(executor, _request(), [_model("pa"), _model("pb"), _model("pc")], policy=policy)
    assert result.succeeded is False
    assert [c.provider for c in result.candidates_tried] == ["pa", "pb"]
    assert result.error is not None and result.error.message == "b"
    assert providers[2].calls == []


def test_max_fallbacks_two_allows_three_candidates() -> None:
    providers = [
        ScriptedProvider("pa", [NetworkError("a")]),
        ScriptedProvider("pb", [NetworkError("b")]),
        ScriptedProvider("pc", [NetworkError("c")]),
    ]
    executor = FallbackExecutor(_registry(*providers))
    policy = FallbackPolicy(max_fallbacks=2)
    result = _run(executor, _request(), [_model("pa"), _model("pb"), _model("pc")], policy=policy)
    assert result.succeeded is False
    assert len(result.candidates_tried) == 3
    assert result.fallback_count == 2
    assert result.error is not None and result.error.message == "c"


def test_longer_list_than_limit_never_reached() -> None:
    providers = [ScriptedProvider(f"p{i}", [NetworkError(f"e{i}")]) for i in range(4)]
    executor = FallbackExecutor(_registry(*providers))
    policy = FallbackPolicy(max_fallbacks=1)
    result = _run(executor, _request(), [_model(f"p{i}") for i in range(4)], policy=policy)
    assert [c.provider for c in result.candidates_tried] == ["p0", "p1"]
    assert providers[2].calls == [] and providers[3].calls == []


def test_timeout_passes_through_to_every_candidate() -> None:
    first = ScriptedProvider("pa", [NetworkError("a")])
    second = ScriptedProvider("pb", [_ok("pb")])
    executor = FallbackExecutor(_registry(first, second), policy=ALLOW)
    result = _run(executor, _request(), [_model("pa"), _model("pb")], timeout_s=7.5)
    assert result.succeeded is True
    assert first.calls[0]["timeout_s"] == 7.5
    assert second.calls[0]["timeout_s"] == 7.5


# --- cancellation ---


def test_cancellation_during_candidate_propagates_without_fallback() -> None:
    first = ScriptedProvider("pa", [asyncio.CancelledError()])
    second = ScriptedProvider("pb", [_ok("pb")])
    executor = FallbackExecutor(_registry(first, second), policy=ALLOW)
    with pytest.raises(asyncio.CancelledError):
        _run(executor, _request(), [_model("pa"), _model("pb")])
    assert second.calls == []


def test_cancellation_during_retry_sleep_propagates() -> None:
    async def cancelling_sleep(_: float) -> None:
        raise asyncio.CancelledError

    first = ScriptedProvider("pa", [NetworkError("blip")])
    second = ScriptedProvider("pb", [_ok("pb")])
    engine = RetryEngine(RetryPolicy(max_attempts=3), sleep=cancelling_sleep)
    executor = FallbackExecutor(_registry(first, second), engine, ALLOW)
    with pytest.raises(asyncio.CancelledError):
        _run(executor, _request(), [_model("pa"), _model("pb")])
    assert second.calls == []


def test_cancellation_between_candidates_propagates() -> None:
    release = asyncio.Event()

    class HangingProvider(ScriptedProvider):
        async def generate(  # type: ignore[override]
            self,
            request: LLMRequest,
            model: ModelConfig,
            *,
            timeout_s: float | None = None,
        ) -> LLMResponse:
            self.calls.append({"request": request, "model": model})
            await release.wait()
            return _ok(self.provider_id)

    first = ScriptedProvider("pa", [NetworkError("a")])
    hanging = HangingProvider("pb", [_ok("pb")])
    third = ScriptedProvider("pc", [_ok("pc")])
    executor = FallbackExecutor(_registry(first, hanging, third), policy=ALLOW)

    async def abandoned() -> None:
        task = asyncio.ensure_future(
            executor.run(_request(), [_model("pa"), _model("pb"), _model("pc")])
        )
        while not hanging.calls:
            await asyncio.sleep(0)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(abandoned())
    assert third.calls == []


# --- programming errors ---


@pytest.mark.parametrize("bug", [TypeError("bug"), ValueError("bug"), RuntimeError("bug")])
def test_programming_errors_escape_without_fallback(bug: BaseException) -> None:
    first = ScriptedProvider("pa", [bug])
    second = ScriptedProvider("pb", [_ok("pb")])
    executor = FallbackExecutor(_registry(first, second), policy=ALLOW)
    with pytest.raises(type(bug)):
        _run(executor, _request(), [_model("pa"), _model("pb")])
    assert second.calls == []


# --- metadata ---


def test_request_id_preserved_across_candidates() -> None:
    seen: list[str] = []

    class ObservingProvider(ScriptedProvider):
        async def generate(  # type: ignore[override]
            self,
            request: LLMRequest,
            model: ModelConfig,
            *,
            timeout_s: float | None = None,
        ) -> LLMResponse:
            seen.append(request.request_id)
            return await super().generate(request, model, timeout_s=timeout_s)

    request = _request()
    first = ObservingProvider("pa", [NetworkError("a")])
    second = ObservingProvider("pb", [_ok("pb")])
    executor = FallbackExecutor(_registry(first, second), policy=ALLOW)
    result = _run(executor, request, [_model("pa"), _model("pb")])
    assert result.request_id == request.request_id
    assert seen == [request.request_id, request.request_id]


def test_attempt_history_distinguishes_candidates_and_attempts() -> None:
    first = ScriptedProvider("pa", [NetworkError("a1"), NetworkError("a2"), NetworkError("a3")])
    second = ScriptedProvider("pb", [_ok("pb")])
    engine = RetryEngine(RetryPolicy(max_attempts=3, initial_backoff_s=0.0))
    executor = FallbackExecutor(_registry(first, second), engine, ALLOW)
    result = _run(executor, _request(), [_model("pa"), _model("pb")])
    assert [(a.provider, a.model, a.attempt, a.succeeded) for a in result.attempts] == [
        ("pa", "m", 1, False),
        ("pa", "m", 2, False),
        ("pa", "m", 3, False),
        ("pb", "m", 1, True),
    ]
    assert [a.error_category for a in result.attempts[:3]] == [ErrorCategory.NETWORK] * 3


def test_final_error_is_last_candidate_terminal_error() -> None:
    first = ScriptedProvider("pa", [NetworkError("first-down")])
    second = ScriptedProvider("pb", [ProviderUnavailableError("second-down")])
    executor = FallbackExecutor(_registry(first, second), policy=ALLOW)
    result = _run(executor, _request(), [_model("pa"), _model("pb")])
    assert result.succeeded is False
    assert result.error is not None
    assert result.error.category is ErrorCategory.PROVIDER_UNAVAILABLE
    assert result.error.message == "second-down"
    assert [a.error_category for a in result.attempts] == [
        ErrorCategory.NETWORK,
        ErrorCategory.PROVIDER_UNAVAILABLE,
    ]


def test_safe_message_redaction_in_terminal_result() -> None:
    first = ScriptedProvider("pa", [NetworkError("Bearer abc.def-ghi")])
    executor = FallbackExecutor(_registry(first), policy=ALLOW)
    result = _run(executor, _request(), [_model("pa")])
    assert result.error is not None
    assert "abc.def-ghi" not in result.error.message
    assert "[REDACTED]" in result.error.message


def test_terminal_result_serializes_without_live_exceptions() -> None:
    first = ScriptedProvider("pa", [NetworkError("a")])
    second = ScriptedProvider("pb", [ModelUnavailableError("b")])
    executor = FallbackExecutor(_registry(first, second), policy=ALLOW)
    result = _run(executor, _request(), [_model("pa"), _model("pb")])
    dumped = result.model_dump(mode="json")
    assert dumped["succeeded"] is False
    assert dumped["error"]["category"] == "model_unavailable"
    assert len(dumped["attempts"]) == 2
    assert [c["provider"] for c in dumped["candidates_tried"]] == ["pa", "pb"]


# --- concurrency ---


def test_concurrent_runs_stay_isolated() -> None:
    slow_release = asyncio.Event()

    class GatedProvider(ScriptedProvider):
        async def generate(  # type: ignore[override]
            self,
            request: LLMRequest,
            model: ModelConfig,
            *,
            timeout_s: float | None = None,
        ) -> LLMResponse:
            self.calls.append({"request": request, "model": model})
            await slow_release.wait()
            return _ok(self.provider_id, content=f"hello-{request.request_id}")

    gated = GatedProvider("pa", [_ok("pa")])
    fast = ScriptedProvider("pb", [_ok("pb")])
    executor = FallbackExecutor(_registry(gated, fast), policy=ALLOW)

    async def both() -> tuple[Any, Any]:
        slow_task = asyncio.ensure_future(executor.run(_request(), [_model("pa")]))
        while not gated.calls:
            await asyncio.sleep(0)
        fast_result = await executor.run(_request(), [_model("pb")])
        slow_release.set()
        return await slow_task, fast_result

    slow_result, fast_result = asyncio.run(both())
    assert slow_result.succeeded and fast_result.succeeded
    assert slow_result.request_id != fast_result.request_id
    assert slow_result.candidates_tried == (CandidateRef(provider="pa", model="m"),)
    assert fast_result.candidates_tried == (CandidateRef(provider="pb", model="m"),)
    assert slow_result.response is not None and fast_result.response is not None
    assert slow_result.response.content != fast_result.response.content


# --- ReviewIQ-shaped flow, generically expressed ---


def test_multi_model_then_cross_provider_flow() -> None:
    """Two models of one provider fail, a second provider succeeds.

    Mirrors the intended consumer shape (preferred vendor models first, then
    cross-provider) without naming any real vendor or domain.
    """

    first_provider = ScriptedProvider(
        "pa",
        [
            LLMTimeoutError("m1 slow"),
            LLMTimeoutError("m1 slow"),
            ProviderUnavailableError("m2 down"),
        ],
    )
    other = ScriptedProvider("pb", [_ok("pb")])
    registry = _registry(first_provider, other)
    engine = RetryEngine(RetryPolicy(max_attempts=2, initial_backoff_s=0.0))
    executor = FallbackExecutor(registry, engine, FallbackPolicy(max_fallbacks=5))
    result = _run(
        executor,
        _request(),
        [_model("pa", "m1"), _model("pa", "m2"), _model("pb", "m3")],
    )
    assert result.succeeded is True
    assert [c.model for c in result.candidates_tried] == ["m1", "m2", "m3"]
    assert result.fallback_count == 2
