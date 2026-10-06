"""Tests for the policy contracts.

Only configuration shape is tested; no retry/fallback/routing behavior exists yet.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from llm_orchestrator.contracts import (
    CandidateRef,
    FallbackPolicy,
    JitterStrategy,
    LLMRequest,
    Message,
    MessageRole,
    ModelConfig,
    RetryPolicy,
    RoutingContext,
    RoutingPolicy,
)
from llm_orchestrator.errors import ErrorCategory

VALID_REQUEST = LLMRequest(messages=[Message(role=MessageRole.USER, content="hi")])


def test_retry_policy_defaults() -> None:
    policy = RetryPolicy()
    assert policy.max_attempts == 1
    assert policy.initial_backoff_s == 0.5
    assert policy.multiplier == 2.0
    assert policy.max_backoff_s == 30.0
    assert policy.jitter is JitterStrategy.FULL
    assert policy.retry_on is None


def test_retry_policy_accepts_categories() -> None:
    policy = RetryPolicy(retry_on=frozenset({ErrorCategory.RATE_LIMIT, ErrorCategory.TIMEOUT}))
    assert policy.retry_on is not None
    assert ErrorCategory.RATE_LIMIT in policy.retry_on


@pytest.mark.parametrize("attempts", [0, -1])
def test_retry_max_attempts_must_be_positive(attempts: int) -> None:
    with pytest.raises(ValidationError):
        RetryPolicy(max_attempts=attempts)


def test_retry_negative_delays_rejected() -> None:
    with pytest.raises(ValidationError):
        RetryPolicy(initial_backoff_s=-1.0)
    with pytest.raises(ValidationError):
        RetryPolicy(max_backoff_s=-1.0)


def test_retry_multiplier_must_be_at_least_one() -> None:
    with pytest.raises(ValidationError):
        RetryPolicy(multiplier=0.5)


def test_retry_backoff_window_must_be_coherent() -> None:
    with pytest.raises(ValidationError):
        RetryPolicy(initial_backoff_s=10.0, max_backoff_s=1.0)


def test_fallback_policy_defaults() -> None:
    policy = FallbackPolicy()
    assert policy.max_fallbacks == 0
    assert policy.candidates is None
    assert policy.fallback_on is None
    assert policy.include_validation_failures is True
    assert policy.cross_provider is True


def test_fallback_negative_budget_rejected() -> None:
    with pytest.raises(ValidationError):
        FallbackPolicy(max_fallbacks=-1)


def test_fallback_empty_candidate_list_rejected() -> None:
    with pytest.raises(ValidationError):
        FallbackPolicy(candidates=())


def test_fallback_accepts_candidate_list() -> None:
    policy = FallbackPolicy(max_fallbacks=1, candidates=(CandidateRef(provider="p", model="m"),))
    assert policy.candidates is not None and len(policy.candidates) == 1


def test_routing_context_is_empty_in_v1() -> None:
    context = RoutingContext()
    assert dict(context.metadata) == {}


def test_routing_policy_protocol_is_satisfied_by_a_concrete_policy() -> None:
    class FirstCandidatePolicy:
        def select(
            self,
            request: LLMRequest,
            candidates: list[ModelConfig],
            context: RoutingContext,
        ) -> tuple[CandidateRef, ...]:
            return tuple(CandidateRef(provider=m.provider, model=m.model) for m in candidates)

    policy: RoutingPolicy = FirstCandidatePolicy()
    result = policy.select(
        VALID_REQUEST,
        [ModelConfig(provider="p", model="m")],
        RoutingContext(),
    )
    assert result == (CandidateRef(provider="p", model="m"),)
