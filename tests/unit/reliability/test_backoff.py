"""Unit tests for pure backoff delay calculation.

No provider, no sleeping, no I/O: these pin the delay formula, the cap, and
the jitter bounds directly against ``RetryPolicy`` semantics.
"""

from __future__ import annotations

import random

import pytest

from llm_orchestrator.contracts import JitterStrategy, RetryPolicy
from llm_orchestrator.reliability.backoff import compute_backoff_delay


def _policy(**kwargs: object) -> RetryPolicy:
    return RetryPolicy(jitter=JitterStrategy.NONE, **kwargs)  # type: ignore[arg-type]


def test_no_jitter_follows_exact_exponential_sequence() -> None:
    policy = _policy(initial_backoff_s=0.5, multiplier=2.0, max_backoff_s=30.0)
    assert [compute_backoff_delay(policy, retry_index=i) for i in range(4)] == [
        0.5,
        1.0,
        2.0,
        4.0,
    ]


def test_delay_never_exceeds_max_backoff() -> None:
    policy = _policy(initial_backoff_s=1.0, multiplier=10.0, max_backoff_s=5.0)
    assert [compute_backoff_delay(policy, retry_index=i) for i in range(4)] == [
        1.0,
        5.0,
        5.0,
        5.0,
    ]


def test_multiplier_of_one_yields_constant_delay() -> None:
    policy = _policy(initial_backoff_s=2.0, multiplier=1.0, max_backoff_s=30.0)
    assert [compute_backoff_delay(policy, retry_index=i) for i in range(3)] == [
        2.0,
        2.0,
        2.0,
    ]


def test_zero_backoff_window_yields_zero_delay() -> None:
    policy = _policy(initial_backoff_s=0.0, multiplier=2.0, max_backoff_s=0.0)
    assert compute_backoff_delay(policy, retry_index=3) == 0.0


def test_negative_retry_index_is_rejected() -> None:
    with pytest.raises(ValueError):
        compute_backoff_delay(_policy(), retry_index=-1)


def test_full_jitter_stays_within_bounds_and_is_reproducible() -> None:
    policy = RetryPolicy(
        initial_backoff_s=1.0, multiplier=2.0, max_backoff_s=8.0, jitter=JitterStrategy.FULL
    )
    first = [compute_backoff_delay(policy, retry_index=i, rng=random.Random(42)) for i in range(4)]
    second = [compute_backoff_delay(policy, retry_index=i, rng=random.Random(42)) for i in range(4)]
    assert first == second
    for i, delay in enumerate(first):
        base = min(8.0, 2.0**i)
        assert 0.0 <= delay <= base


def test_equal_jitter_keeps_half_fixed() -> None:
    policy = RetryPolicy(
        initial_backoff_s=4.0, multiplier=1.0, max_backoff_s=30.0, jitter=JitterStrategy.EQUAL
    )
    for _ in range(25):
        delay = compute_backoff_delay(policy, retry_index=0, rng=random.Random())
        assert 2.0 <= delay <= 4.0


def test_decorrelated_jitter_is_bounded_and_uses_previous_delay() -> None:
    policy = RetryPolicy(
        initial_backoff_s=0.5,
        multiplier=2.0,
        max_backoff_s=30.0,
        jitter=JitterStrategy.DECORRELATED,
    )
    delay = compute_backoff_delay(policy, retry_index=0, rng=random.Random(7))
    assert 0.0 <= delay <= policy.max_backoff_s
    chained = compute_backoff_delay(
        policy, retry_index=1, rng=random.Random(7), previous_delay_s=delay
    )
    assert 0.0 <= chained <= min(policy.max_backoff_s, 3.0 * delay)
