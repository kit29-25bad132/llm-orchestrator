"""Backoff delay calculation for the retry engine.

This module is pure policy math: given a :class:`RetryPolicy` and which retry
is being scheduled, it returns how long to wait. It performs no I/O, sleeps
nothing, and knows nothing about providers. The retry loop in ``retry.py``
decides *whether* to wait; this module decides *how long*.

Delay model
-----------

For the ``retry_index``-th sleep (0-based: ``0`` is the wait before the second
attempt)::

    base = min(max_backoff_s, initial_backoff_s * multiplier ** retry_index)

Jitter is then applied according to ``policy.jitter``:

- ``NONE``: ``base`` exactly (fully deterministic).
- ``FULL``: uniform in ``[0, base]``.
- ``EQUAL``: ``base / 2 + uniform(0, base / 2)`` (half fixed, half jittered).
- ``DECORRELATED``: uniform in ``[initial_backoff_s, min(max_backoff_s,
  3 * previous)]`` where ``previous`` is the last actual sleep (seeded with
  ``initial_backoff_s`` for the first retry).

The result is always clamped to ``[0, max_backoff_s]``, so no caller can ever
sleep a negative or unbounded duration.
"""

from __future__ import annotations

import random

from ..contracts import JitterStrategy, RetryPolicy


def compute_backoff_delay(
    policy: RetryPolicy,
    *,
    retry_index: int,
    rng: random.Random | None = None,
    previous_delay_s: float | None = None,
) -> float:
    """Return the sleep duration before the next attempt.

    ``retry_index`` is 0-based (``0`` schedules the wait before attempt 2).
    ``rng`` supplies randomness; when omitted a fresh PRNG is used so no
    shared global state is touched. Tests should pass ``jitter=NONE`` for
    exact sequences or a seeded ``random.Random`` for reproducible jitter.
    """
    if retry_index < 0:
        raise ValueError("retry_index must be >= 0")
    source = rng if rng is not None else random.Random()
    base = min(
        policy.max_backoff_s,
        policy.initial_backoff_s * (policy.multiplier**retry_index),
    )
    delay = _apply_jitter(policy, base=base, source=source, previous_delay_s=previous_delay_s)
    return min(max(delay, 0.0), policy.max_backoff_s)


def _apply_jitter(
    policy: RetryPolicy,
    *,
    base: float,
    source: random.Random,
    previous_delay_s: float | None,
) -> float:
    match policy.jitter:
        case JitterStrategy.NONE:
            return base
        case JitterStrategy.FULL:
            return source.uniform(0.0, base)
        case JitterStrategy.EQUAL:
            return base / 2.0 + source.uniform(0.0, base / 2.0)
        case JitterStrategy.DECORRELATED:
            previous = (
                previous_delay_s if previous_delay_s is not None else policy.initial_backoff_s
            )
            ceiling = min(policy.max_backoff_s, 3.0 * max(previous, 0.0))
            floor = min(policy.initial_backoff_s, ceiling)
            return source.uniform(floor, ceiling)
