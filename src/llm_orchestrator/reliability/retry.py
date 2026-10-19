"""Per-candidate retry engine.

A :class:`RetryEngine` wraps one :class:`LLMProvider` call with bounded,
policy-driven retries of the *same* provider, *same* model, and *same*
request. It never switches provider or model, never reorders candidates, and
never performs fallback — that is a later phase's job.

Placement in the stack::

    Provider SDK
        ↓ (normalized by the adapter)
    Normalized LLMError
        ↓ (classified here)
    RetryEngine → same provider.generate(...) again, or propagate

``max_attempts`` counts total provider attempts: ``1`` means one attempt and
zero retries. Only errors the normalized taxonomy marks retryable (or that
``RetryPolicy.retry_on`` explicitly lists) are retried; anything else —
including unexpected non-``LLMError`` exceptions and task cancellation —
propagates immediately without a retry.
"""

from __future__ import annotations

import asyncio
import math
import random
from collections.abc import Awaitable, Callable

from ..contracts import LLMRequest, LLMResponse, ModelConfig, RetryPolicy
from ..errors import LLMError, require_timeout_s
from ..providers import LLMProvider
from .backoff import compute_backoff_delay

#: An injectable async sleep. Tests record calls instead of waiting.
SleepFn = Callable[[float], Awaitable[None]]


def _is_retryable(exc: LLMError, policy: RetryPolicy) -> bool:
    """Whether ``exc`` qualifies for another attempt under ``policy``.

    An explicit ``retry_on`` category set overrides the taxonomy defaults;
    otherwise each error's own ``retryable`` flag (which already folds in any
    per-instance override) decides.
    """
    if policy.retry_on is not None:
        return exc.category in policy.retry_on
    return exc.retryable


def _honor_retry_after(exc: LLMError, delay_s: float, policy: RetryPolicy) -> float:
    """Raise the delay to a server-provided ``retry_after_s`` when sane.

    Only errors carrying that attribute (today: ``RateLimitError``) are
    affected. Invalid values (missing, non-numeric, negative, non-finite) are
    ignored, and the policy maximum is never exceeded.
    """
    retry_after = getattr(exc, "retry_after_s", None)
    if isinstance(retry_after, bool) or not isinstance(retry_after, int | float):
        return delay_s
    if not math.isfinite(retry_after) or retry_after < 0:
        return delay_s
    return min(policy.max_backoff_s, max(delay_s, float(retry_after)))


class RetryEngine:
    """Executes one provider call with bounded retries of the same candidate.

    The engine is stateless across invocations: all attempt state lives in
    locals of :meth:`generate`, so concurrent calls never interfere. The
    ``sleep`` callable and ``rng`` are injected for deterministic tests.
    """

    def __init__(
        self,
        policy: RetryPolicy | None = None,
        *,
        sleep: SleepFn = asyncio.sleep,
        rng: random.Random | None = None,
    ) -> None:
        self._policy = policy if policy is not None else RetryPolicy()
        self._sleep = sleep
        self._rng = rng

    @property
    def policy(self) -> RetryPolicy:
        """The policy this engine retries under."""
        return self._policy

    async def generate(
        self,
        provider: LLMProvider,
        request: LLMRequest,
        model: ModelConfig,
        *,
        timeout_s: float | None = None,
    ) -> LLMResponse:
        """Attempt ``provider.generate`` up to ``policy.max_attempts`` times.

        The same ``provider``, ``model``, and ``request`` are used for every
        attempt; the request is never mutated. On success the response returns
        immediately. When attempts are exhausted (or the error is not
        retryable), the *original normalized error* propagates with its
        ``attempt`` field stamped — never a replacement exception.
        """
        require_timeout_s(timeout_s)
        policy = self._policy
        previous_delay_s: float | None = None
        attempt = 0
        while True:
            attempt += 1
            try:
                return await provider.generate(request, model, timeout_s=timeout_s)
            except asyncio.CancelledError:
                # Cancellation is not a failure to retry; propagate at once.
                raise
            except LLMError as exc:
                exc.attempt = attempt
                if attempt >= policy.max_attempts or not _is_retryable(exc, policy):
                    raise
                delay_s = compute_backoff_delay(
                    policy,
                    retry_index=attempt - 1,
                    rng=self._rng,
                    previous_delay_s=previous_delay_s,
                )
                delay_s = _honor_retry_after(exc, delay_s, policy)
                previous_delay_s = delay_s
                await self._sleep(delay_s)
            except Exception:
                # Unknown failures (programming bugs, unexpected transports)
                # must not trigger blind retries.
                raise
