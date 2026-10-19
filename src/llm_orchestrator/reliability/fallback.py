"""Candidate-level fallback execution.

A :class:`FallbackExecutor` moves execution to another candidate only after
the current candidate reaches a *terminal* failure inside :class:`RetryEngine`
— it never retries, never sleeps, and never computes backoff. The layering is:

.. code-block:: text

    Routing  → which candidate to try first (an ordered ``ModelConfig`` list)
    Retry    → re-attempt the same candidate (``RetryEngine.generate``)
    Fallback → move to the next candidate (this module, on eligible failure)

Eligibility is decided from the normalized :class:`LLMError` contract
(``fallback_eligible``, optionally overridden by
``FallbackPolicy.fallback_on``). Cancellation and non-``LLMError`` programming
errors propagate untouched and never trigger a transition.

Attempt history records one :class:`AttemptRecord` per actual provider call
(``Candidate A attempt 1, attempt 2, Candidate B attempt 1, ...``). This is
deliberately finer than the single-candidate orchestration node, which keeps
one record per candidate: the fallback result must show *where* each failure
happened, and the terminal ``attempt`` count alone cannot do that.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Sequence

from ..contracts import (
    AttemptRecord,
    CandidateRef,
    ErrorInfo,
    FallbackPolicy,
    LLMRequest,
    LLMResponse,
    ModelConfig,
    OrchestrationResult,
)
from ..errors import ConfigurationError, ErrorCategory, LLMError, require_timeout_s
from ..providers import LLMProvider, ProviderRegistry
from .retry import RetryEngine


def _is_fallback_eligible(exc: LLMError, policy: FallbackPolicy) -> bool:
    """Whether ``exc`` permits moving to the next candidate under ``policy``.

    An explicit ``fallback_on`` category set is authoritative (mirroring how
    ``RetryPolicy.retry_on`` overrides taxonomy defaults). Otherwise each
    error's own ``fallback_eligible`` flag decides, except that structured
    output validation failures additionally honor
    ``include_validation_failures``.
    """
    if policy.fallback_on is not None:
        return exc.category in policy.fallback_on
    if exc.category is ErrorCategory.STRUCTURED_OUTPUT_VALIDATION:
        return policy.include_validation_failures and exc.fallback_eligible
    return exc.fallback_eligible


def _plan_candidates(
    candidates: Sequence[ModelConfig],
    initial: ModelConfig | None,
    policy: FallbackPolicy,
    *,
    request_id: str,
) -> tuple[ModelConfig, ...]:
    """Build the deterministic, bounded execution order.

    The initial candidate (Phase 10 output, or the first candidate) always
    comes first; duplicates collapse to their first occurrence; an explicit
    ``policy.candidates`` list acts as an allowlist preserving routing order
    (routing owns ordering, never the fallback layer); ``cross_provider=False``
    keeps only the initial provider; the plan is truncated to
    ``1 + max_fallbacks`` entries so transitions can never exceed the budget.
    """
    raw: list[ModelConfig] = (
        [initial, *list(candidates)] if initial is not None else list(candidates)
    )
    for entry in raw:
        if not isinstance(entry, ModelConfig):
            raise ConfigurationError(
                "fallback candidates must be ModelConfig instances, "
                f"got {type(entry).__name__} for request {request_id}",
                request_id=request_id,
            )
    ordered: list[ModelConfig] = []
    seen: set[tuple[str, str]] = set()
    for entry in raw:
        key = (entry.provider, entry.model)
        if key not in seen:
            seen.add(key)
            ordered.append(entry)
    if policy.candidates is not None:
        allowed = {(ref.provider, ref.model) for ref in policy.candidates}
        ordered = [c for c in ordered if (c.provider, c.model) in allowed]
    if not policy.cross_provider and ordered:
        first_provider = ordered[0].provider
        ordered = [c for c in ordered if c.provider == first_provider]
    plan = tuple(ordered[: policy.max_fallbacks + 1])
    if not plan:
        raise ConfigurationError(
            f"no fallback candidates remain for request {request_id}",
            request_id=request_id,
        )
    return plan


class _RecordingProvider:
    """Observes one candidate's provider calls without changing retry behavior.

    Each ``generate`` invocation appends exactly one :class:`AttemptRecord`
    (success or normalized failure) and re-raises, so :class:`RetryEngine`
    sees the untouched provider contract. Cancellation and non-``LLMError``
    failures are never recorded: they are not attempts with an outcome.
    Lifecycle stays with the wrapped provider (:meth:`aclose` is a no-op).
    """

    def __init__(
        self, inner: LLMProvider, model: ModelConfig, records: list[AttemptRecord]
    ) -> None:
        self._inner = inner
        self.provider_id = inner.provider_id
        self._model = model
        self._records = records
        self.calls = 0

    async def generate(
        self,
        request: LLMRequest,
        model: ModelConfig,
        *,
        timeout_s: float | None = None,
    ) -> LLMResponse:
        self.calls += 1
        start = time.perf_counter()
        try:
            response = await self._inner.generate(request, model, timeout_s=timeout_s)
        except LLMError as exc:
            self._records.append(
                AttemptRecord(
                    provider=self._model.provider,
                    model=self._model.model,
                    attempt=self.calls,
                    succeeded=False,
                    latency_ms=(time.perf_counter() - start) * 1000.0,
                    error_category=exc.category,
                )
            )
            raise
        self._records.append(
            AttemptRecord(
                provider=self._model.provider,
                model=self._model.model,
                attempt=self.calls,
                succeeded=True,
                latency_ms=response.latency_ms,
            )
        )
        return response

    async def aclose(self) -> None:
        """No-op: the wrapped provider's lifecycle is owned by its registrant."""


class FallbackExecutor:
    """Executes an ordered candidate plan with per-candidate retries.

    The executor is stateless across invocations: the plan, attempt records,
    and failure state live in :meth:`run` locals, so concurrent runs never
    interfere. It owns no providers and implements no retry logic.
    """

    def __init__(
        self,
        registry: ProviderRegistry,
        retry_engine: RetryEngine | None = None,
        policy: FallbackPolicy | None = None,
    ) -> None:
        self._registry = registry
        self._retry_engine = retry_engine if retry_engine is not None else RetryEngine()
        self._policy = policy if policy is not None else FallbackPolicy()

    @property
    def registry(self) -> ProviderRegistry:
        """The provider registry candidates are resolved from."""
        return self._registry

    @property
    def retry_engine(self) -> RetryEngine:
        """The engine each candidate is retried through."""
        return self._retry_engine

    @property
    def policy(self) -> FallbackPolicy:
        """The default policy used when ``run`` is called without one."""
        return self._policy

    async def run(
        self,
        request: LLMRequest,
        candidates: Sequence[ModelConfig],
        *,
        initial: ModelConfig | None = None,
        policy: FallbackPolicy | None = None,
        timeout_s: float | None = None,
    ) -> OrchestrationResult:
        """Attempt the plan in order; return the first success or terminal failure.

        ``candidates`` is the routing-ordered set; ``initial`` (default: the
        first candidate) is always attempted first. Success returns
        immediately without touching later candidates. A candidate's
        :class:`RetryEngine` failure advances only when the terminal error is
        fallback-eligible *and* the ``max_fallbacks`` budget remains;
        otherwise the last error becomes the failed result with the complete
        attempt history. ``timeout_s`` passes through to every candidate.
        """
        require_timeout_s(timeout_s)
        active = policy if policy is not None else self._policy
        plan = _plan_candidates(candidates, initial, active, request_id=request.request_id)
        start = time.perf_counter()
        records: list[AttemptRecord] = []
        tried: list[CandidateRef] = []
        last_error: LLMError | None = None

        for index, model in enumerate(plan):
            tried.append(CandidateRef(provider=model.provider, model=model.model))
            try:
                provider = self._registry.get(model.provider)
            except LLMError as exc:
                # Resolution miss with no attempts; eligibility still decides.
                last_error = exc
            else:
                recording = _RecordingProvider(provider, model, records)
                try:
                    response = await self._retry_engine.generate(
                        recording, request, model, timeout_s=timeout_s
                    )
                except asyncio.CancelledError:
                    raise
                except LLMError as exc:
                    last_error = exc
                else:
                    return OrchestrationResult(
                        request_id=request.request_id,
                        succeeded=True,
                        response=response,
                        error=None,
                        candidates_tried=tuple(tried),
                        attempts=tuple(records),
                        total_latency_ms=(time.perf_counter() - start) * 1000.0,
                    )
            if last_error is not None and (
                index >= len(plan) - 1 or not _is_fallback_eligible(last_error, active)
            ):
                return OrchestrationResult(
                    request_id=request.request_id,
                    succeeded=False,
                    response=None,
                    error=ErrorInfo.from_error(last_error),
                    candidates_tried=tuple(tried),
                    attempts=tuple(records),
                    total_latency_ms=(time.perf_counter() - start) * 1000.0,
                )
        raise RuntimeError("fallback plan completed without a result")
