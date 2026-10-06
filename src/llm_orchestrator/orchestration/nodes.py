"""Graph nodes for single-candidate orchestration.

Nodes orchestrate existing infrastructure — they never reimplement it:

- provider execution belongs to :class:`LLMProvider` implementations,
- retries belong to :class:`RetryEngine`,
- failure semantics belong to :class:`LLMError`.

The ``execute_model`` node resolves the provider for the current candidate,
runs exactly one :class:`RetryEngine` execution, and records the terminal
outcome (response *or* normalized error, attempt records, domain result).
Only normalized ``LLMError`` is captured; programming bugs and cancellation
propagate untouched.
"""

from __future__ import annotations

import time
from typing import Any

from ..contracts import (
    AttemptRecord,
    CandidateRef,
    ErrorInfo,
    LLMRequest,
    LLMResponse,
    ModelConfig,
    OrchestrationResult,
)
from ..errors import LLMError
from ..providers import LLMProvider, ProviderRegistry
from ..reliability import RetryEngine
from .state import OrchestrationState


class _AttemptCountingProvider:
    """Wraps a provider to count ``generate`` calls for attempt records.

    Lifecycle stays with the wrapped provider: :meth:`aclose` is a deliberate
    no-op here so the graph never closes a provider it does not own.
    """

    def __init__(self, inner: LLMProvider) -> None:
        self._inner = inner
        self.provider_id = inner.provider_id
        self.calls = 0

    async def generate(
        self,
        request: LLMRequest,
        model: ModelConfig,
        *,
        timeout_s: float | None = None,
    ) -> LLMResponse:
        self.calls += 1
        return await self._inner.generate(request, model, timeout_s=timeout_s)

    async def aclose(self) -> None:
        """No-op: the wrapped provider's lifecycle is owned by its registrant."""


def _failure_result(
    error: LLMError,
    *,
    request: LLMRequest,
    model: ModelConfig,
    attempts: tuple[AttemptRecord, ...],
    total_latency_ms: float,
) -> dict[str, Any]:
    return {
        "response": None,
        "error": error,
        "attempts": attempts,
        "result": OrchestrationResult(
            request_id=request.request_id,
            succeeded=False,
            response=None,
            error=ErrorInfo.from_error(error),
            candidates_tried=(CandidateRef(provider=model.provider, model=model.model),),
            attempts=attempts,
            total_latency_ms=total_latency_ms,
        ),
    }


async def execute_model(
    state: OrchestrationState,
    *,
    registry: ProviderRegistry,
    retry_engine: RetryEngine,
) -> dict[str, Any]:
    """Run the current candidate through the retry engine and record the outcome.

    Returns a state update; never mutates shared state. The request, model,
    and timeout pass through untouched. Registry misses surface as normalized
    ``LLMError`` failures (``ProviderNotRegisteredError`` is one); anything
    that is not an ``LLMError`` propagates to the caller.
    """
    request = state["request"]
    model = state["model"]
    timeout_s = state.get("timeout_s")
    candidates = (CandidateRef(provider=model.provider, model=model.model),)
    start = time.perf_counter()

    try:
        provider = registry.get(model.provider)
    except LLMError as exc:
        # No provider attempt happened; record the miss without fabricating one.
        total_ms = (time.perf_counter() - start) * 1000.0
        return _failure_result(
            exc,
            request=request,
            model=model,
            attempts=(),
            total_latency_ms=total_ms,
        )

    counting = _AttemptCountingProvider(provider)
    try:
        response = await retry_engine.generate(counting, request, model, timeout_s=timeout_s)
    except LLMError as exc:
        total_ms = (time.perf_counter() - start) * 1000.0
        record = AttemptRecord(
            provider=model.provider,
            model=model.model,
            attempt=exc.attempt if exc.attempt else counting.calls,
            succeeded=False,
            error_category=exc.category,
        )
        return _failure_result(
            exc,
            request=request,
            model=model,
            attempts=(record,),
            total_latency_ms=total_ms,
        )

    total_ms = (time.perf_counter() - start) * 1000.0
    record = AttemptRecord(
        provider=model.provider,
        model=model.model,
        attempt=counting.calls,
        succeeded=True,
        latency_ms=response.latency_ms,
    )
    return {
        "response": response,
        "error": None,
        "attempts": (record,),
        "result": OrchestrationResult(
            request_id=request.request_id,
            succeeded=True,
            response=response,
            error=None,
            candidates_tried=candidates,
            attempts=(record,),
            total_latency_ms=total_ms,
        ),
    }
