"""LangGraph state for single-candidate orchestration.

The state threads domain contracts through the graph — it never redefines
them. ``LLMRequest``, ``ModelConfig``, ``LLMResponse``, ``LLMError``,
``AttemptRecord``, and ``OrchestrationResult`` stay canonical in
``contracts``/``errors``; this module only declares which of them each node
reads and writes.

The live ``LLMError`` is kept in state (alongside the serializable
``OrchestrationResult``) so future nodes — fallback, validation,
observability — can branch on full failure semantics without re-deriving
them. State is per-invocation and never shared between runs.
"""

from __future__ import annotations

from typing import TypedDict

from ..contracts import (
    AttemptRecord,
    LLMRequest,
    LLMResponse,
    ModelConfig,
    OrchestrationResult,
)
from ..errors import LLMError


class OrchestrationState(TypedDict, total=False):
    """Typed workflow state for one orchestration run.

    Inputs (set by the caller): ``request``, ``model``, ``timeout_s``.
    Written by ``execute_model``: ``response``/``error``, ``attempts``,
    ``result``. Exactly one of ``response``/``error`` is set on completion.
    """

    request: LLMRequest
    model: ModelConfig
    timeout_s: float | None
    response: LLMResponse | None
    error: LLMError | None
    attempts: tuple[AttemptRecord, ...]
    result: OrchestrationResult


def initial_state(
    request: LLMRequest,
    model: ModelConfig,
    *,
    timeout_s: float | None = None,
) -> OrchestrationState:
    """Build the input state for one orchestration run."""
    return {"request": request, "model": model, "timeout_s": timeout_s}
