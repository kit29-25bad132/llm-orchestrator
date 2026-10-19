"""Scripted ``LLMProvider`` doubles for offline consumer tests.

``llm_orchestrator.providers.LLMProvider`` is a runtime-checkable protocol,
so a consumer may supply its own implementation without inheriting from a
library base class. Each fake replays a scripted sequence of outcomes and
records every call, letting tests assert routing, retry, and fallback
behaviour deterministically while the library's real executor, retry engine,
and router do all the work.
"""

from __future__ import annotations

import json

from llm_orchestrator.contracts import FinishReason, LLMRequest, LLMResponse, ModelConfig

Outcome = LLMResponse | BaseException


def make_response(provider: str, model: str, content: str) -> LLMResponse:
    """Build a well-formed canned response, stamped with the serving model."""
    return LLMResponse(
        request_id="req-scripted",
        provider=provider,
        model=model,
        finish_reason=FinishReason.STOP,
        latency_ms=1.0,
        content=content,
    )


def classification_payload(
    category: str = "billing",
    urgency: str = "high",
    summary: str = "Duplicate charge on October invoice",
    suggested_reply: str = "We have refunded the extra charge already.",
    confidence: float = 0.93,
) -> str:
    """A JSON string in the ``TicketClassification`` shape, for canned output."""
    return json.dumps(
        {
            "category": category,
            "urgency": urgency,
            "summary": summary,
            "suggested_reply": suggested_reply,
            "confidence": confidence,
        }
    )


class ScriptedProvider:
    """A ``LLMProvider`` that replays scripted outcomes, in order."""

    def __init__(self, provider_id: str, outcomes: list[Outcome]) -> None:
        self.provider_id = provider_id
        self._outcomes = list(outcomes)
        self.calls: list[ModelConfig] = []

    async def generate(
        self,
        request: LLMRequest,
        model: ModelConfig,
        *,
        timeout_s: float | None = None,
    ) -> LLMResponse:
        self.calls.append(model)
        if not self._outcomes:
            raise AssertionError(
                f"ScriptedProvider {self.provider_id!r} ran out of scripted outcomes"
            )
        outcome = self._outcomes.pop(0)
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome.model_copy(update={"provider": model.provider, "model": model.model})

    async def aclose(self) -> None:
        return None
