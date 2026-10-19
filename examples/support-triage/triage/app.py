"""Support-ticket triage CLI built entirely on the public llm-orchestrator API.

Workflow: ``LLMRequest`` -> ``ModelRouter`` (Groq preferred) ->
``FallbackExecutor`` (with a ``RetryEngine`` inside) ->
``validate_structured_output`` -> ``TicketClassification``.

All business rules (prompt, schema, exit codes) live here in the consumer;
the library is never imported below its documented public modules. Credentials
are read only through the library's documented ``build_registry`` env
resolution, and no key is ever printed or logged.
"""

from __future__ import annotations

import asyncio
import json
import os
import random
import sys
from collections.abc import Awaitable, Callable

from llm_orchestrator.config import OrchestratorSettings, build_registry
from llm_orchestrator.contracts import (
    ErrorInfo,
    FallbackPolicy,
    LLMRequest,
    Message,
    MessageRole,
    ModelConfig,
    ProviderConfig,
    RetryPolicy,
)
from llm_orchestrator.errors import ConfigurationError, LLMError, StructuredOutputValidationError
from llm_orchestrator.providers import ProviderRegistry
from llm_orchestrator.reliability import FallbackExecutor, RetryEngine
from llm_orchestrator.routing import ModelRouter, StaticRoutingPolicy
from llm_orchestrator.structured import validate_structured_output

from .schemas import TicketClassification

EXIT_OK = 0
EXIT_USAGE = 1
EXIT_CONFIG = 2
EXIT_INVALID_OUTPUT = 3
EXIT_UPSTREAM = 4

SYSTEM_PROMPT = (
    "You are a support-triage assistant. Classify the ticket you receive and "
    "respond with JSON only, matching this shape exactly: "
    '{"category": "billing|account_access|bug_report|feature_request|how_to|other", '
    '"urgency": "low|normal|high|critical", "summary": "<one sentence>", '
    '"suggested_reply": "<one friendly sentence>", "confidence": <0.0-1.0>}'
)


class TriageFailed(Exception):
    """The orchestration run failed; carries the normalized ``ErrorInfo``."""

    def __init__(self, error: ErrorInfo) -> None:
        super().__init__(error.message)
        self.error = error


def default_settings() -> OrchestratorSettings:
    """Production settings: Groq preferred, Gemini fallback, three attempts."""
    return OrchestratorSettings(
        providers=(
            ProviderConfig(name="groq", credential_env_var="GROQ_API_KEY"),
            ProviderConfig(name="gemini", credential_env_var="GEMINI_API_KEY"),
        ),
        models=(
            ModelConfig(
                provider="groq",
                model=os.environ.get("TRIAGE_GROQ_MODEL", "llama-3.3-70b-versatile"),
            ),
            ModelConfig(
                provider="gemini",
                model=os.environ.get("TRIAGE_GEMINI_MODEL", "gemini-2.0-flash"),
            ),
        ),
        retry=RetryPolicy(max_attempts=3, initial_backoff_s=0.5),
        fallback=FallbackPolicy(max_fallbacks=1, cross_provider=True),
        routing=StaticRoutingPolicy(preferred_provider="groq"),
    )


def build_request(ticket: str) -> LLMRequest:
    """Wrap one ticket text in the system/user messages the models expect."""
    return LLMRequest(
        messages=(
            Message(role=MessageRole.SYSTEM, content=SYSTEM_PROMPT),
            Message(role=MessageRole.USER, content=ticket),
        )
    )


def _retry_engine(
    settings: OrchestratorSettings,
    sleep: Callable[[float], Awaitable[None]] | None,
    rng: random.Random | None,
) -> RetryEngine:
    """Build the retry engine, honouring injectable sleep/rng for tests."""
    if sleep is None and rng is None:
        return RetryEngine(settings.retry)
    if sleep is None:
        return RetryEngine(settings.retry, rng=rng)
    if rng is None:
        return RetryEngine(settings.retry, sleep=sleep)
    return RetryEngine(settings.retry, sleep=sleep, rng=rng)


async def triage_ticket(
    ticket: str,
    *,
    registry: ProviderRegistry,
    settings: OrchestratorSettings,
    sleep: Callable[[float], Awaitable[None]] | None = None,
    rng: random.Random | None = None,
) -> TicketClassification:
    """Classify one ticket through routing, retry, fallback, and validation."""
    request = build_request(ticket)
    router = ModelRouter(settings.routing)
    initial = router.select(request, settings.models)
    executor = FallbackExecutor(registry, _retry_engine(settings, sleep, rng), settings.fallback)
    result = await executor.run(request, settings.models, initial=initial)
    if not result.succeeded or result.response is None:
        assert result.error is not None, "failed OrchestrationResult must carry an ErrorInfo"
        raise TriageFailed(result.error)
    return validate_structured_output(result.response, TicketClassification)


async def triage_ticket_single(
    ticket: str,
    *,
    registry: ProviderRegistry,
    settings: OrchestratorSettings,
    sleep: Callable[[float], Awaitable[None]] | None = None,
    rng: random.Random | None = None,
) -> TicketClassification:
    """Same workflow through the LangGraph ``Orchestrator`` (extra: langgraph).

    Imported lazily so a core-only install can use this module for everything
    else: importing ``llm_orchestrator.orchestration`` without the extra
    raises an actionable ``ConfigurationError`` (see docs/configuration.md).
    """
    from llm_orchestrator.orchestration import Orchestrator

    request = build_request(ticket)
    router = ModelRouter(settings.routing)
    model = router.select(request, settings.models)
    orchestrator = Orchestrator(registry, _retry_engine(settings, sleep, rng))
    result = await orchestrator.run(request, model)
    if not result.succeeded or result.response is None:
        assert result.error is not None, "failed OrchestrationResult must carry an ErrorInfo"
        raise TriageFailed(result.error)
    return validate_structured_output(result.response, TicketClassification)


def main(argv: list[str] | None = None) -> int:
    """Run the CLI: ``python -m triage.app "<ticket text>"``."""
    args = sys.argv if argv is None else argv
    if len(args) != 2:
        print('usage: python -m triage.app "<ticket text>"', file=sys.stderr)
        return EXIT_USAGE
    settings = default_settings()
    try:
        registry = build_registry(settings)
    except ConfigurationError as exc:
        print(f"configuration error: {exc}", file=sys.stderr)
        return EXIT_CONFIG
    try:
        classification = asyncio.run(triage_ticket(args[1], registry=registry, settings=settings))
    except ConfigurationError as exc:
        print(f"configuration error: {exc}", file=sys.stderr)
        return EXIT_CONFIG
    except StructuredOutputValidationError as exc:
        print(f"classification output invalid: {exc}", file=sys.stderr)
        return EXIT_INVALID_OUTPUT
    except TriageFailed as exc:
        print(f"triage failed ({exc.error.category}): {exc.error.message}", file=sys.stderr)
        return EXIT_UPSTREAM
    except LLMError as exc:
        print(f"llm error: {exc}", file=sys.stderr)
        return EXIT_UPSTREAM
    print(json.dumps(classification.model_dump(mode="json"), indent=2))
    return EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main())
