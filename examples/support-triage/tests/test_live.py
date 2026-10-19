"""Opt-in live Groq round-trip through the consumer app (skipped by default).

Follows the repository's established live-test convention (see
``tests/integration/test_groq_live.py`` in the library): the test runs only
when ``RUN_GROQ_INTEGRATION=1`` is set together with ``GROQ_API_KEY`` and
``GROQ_TEST_MODEL``. Credentials are read through the library's documented
``build_registry`` env resolution and are never printed.

The adapter import happens inside ``build_registry`` (called from within the
test body) so this module still *collects* in a core-only environment, where a
top-level ``llm_orchestrator.providers.groq`` import would raise
``ConfigurationError`` for the missing extra.
"""

from __future__ import annotations

import asyncio
import os

import pytest
from triage.app import triage_ticket
from triage.schemas import TicketClassification

from llm_orchestrator.config import OrchestratorSettings, build_registry
from llm_orchestrator.contracts import FallbackPolicy, ModelConfig, RetryPolicy
from llm_orchestrator.routing import StaticRoutingPolicy

pytestmark = pytest.mark.live


def _live_config() -> tuple[str, str] | None:
    if os.environ.get("RUN_GROQ_INTEGRATION") != "1":
        return None
    api_key = os.environ.get("GROQ_API_KEY")
    model = os.environ.get("GROQ_TEST_MODEL")
    if not api_key or not model:
        return None
    return api_key, model


@pytest.mark.skipif(_live_config() is None, reason="live Groq integration not enabled")
def test_live_groq_triage_round_trip() -> None:
    assert _live_config() is not None
    settings = OrchestratorSettings(
        providers=[],
        models=(ModelConfig(provider="groq", model=os.environ["GROQ_TEST_MODEL"]),),
        retry=RetryPolicy(max_attempts=2, initial_backoff_s=0.5),
        fallback=FallbackPolicy(max_fallbacks=0),
        routing=StaticRoutingPolicy(preferred_provider="groq"),
    )
    registry = build_registry(settings)  # reads GROQ_API_KEY from the real environment
    classification = asyncio.run(
        triage_ticket(
            "I was charged twice this month and need the extra charge refunded.",
            registry=registry,
            settings=settings,
        )
    )
    assert isinstance(classification, TicketClassification)
    assert classification.category in {
        "billing",
        "account_access",
        "bug_report",
        "feature_request",
        "how_to",
        "other",
    }
    assert classification.urgency in {"low", "normal", "high", "critical"}
    assert 0.0 <= classification.confidence <= 1.0
