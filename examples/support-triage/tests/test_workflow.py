"""Offline workflow tests: happy path, preferred routing, retry, fallback, validation.

The library's real components (``ModelRouter``, ``RetryEngine`` inside
``FallbackExecutor``, ``validate_structured_output``) perform all orchestration
through the consumer workflow in ``triage.app``; only the provider is a
consumer-side scripted fake.
"""

from __future__ import annotations

import asyncio
import random
import sysconfig
from collections.abc import Awaitable
from pathlib import Path
from typing import TypeVar

import pytest
from triage.app import build_request, triage_ticket
from triage.fakes import ScriptedProvider, classification_payload, make_response

import llm_orchestrator
from llm_orchestrator.config import OrchestratorSettings
from llm_orchestrator.contracts import FallbackPolicy, ModelConfig, RetryPolicy
from llm_orchestrator.errors import (
    ErrorCategory,
    ProviderUnavailableError,
    StructuredOutputValidationError,
)
from llm_orchestrator.providers import ProviderRegistry
from llm_orchestrator.routing import ModelRouter, SelectionReason, StaticRoutingPolicy

T = TypeVar("T")


def run(awaitable: Awaitable[T]) -> T:
    """Drive a coroutine to completion without an async test plugin."""
    return asyncio.run(awaitable)


def offline_settings() -> OrchestratorSettings:
    """Groq-preferred, Gemini-fallback settings with fast, seeded retries."""
    return OrchestratorSettings(
        models=(
            ModelConfig(provider="groq", model="groq-triage-model"),
            ModelConfig(provider="gemini", model="gemini-triage-model"),
        ),
        retry=RetryPolicy(max_attempts=3, initial_backoff_s=0.05),
        fallback=FallbackPolicy(max_fallbacks=1, cross_provider=True),
        routing=StaticRoutingPolicy(preferred_provider="groq"),
    )


def test_package_is_imported_from_the_installed_wheel() -> None:
    package_path = Path(llm_orchestrator.__file__).resolve()
    site_packages = Path(sysconfig.get_paths()["purelib"]).resolve()
    repo_src = Path(__file__).resolve().parents[3] / "src" / "llm_orchestrator"
    assert package_path.is_relative_to(site_packages), (
        f"not in running interpreter's site-packages: {package_path}"
    )
    assert not package_path.is_relative_to(repo_src), (
        f"resolved to the repository source tree: {package_path}"
    )
    assert llm_orchestrator.__version__ == "0.1.0"


def test_groq_preferred_when_both_models_are_eligible() -> None:
    settings = offline_settings()
    request = build_request("My invoice looks wrong this month")
    router = ModelRouter(settings.routing)
    selected = router.select(request, settings.models)
    decision = router.explain(request, settings.models)
    assert selected.provider == "groq"
    assert selected.model == "groq-triage-model"
    assert decision.reason == SelectionReason.PREFERRED_PROVIDER
    assert decision.selected.provider == "groq"
    assert {candidate.provider for candidate in decision.eligible} == {"groq", "gemini"}


def test_happy_path_classifies_ticket_with_groq() -> None:
    groq = ScriptedProvider(
        "groq", [make_response("groq", "groq-triage-model", classification_payload())]
    )
    registry = ProviderRegistry()
    registry.register(groq)
    classification = run(
        triage_ticket(
            "I was charged twice for October", registry=registry, settings=offline_settings()
        )
    )
    assert classification.category == "billing"
    assert classification.urgency == "high"
    assert classification.summary == "Duplicate charge on October invoice"
    assert classification.confidence == pytest.approx(0.93)
    assert [call.model for call in groq.calls] == ["groq-triage-model"]


def test_transient_failure_retries_then_succeeds_deterministically() -> None:
    settings = offline_settings()

    def fresh_groq() -> ScriptedProvider:
        return ScriptedProvider(
            "groq",
            [
                ProviderUnavailableError("transient upstream failure"),
                make_response("groq", "groq-triage-model", classification_payload()),
            ],
        )

    def attempt() -> tuple[ScriptedProvider, list[float]]:
        groq = fresh_groq()
        registry = ProviderRegistry()
        registry.register(groq)
        delays: list[float] = []

        async def recording_sleep(delay: float) -> None:
            delays.append(delay)

        classification = run(
            triage_ticket(
                "Site is down for my team",
                registry=registry,
                settings=settings,
                sleep=recording_sleep,
                rng=random.Random(7),
            )
        )
        assert classification.category == "billing"
        return groq, delays

    groq, delays = attempt()
    assert len(groq.calls) == 2
    assert len(delays) == 1
    assert 0.0 <= delays[0] <= settings.retry.initial_backoff_s

    groq_again, delays_again = attempt()
    assert len(groq_again.calls) == 2
    assert delays_again == delays  # same seed -> identical recorded backoff


def test_exhausted_retries_fall_back_to_gemini() -> None:
    groq = ScriptedProvider(
        "groq",
        [
            ProviderUnavailableError("groq down"),
            ProviderUnavailableError("groq still down"),
            ProviderUnavailableError("groq down again"),
        ],
    )
    gemini = ScriptedProvider(
        "gemini",
        [
            make_response(
                "gemini",
                "gemini-triage-model",
                classification_payload(summary="Duplicate charge reported for October invoice"),
            )
        ],
    )
    registry = ProviderRegistry()
    registry.register(groq)
    registry.register(gemini)
    delays: list[float] = []

    async def recording_sleep(delay: float) -> None:
        delays.append(delay)

    classification = run(
        triage_ticket(
            "I was charged twice for October",
            registry=registry,
            settings=offline_settings(),
            sleep=recording_sleep,
            rng=random.Random(11),
        )
    )
    assert len(groq.calls) == 3  # every attempt exhausted before handoff
    assert [call.model for call in gemini.calls] == ["gemini-triage-model"]
    assert classification.summary == "Duplicate charge reported for October invoice"
    assert len(delays) == 2  # backoff between the three attempts, then fallback


def test_invalid_structured_output_raises_documented_validation_error() -> None:
    groq = ScriptedProvider(
        "groq",
        [make_response("groq", "groq-triage-model", "I could not classify this ticket.")],
    )
    registry = ProviderRegistry()
    registry.register(groq)
    with pytest.raises(StructuredOutputValidationError) as exc_info:
        run(triage_ticket("Something odd happened", registry=registry, settings=offline_settings()))
    assert exc_info.value.category is ErrorCategory.STRUCTURED_OUTPUT_VALIDATION
    assert [call.model for call in groq.calls] == ["groq-triage-model"]
