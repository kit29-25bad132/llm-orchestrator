"""Deterministic offline end-to-end test of the composed orchestration path.

This suite exercises the real composition of existing public components only:

``OrchestratorSettings`` → ``load_settings`` (injected env mapping) →
``build_registry`` (injected fake SDK clients) → ``ModelRouter`` →
``RetryEngine`` + ``FallbackExecutor`` → ``OrchestrationResult``
(plus a single-candidate ``Orchestrator`` run over the same registry).

The registry holds the **real** ``GeminiProvider``/``GroqProvider`` adapters;
only the SDK clients beneath them are scripted, using real SDK response and
error types so adapter normalization is genuinely exercised. No test touches
the network, requires credentials, reads ``os.environ``, sets integration
flags, or depends on any ``.env`` file. Nothing here is marked ``live``.
"""

from __future__ import annotations

from typing import Any

import groq
import httpx
from google.genai import types as genai_types

from llm_orchestrator.config import (
    OrchestratorSettings,
    build_registry,
    load_settings,
    resolve_timeout_s,
)
from llm_orchestrator.contracts import (
    FallbackPolicy,
    JitterStrategy,
    LLMRequest,
    Message,
    MessageRole,
    ModelConfig,
    ProviderConfig,
    RetryPolicy,
)
from llm_orchestrator.errors import ErrorCategory
from llm_orchestrator.orchestration import Orchestrator
from llm_orchestrator.reliability import FallbackExecutor, RetryEngine
from llm_orchestrator.routing import ModelRouter
from llm_orchestrator.routing.static import StaticRoutingPolicy
from support.fakes import FakeGeminiSdkClient, FakeGroqSdkClient, RecordingSleep, run_sync

GEMINI_MODEL = "gemini-e2e-model"
GROQ_MODEL = "groq-e2e-model"
# Fake values shaped like real keys so redaction coverage is meaningful.
FAKE_GEMINI_KEY = "AIzaTestFakeKey0123456789abcdef"
FAKE_GROQ_KEY = "gsk_test_fake_value_0123456789abcdef"


def _gemini_success(text: str) -> Any:
    content = genai_types.Content(role="model", parts=[genai_types.Part(text=text)])
    return genai_types.GenerateContentResponse(
        model_version="gemini-e2e-v1",
        candidates=[genai_types.Candidate(content=content, finish_reason="STOP")],
        usage_metadata=genai_types.GenerateContentResponseUsageMetadata(
            prompt_token_count=3, candidates_token_count=5, total_token_count=8
        ),
        response_id="resp-e2e-1",
    )


def _groq_outage() -> groq.InternalServerError:
    request = httpx.Request("POST", "https://api.groq.com/openai/v1/chat/completions")
    response = httpx.Response(503, request=request)
    return groq.InternalServerError("service unavailable", response=response, body=None)


def _groq_success() -> Any:
    from groq.types.chat import ChatCompletion
    from groq.types.chat.chat_completion import Choice
    from groq.types.chat.chat_completion_message import ChatCompletionMessage
    from groq.types.completion_usage import CompletionUsage

    return ChatCompletion(
        id="chatcmpl-e2e-1",
        created=1700000000,
        model=GROQ_MODEL,
        object="chat.completion",
        choices=[
            Choice(
                index=0,
                finish_reason="stop",  # type: ignore[arg-type]
                message=ChatCompletionMessage(role="assistant", content="groq says hi"),
            )
        ],
        usage=CompletionUsage(prompt_tokens=3, completion_tokens=5, total_tokens=8),
    )


def _explicit_settings() -> OrchestratorSettings:
    return OrchestratorSettings(
        providers=(
            ProviderConfig(name="gemini", credential_env_var="GEMINI_API_KEY"),
            ProviderConfig(name="groq", credential_env_var="GROQ_API_KEY"),
        ),
        models=(
            ModelConfig(provider="groq", model=GROQ_MODEL),
            ModelConfig(provider="gemini", model=GEMINI_MODEL),
        ),
        retry=RetryPolicy(max_attempts=2, initial_backoff_s=0.0, jitter=JitterStrategy.NONE),
        fallback=FallbackPolicy(max_fallbacks=1),
        routing=StaticRoutingPolicy(preferred_provider="groq"),
    )


def _request() -> LLMRequest:
    return LLMRequest(
        messages=[Message(role=MessageRole.USER, content="hello")],
        request_id="req-e2e-1",
    )


def test_offline_end_to_end_failure_then_cross_provider_recovery() -> None:
    """Groq 503s exhaust its retry budget; the run recovers via Gemini.

    Full path: settings → env overlay → registry (real adapters, fake SDK
    clients) → routing → retry/fallback execution → result.
    """
    settings = load_settings(
        explicit=_explicit_settings(),
        env={"LLM_ORCH_DEFAULT_TIMEOUT_S": "9.5"},
    )
    assert settings.default_timeout_s == 9.5
    assert settings.retry.max_attempts == 2
    assert settings.fallback.max_fallbacks == 1

    groq_sdk = FakeGroqSdkClient([_groq_outage()])
    gemini_sdk = FakeGeminiSdkClient([_gemini_success("recovered via gemini")])
    registry = build_registry(
        settings,
        env={"GEMINI_API_KEY": FAKE_GEMINI_KEY, "GROQ_API_KEY": FAKE_GROQ_KEY},
        client_factory={"gemini": gemini_sdk, "groq": groq_sdk},
    )
    assert registry.provider_ids() == ("gemini", "groq")

    request = _request()
    router = ModelRouter(settings.routing)
    selected = router.select(request, settings.models)
    assert (selected.provider, selected.model) == ("groq", GROQ_MODEL)

    recorder = RecordingSleep()
    executor = FallbackExecutor(
        registry,
        RetryEngine(settings.retry, sleep=recorder.sleep),
        settings.fallback,
    )
    result = run_sync(executor.run(request, settings.models, initial=selected, timeout_s=7.5))

    assert result.succeeded is True
    assert result.request_id == request.request_id
    assert result.response is not None
    assert result.response.request_id == request.request_id
    assert result.response.provider == "gemini"
    assert result.response.model == GEMINI_MODEL
    assert result.response.content == "recovered via gemini"
    assert [(c.provider, c.model) for c in result.candidates_tried] == [
        ("groq", GROQ_MODEL),
        ("gemini", GEMINI_MODEL),
    ]
    assert result.fallback_count == 1
    assert [(a.provider, a.model, a.attempt, a.succeeded) for a in result.attempts] == [
        ("groq", GROQ_MODEL, 1, False),
        ("groq", GROQ_MODEL, 2, False),
        ("gemini", GEMINI_MODEL, 1, True),
    ]
    assert [a.error_category for a in result.attempts[:2]] == [
        ErrorCategory.PROVIDER_UNAVAILABLE
    ] * 2
    # Same candidate retried (one deterministic zero wait), then one transition.
    assert recorder.delays == [0.0]
    # The failing candidate spent its whole budget before fallback engaged.
    assert len(groq_sdk.calls) == 2
    assert len(gemini_sdk.calls) == 1
    # Timeout propagated through the existing per-call seam to both SDKs.
    assert [call["timeout"] for call in groq_sdk.calls] == [7.5, 7.5]
    gemini_config = gemini_sdk.calls[0]["config"]
    assert gemini_config.http_options is not None
    assert gemini_config.http_options.timeout == 7500  # seconds → SDK milliseconds
    # Credentials never leak into the observable result.
    dumped = result.model_dump_json()
    assert FAKE_GEMINI_KEY not in dumped
    assert FAKE_GROQ_KEY not in dumped
    assert "AIza" not in dumped
    assert "gsk_" not in dumped


def test_offline_end_to_end_single_candidate_orchestration() -> None:
    """The same configured registry also serves a single-candidate run."""
    settings = load_settings(explicit=_explicit_settings(), env={})
    assert settings.default_timeout_s is None

    groq_sdk = FakeGroqSdkClient([_groq_success()])
    gemini_sdk = FakeGeminiSdkClient([_gemini_success("direct gemini answer")])
    registry = build_registry(
        settings,
        env={"GEMINI_API_KEY": FAKE_GEMINI_KEY, "GROQ_API_KEY": FAKE_GROQ_KEY},
        client_factory={"gemini": gemini_sdk, "groq": groq_sdk},
    )

    recorder = RecordingSleep()
    orchestrator = Orchestrator(registry, RetryEngine(settings.retry, sleep=recorder.sleep))
    request = _request()
    result = run_sync(orchestrator.run(request, ModelConfig(provider="gemini", model=GEMINI_MODEL)))

    assert result.succeeded is True
    assert result.request_id == request.request_id
    assert result.response is not None
    assert result.response.provider == "gemini"
    assert result.response.content == "direct gemini answer"
    assert recorder.delays == []
    assert len(gemini_sdk.calls) == 1
    assert groq_sdk.calls == []


def test_offline_timeout_precedence_uses_configured_default() -> None:
    """The config-layer timeout rule feeds the existing propagation seams."""
    settings = load_settings(
        explicit=_explicit_settings(),
        env={"LLM_ORCH_DEFAULT_TIMEOUT_S": "9.5"},
    )
    assert resolve_timeout_s(settings) == 9.5
    assert resolve_timeout_s(settings, timeout_s=7.5) == 7.5
    assert (
        resolve_timeout_s(settings, request=_request().model_copy(update={"timeout_s": 3.0})) == 3.0
    )
