"""Degenerate-input and serialization round-trip gaps.

The per-contract suites already cover blank identifiers, empty messages, and
core round-trips. These tests close only the verified gaps: empty routing and
fallback inputs at the execution boundary, combined budget floors, optional
telemetry absence, standalone descriptor/policy round-trips, and a
structured-output schema round-trip. Only guarantees the public API actually
makes are asserted.
"""

from __future__ import annotations

import pytest
from pydantic import BaseModel, ValidationError

from llm_orchestrator.config import OrchestratorSettings
from llm_orchestrator.contracts import (
    AttemptRecord,
    ErrorInfo,
    FallbackPolicy,
    FinishReason,
    LLMResponse,
    ModelCapabilities,
    ModelConfig,
    ModelPricing,
    RetryPolicy,
    RoutingContext,
    Usage,
)
from llm_orchestrator.errors import (
    ConfigurationError,
    ErrorCategory,
    NetworkError,
    UnknownProviderError,
)
from llm_orchestrator.reliability import FallbackExecutor, RetryEngine
from llm_orchestrator.routing import ModelRouter
from llm_orchestrator.routing.static import StaticRoutingPolicy
from support.fakes import (
    RecordingSleep,
    ScriptedProvider,
    make_model,
    make_registry,
    make_request,
    make_response,
    run_sync,
)


def test_router_select_with_no_candidates_raises_configuration_error() -> None:
    """An empty candidate set is rejected exactly; nothing is substituted."""
    with pytest.raises(ConfigurationError):
        ModelRouter().select(make_request(), [])


def test_empty_settings_models_fail_selection_at_use_time() -> None:
    """An empty configuration stays valid until selection is attempted."""
    settings = OrchestratorSettings()
    assert settings.models == ()
    with pytest.raises(ConfigurationError):
        ModelRouter(settings.routing).select(make_request(), settings.models)


def test_policy_ordering_over_empty_candidates_is_empty() -> None:
    """The static policy returns no ordering (the router turns that into an error)."""
    assert StaticRoutingPolicy().ordered(make_request(), [], RoutingContext()) == ()


def test_empty_registry_yields_terminal_configuration_result() -> None:
    """Resolution misses produce a failed result, not an exception or retry."""
    executor = FallbackExecutor(make_registry(), policy=FallbackPolicy(max_fallbacks=5))
    result = run_sync(executor.run(make_request(), [make_model("ghost")]))
    assert result.succeeded is False
    assert result.error is not None
    assert result.error.category is ErrorCategory.CONFIGURATION
    assert result.attempts == ()
    assert len(result.candidates_tried) == 1


def test_combined_budget_floor_runs_exactly_one_attempt() -> None:
    """``max_attempts=1`` plus ``max_fallbacks=0`` means one call, no waits."""
    recorder = RecordingSleep()
    provider = ScriptedProvider("pa", [NetworkError("down")])
    engine = RetryEngine(RetryPolicy(max_attempts=1), sleep=recorder.sleep)
    executor = FallbackExecutor(make_registry(provider), engine, FallbackPolicy(max_fallbacks=0))
    result = run_sync(executor.run(make_request(), [make_model("pa"), make_model("pb")]))

    assert result.succeeded is False
    assert len(provider.calls) == 1
    assert recorder.delays == []
    assert len(result.attempts) == 1
    assert result.attempts[0].error_category is ErrorCategory.NETWORK
    assert result.retry_count == 0
    assert result.fallback_count == 0


def test_response_without_usage_or_metadata_round_trips() -> None:
    """Missing optional telemetry stays missing through serialization."""
    response = make_response("p")
    assert response.usage is None
    assert dict(response.metadata) == {}
    restored = LLMResponse.model_validate_json(response.model_dump_json())
    assert restored == response
    assert restored.usage is None


def test_empty_usage_round_trips_with_all_fields_unset() -> None:
    """Providers that report no token counts keep ``None``, never zeros."""
    restored = Usage.model_validate_json(Usage().model_dump_json())
    assert restored == Usage(input_tokens=None, output_tokens=None, total_tokens=None)


@pytest.mark.parametrize(
    "sample",
    [
        ModelCapabilities(),
        ModelCapabilities(structured_output=True, max_context_tokens=4096),
        ModelPricing(),
        ModelPricing(input_per_1k_tokens=0.5, output_per_1k_tokens=1.5, currency="USD"),
        RoutingContext(),
        RoutingContext(metadata={"tenant": "demo"}),
        RetryPolicy(),
        RetryPolicy(max_attempts=3),
        FallbackPolicy(),
        StaticRoutingPolicy(),
        StaticRoutingPolicy(
            preferred_provider="gemini",
            allowed_providers=frozenset({"gemini", "groq"}),
            min_context_tokens=4096,
        ),
        AttemptRecord(provider="p", model="m", attempt=1, succeeded=True),
        AttemptRecord(
            provider="p",
            model="m",
            attempt=2,
            succeeded=False,
            error_category=ErrorCategory.NETWORK,
        ),
        ErrorInfo.from_error(UnknownProviderError("mystery")),
        make_response("p"),
    ],
)
def test_standalone_contracts_round_trip(sample: BaseModel) -> None:
    """Descriptor, policy, history, and snapshot objects survive JSON round-trips."""
    assert type(sample).model_validate_json(sample.model_dump_json()) == sample


def test_unknown_error_snapshot_preserves_conservative_semantics() -> None:
    """Unmapped failures stay non-retryable but fallback-eligible after a round-trip."""
    restored = ErrorInfo.model_validate_json(
        ErrorInfo.from_error(UnknownProviderError("mystery")).model_dump_json()
    )
    assert restored.category is ErrorCategory.UNKNOWN
    assert restored.retryable is False
    assert restored.fallback_eligible is True


def test_structured_schema_result_round_trips() -> None:
    """A caller-owned schema instance serializes without loss."""

    class ShipmentLabel(BaseModel):
        recipient: str
        zip_code: str

    label = ShipmentLabel(recipient="Ada", zip_code="10115")
    assert ShipmentLabel.model_validate_json(label.model_dump_json()) == label


def test_success_response_with_content_filter_reason_round_trips() -> None:
    """Blocked-prompt responses (no content) keep their reason through JSON."""
    response = LLMResponse(
        request_id="req-1",
        provider="p",
        model="m",
        finish_reason=FinishReason.CONTENT_FILTER,
        latency_ms=1.0,
        content=None,
        metadata={"block_reason": "SAFETY"},
    )
    restored = LLMResponse.model_validate_json(response.model_dump_json())
    assert restored == response
    assert restored.content is None
    assert restored.finish_reason is FinishReason.CONTENT_FILTER


def test_empty_model_list_never_reaches_a_provider() -> None:
    """Empty candidate input raises before any provider is resolved or called."""
    provider = ScriptedProvider("pa", [make_response("pa")])
    executor = FallbackExecutor(make_registry(provider), policy=FallbackPolicy(max_fallbacks=5))
    with pytest.raises(ConfigurationError):
        run_sync(executor.run(make_request(), []))
    assert provider.calls == []


def test_blank_model_config_identifiers_rejected() -> None:
    """Degenerate candidate construction fails at the contract boundary."""
    with pytest.raises(ValidationError):
        ModelConfig(provider="  ", model="m")
