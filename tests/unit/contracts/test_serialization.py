"""Serialization tests: dict and JSON round-trips must be predictable."""

from __future__ import annotations

import json

from llm_orchestrator.contracts import (
    AttemptRecord,
    CandidateRef,
    ErrorInfo,
    FallbackPolicy,
    FinishReason,
    LLMRequest,
    LLMResponse,
    Message,
    MessageRole,
    ModelCapabilities,
    ModelConfig,
    OrchestrationResult,
    ProviderConfig,
    RetryPolicy,
    Usage,
)
from llm_orchestrator.errors import ErrorCategory, RateLimitError

SAMPLE_REQUEST = LLMRequest(
    messages=[Message(role=MessageRole.USER, content="hello")],
    temperature=0.2,
    max_tokens=128,
    metadata={"tenant": "demo"},
)

SAMPLES = [
    SAMPLE_REQUEST,
    Message(role=MessageRole.SYSTEM, content="be terse"),
    ModelConfig(
        provider="some-provider",
        model="some-model",
        capabilities=ModelCapabilities(structured_output=True, max_context_tokens=4096),
    ),
    ProviderConfig(name="some-provider", credential_env_var="SOME_API_KEY"),
    RetryPolicy(max_attempts=3),
    FallbackPolicy(max_fallbacks=1, candidates=(CandidateRef(provider="p", model="m"),)),
    LLMResponse(
        request_id="req-1",
        provider="p",
        model="m",
        finish_reason=FinishReason.LENGTH,
        latency_ms=1.5,
        usage=Usage(input_tokens=1, output_tokens=2, total_tokens=3),
    ),
    ErrorInfo.from_error(RateLimitError("slow down", status_code=429)),
    OrchestrationResult(
        request_id="req-1",
        succeeded=True,
        response=LLMResponse(
            request_id="req-1",
            provider="p",
            model="m",
            finish_reason=FinishReason.STOP,
            latency_ms=2.0,
            content="ok",
        ),
        candidates_tried=(CandidateRef(provider="p", model="m"),),
        attempts=(AttemptRecord(provider="p", model="m", attempt=1, succeeded=True),),
    ),
]


def test_python_and_json_round_trips_are_stable() -> None:
    for sample in SAMPLES:
        dumped = sample.model_dump()
        assert isinstance(dumped, dict)
        assert type(sample).model_validate(dumped) == sample

        json_text = sample.model_dump_json()
        assert json.loads(json_text) is not None
        assert type(sample).model_validate_json(json_text) == sample


def test_json_dump_serializes_enums_as_values() -> None:
    payload = json.loads(SAMPLE_REQUEST.model_dump_json())
    assert payload["messages"][0]["role"] == "user"


def test_error_info_json_round_trip_preserves_semantics() -> None:
    info = ErrorInfo.from_error(RateLimitError("throttled", status_code=429))
    restored = ErrorInfo.model_validate_json(info.model_dump_json())
    assert restored == info
    assert restored.category is ErrorCategory.RATE_LIMIT
    assert restored.retryable is True


def test_result_json_round_trip_keeps_derived_counts() -> None:
    result = OrchestrationResult(
        request_id="req-1",
        succeeded=False,
        error=ErrorInfo.from_error(RateLimitError("throttled")),
    )
    restored = OrchestrationResult.model_validate_json(result.model_dump_json())
    assert restored.succeeded is False
    assert restored.error is not None
