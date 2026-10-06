"""Tests for the response and usage contracts."""

from __future__ import annotations

import math

import pytest
from pydantic import ValidationError

from llm_orchestrator.contracts import FinishReason, LLMResponse, Usage


def _response(**overrides: object) -> LLMResponse:
    payload: dict[str, object] = {
        "request_id": "req-1",
        "provider": "some-provider",
        "model": "some-model",
        "finish_reason": FinishReason.STOP,
        "latency_ms": 12.5,
        "content": "hi",
    }
    payload.update(overrides)
    return LLMResponse(**payload)


def test_valid_response() -> None:
    response = _response()
    assert response.request_id == "req-1"
    assert response.provider == "some-provider"
    assert response.model == "some-model"
    assert response.finish_reason is FinishReason.STOP
    assert response.latency_ms == 12.5
    assert response.content == "hi"
    assert response.structured is None
    assert response.usage is None


def test_finish_reason_accepts_string_values() -> None:
    assert _response(finish_reason="length").finish_reason is FinishReason.LENGTH


def test_all_finish_reasons_exist() -> None:
    expected = {"stop", "length", "tool_call", "content_filter", "error", "unknown"}
    assert {reason.value for reason in FinishReason} == expected


@pytest.mark.parametrize("field", ["request_id", "provider", "model"])
def test_blank_identifiers_rejected(field: str) -> None:
    with pytest.raises(ValidationError):
        _response(**{field: "  "})


def test_negative_latency_rejected() -> None:
    with pytest.raises(ValidationError):
        _response(latency_ms=-1.0)


def test_non_finite_latency_rejected() -> None:
    with pytest.raises(ValidationError):
        _response(latency_ms=math.nan)


def test_structured_output_is_a_mapping() -> None:
    response = _response(structured={"summary": "ok"}, content=None)
    assert dict(response.structured or {}) == {"summary": "ok"}


def test_usage_allows_unknown_metrics() -> None:
    usage = Usage()
    assert usage.input_tokens is None
    assert usage.output_tokens is None
    assert usage.total_tokens is None


def test_usage_accepts_partial_metrics() -> None:
    usage = Usage(input_tokens=10, total_tokens=15)
    assert usage.output_tokens is None


def test_usage_rejects_negative_counts() -> None:
    with pytest.raises(ValidationError):
        Usage(input_tokens=-1)


def test_response_unknown_field_rejected() -> None:
    with pytest.raises(ValidationError):
        _response(groq_x_groq={"id": "abc"})


def test_response_is_immutable() -> None:
    response = _response()
    with pytest.raises(ValidationError):
        response.content = "changed"
