"""Tests for the orchestration outcome contracts."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from llm_orchestrator.contracts import (
    AttemptRecord,
    CandidateRef,
    ErrorInfo,
    FinishReason,
    LLMResponse,
    OrchestrationResult,
)
from llm_orchestrator.errors import AuthenticationError, ErrorCategory, RateLimitError


def _response() -> LLMResponse:
    return LLMResponse(
        request_id="req-1",
        provider="p",
        model="m",
        finish_reason=FinishReason.STOP,
        latency_ms=5.0,
        content="ok",
    )


def _error_info() -> ErrorInfo:
    return ErrorInfo(
        category=ErrorCategory.RATE_LIMIT,
        message="throttled",
        retryable=True,
        fallback_eligible=True,
    )


def test_successful_attempt_record() -> None:
    record = AttemptRecord(provider="p", model="m", attempt=1, succeeded=True, latency_ms=3.0)
    assert record.succeeded is True
    assert record.error_category is None


def test_failed_attempt_record() -> None:
    record = AttemptRecord(
        provider="p",
        model="m",
        attempt=2,
        succeeded=False,
        error_category=ErrorCategory.TIMEOUT,
    )
    assert record.latency_ms is None


def test_attempt_requires_positive_index() -> None:
    with pytest.raises(ValidationError):
        AttemptRecord(provider="p", model="m", attempt=0, succeeded=True)


def test_attempt_outcome_must_be_consistent() -> None:
    with pytest.raises(ValidationError):
        AttemptRecord(
            provider="p", model="m", attempt=1, succeeded=True, error_category=ErrorCategory.TIMEOUT
        )
    with pytest.raises(ValidationError):
        AttemptRecord(provider="p", model="m", attempt=1, succeeded=False)


def test_error_info_from_error_uses_sanitized_message() -> None:
    secret = "AIzaSyABCDEFGHIJKLMNOPQRSTUVWXYZ012345"
    error = AuthenticationError(f"bad key {secret}", provider="p", model="m", attempt=1)
    info = ErrorInfo.from_error(error)
    assert secret not in info.message
    assert info.category is ErrorCategory.AUTHENTICATION
    assert info.retryable is False
    assert info.fallback_eligible is True
    assert info.provider == "p"
    assert info.model == "m"
    assert info.attempt == 1


def test_error_info_preserves_retry_semantics() -> None:
    info = ErrorInfo.from_error(RateLimitError("slow down", status_code=429))
    assert info.retryable is True
    assert info.status_code == 429


def test_successful_orchestration_result() -> None:
    result = OrchestrationResult(
        request_id="req-1",
        succeeded=True,
        response=_response(),
        candidates_tried=(CandidateRef(provider="p", model="m"),),
        attempts=(AttemptRecord(provider="p", model="m", attempt=1, succeeded=True),),
        total_latency_ms=5.0,
    )
    assert result.succeeded is True
    assert result.fallback_count == 0
    assert result.retry_count == 0


def test_failed_orchestration_result() -> None:
    result = OrchestrationResult(request_id="req-1", succeeded=False, error=_error_info())
    assert result.response is None
    assert result.error is not None


def test_result_requires_response_on_success() -> None:
    with pytest.raises(ValidationError):
        OrchestrationResult(request_id="req-1", succeeded=True)


def test_result_requires_error_on_failure() -> None:
    with pytest.raises(ValidationError):
        OrchestrationResult(request_id="req-1", succeeded=False)


def test_result_must_not_mix_success_and_error() -> None:
    with pytest.raises(ValidationError):
        OrchestrationResult(
            request_id="req-1", succeeded=True, response=_response(), error=_error_info()
        )


def test_result_derived_fallback_and_retry_counts() -> None:
    result = OrchestrationResult(
        request_id="req-1",
        succeeded=True,
        response=_response(),
        candidates_tried=(
            CandidateRef(provider="p1", model="m1"),
            CandidateRef(provider="p2", model="m2"),
        ),
        attempts=(
            AttemptRecord(
                provider="p1",
                model="m1",
                attempt=1,
                succeeded=False,
                error_category=ErrorCategory.TIMEOUT,
            ),
            AttemptRecord(
                provider="p1",
                model="m1",
                attempt=2,
                succeeded=False,
                error_category=ErrorCategory.TIMEOUT,
            ),
            AttemptRecord(provider="p2", model="m2", attempt=1, succeeded=True),
        ),
    )
    assert result.fallback_count == 1
    assert result.retry_count == 1


def test_blank_result_request_id_rejected() -> None:
    with pytest.raises(ValidationError):
        OrchestrationResult(request_id="  ", succeeded=False, error=_error_info())
