"""Tests for the normalized error taxonomy."""

from __future__ import annotations

import pytest

from llm_orchestrator.errors import (
    AuthenticationError,
    AuthorizationError,
    ConfigurationError,
    ErrorCategory,
    InvalidRequestError,
    LLMError,
    LLMTimeoutError,
    MalformedResponseError,
    ModelUnavailableError,
    NetworkError,
    ProviderUnavailableError,
    RateLimitError,
    StructuredOutputValidationError,
    UnknownProviderError,
    sanitize_message,
)

# (category, retryable_by_default, fallback_eligible_by_default)
EXPECTED_SEMANTICS = [
    (ErrorCategory.AUTHENTICATION, False, True),
    (ErrorCategory.AUTHORIZATION, False, True),
    (ErrorCategory.RATE_LIMIT, True, True),
    (ErrorCategory.TIMEOUT, True, True),
    (ErrorCategory.NETWORK, True, True),
    (ErrorCategory.PROVIDER_UNAVAILABLE, True, True),
    (ErrorCategory.MODEL_UNAVAILABLE, False, True),
    (ErrorCategory.INVALID_REQUEST, False, False),
    (ErrorCategory.MALFORMED_RESPONSE, True, True),
    (ErrorCategory.STRUCTURED_OUTPUT_VALIDATION, True, True),
    (ErrorCategory.CONFIGURATION, False, False),
    (ErrorCategory.UNKNOWN, False, True),
]

CATEGORY_TO_CLASS = {
    ErrorCategory.AUTHENTICATION: AuthenticationError,
    ErrorCategory.AUTHORIZATION: AuthorizationError,
    ErrorCategory.RATE_LIMIT: RateLimitError,
    ErrorCategory.TIMEOUT: LLMTimeoutError,
    ErrorCategory.NETWORK: NetworkError,
    ErrorCategory.PROVIDER_UNAVAILABLE: ProviderUnavailableError,
    ErrorCategory.MODEL_UNAVAILABLE: ModelUnavailableError,
    ErrorCategory.INVALID_REQUEST: InvalidRequestError,
    ErrorCategory.MALFORMED_RESPONSE: MalformedResponseError,
    ErrorCategory.STRUCTURED_OUTPUT_VALIDATION: StructuredOutputValidationError,
    ErrorCategory.CONFIGURATION: ConfigurationError,
    ErrorCategory.UNKNOWN: UnknownProviderError,
}


@pytest.mark.parametrize(("category", "retryable", "fallback"), EXPECTED_SEMANTICS)
def test_category_default_flags(category: ErrorCategory, retryable: bool, fallback: bool) -> None:
    assert category.retryable_by_default is retryable
    assert category.fallback_eligible_by_default is fallback


@pytest.mark.parametrize(("category", "error_cls"), list(CATEGORY_TO_CLASS.items()))
def test_every_category_has_a_class_and_is_an_llm_error(
    category: ErrorCategory, error_cls: type[LLMError]
) -> None:
    error = error_cls("boom")
    assert isinstance(error, LLMError)
    assert error.category is category


def test_not_every_error_is_retryable() -> None:
    assert ConfigurationError("bad config").retryable is False
    assert InvalidRequestError("bad request").retryable is False
    assert AuthenticationError("bad key").retryable is False
    assert UnknownProviderError("???").retryable is False


def test_transient_errors_are_retryable() -> None:
    assert RateLimitError("slow down").retryable is True
    assert LLMTimeoutError("too slow").retryable is True
    assert NetworkError("reset").retryable is True


def test_flags_can_be_overridden_per_instance() -> None:
    error = InvalidRequestError("context too long", fallback_eligible=True)
    assert error.fallback_eligible is True
    assert error.retryable is False

    forced_retry = ConfigurationError("x", retryable=True)
    assert forced_retry.retryable is True


def test_context_fields_are_preserved() -> None:
    cause = ValueError("root cause")
    error = RateLimitError(
        "throttled",
        provider="gemini",
        model="some-model",
        request_id="req-1",
        attempt=2,
        status_code=429,
        cause=cause,
        metadata={"retry_after": 5},
    )
    assert error.provider == "gemini"
    assert error.model == "some-model"
    assert error.request_id == "req-1"
    assert error.attempt == 2
    assert error.status_code == 429
    assert error.cause is cause
    assert error.metadata == {"retry_after": 5}


def test_rate_limit_exposes_retry_after() -> None:
    error = RateLimitError("throttled", retry_after_s=12.5)
    assert error.retry_after_s == 12.5


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        (None, None),
        (0, 0.0),
        (5, 5.0),
        (2.5, 2.5),
        (-1.0, None),
        (True, None),
        ("5", None),
        (float("nan"), None),
        (float("inf"), None),
        (float("-inf"), None),
    ],
)
def test_retry_after_is_normalized_to_safe_values(raw: object, expected: float | None) -> None:
    error = RateLimitError("throttled", retry_after_s=raw)  # type: ignore[arg-type]
    if expected is None:
        assert error.retry_after_s is None
    else:
        assert error.retry_after_s == expected
        assert isinstance(error.retry_after_s, float)


def test_retry_after_defaults_to_none() -> None:
    assert RateLimitError("throttled").retry_after_s is None


def test_secrets_are_redacted_from_message_and_str() -> None:
    secret = "AIzaSyABCDEFGHIJKLMNOPQRSTUVWXYZ012345"
    error = AuthenticationError(f"invalid key: {secret}")
    assert secret not in error.safe_message
    assert secret not in str(error)
    assert "[REDACTED]" in error.safe_message


@pytest.mark.parametrize(
    "raw",
    [
        "Authorization: Bearer abc123def456",
        "api_key=super-secret-value",
        "token: gsk_abcdefghijklmnop",
        "password=hunter2",
    ],
)
def test_sanitize_message_removes_credentials(raw: str) -> None:
    sanitized = sanitize_message(raw)
    assert "[REDACTED]" in sanitized
    assert sanitized != raw


def test_sanitize_leaves_ordinary_text_intact() -> None:
    text = "provider returned HTTP 503 while generating a summary"
    assert sanitize_message(text) == text
