"""Gemini and Groq must classify equivalent failures identically.

These tests drive both real adapters' boundary mappers with equivalent SDK
failures and assert the normalized semantics converge: same category, same
retryability, same fallback eligibility. Upper layers must never need
``if provider == ...`` or SDK-specific branches.
"""

from __future__ import annotations

from collections.abc import Callable

import groq
import httpx
import pytest
from google.genai import errors as genai_errors

from llm_orchestrator.contracts import ErrorInfo
from llm_orchestrator.errors import (
    ErrorCategory,
    LLMError,
    RateLimitError,
)
from llm_orchestrator.providers import gemini as gemini_adapter
from llm_orchestrator.providers import groq as groq_adapter

MODEL = "some-model"
REQUEST_ID = "req-1"


def _gemini_api_error(code: int) -> genai_errors.APIError:
    cls = genai_errors.ServerError if code >= 500 else genai_errors.ClientError
    return cls(code, {"error": {"code": code, "message": "boom", "status": "ERR"}})


def _groq_api_error(code: int) -> groq.APIStatusError:
    cls: type[groq.APIStatusError] = {
        400: groq.BadRequestError,
        401: groq.AuthenticationError,
        403: groq.PermissionDeniedError,
        404: groq.NotFoundError,
        408: groq.BadRequestError,
        422: groq.UnprocessableEntityError,
        429: groq.RateLimitError,
        500: groq.InternalServerError,
        503: groq.InternalServerError,
    }[code]
    request = httpx.Request("POST", "https://api.groq.com/openai/v1/chat/completions")
    response = httpx.Response(code, request=request)
    return cls("boom", response=response, body={"message": "boom"})


def _classify(factory: Callable[[], Exception], *, provider: str) -> LLMError:
    if provider == "gemini":
        mapper = gemini_adapter._map_exception
    else:
        mapper = groq_adapter._map_exception
    failure = factory()
    # The mappers return the normalized error (the adapters add ``raise ...
    # from`` chaining at the generate boundary, covered by per-provider tests).
    normalized = mapper(failure, model=MODEL, request_id=REQUEST_ID)
    assert normalized.cause is failure
    return normalized


# (status, expected category, retryable, fallback_eligible)
STATUS_MATRIX = [
    (400, ErrorCategory.INVALID_REQUEST, False, False),
    (401, ErrorCategory.AUTHENTICATION, False, True),
    (403, ErrorCategory.AUTHORIZATION, False, True),
    (404, ErrorCategory.MODEL_UNAVAILABLE, False, True),
    (408, ErrorCategory.TIMEOUT, True, True),
    (422, ErrorCategory.INVALID_REQUEST, False, False),
    (429, ErrorCategory.RATE_LIMIT, True, True),
    (500, ErrorCategory.PROVIDER_UNAVAILABLE, True, True),
    (503, ErrorCategory.PROVIDER_UNAVAILABLE, True, True),
]


@pytest.mark.parametrize(("status", "category", "retryable", "fallback"), STATUS_MATRIX)
def test_equivalent_statuses_converge(
    status: int, category: ErrorCategory, retryable: bool, fallback: bool
) -> None:
    gemini_error = _classify(lambda: _gemini_api_error(status), provider="gemini")
    groq_error = _classify(lambda: _groq_api_error(status), provider="groq")
    for error in (gemini_error, groq_error):
        assert error.category is category
        assert error.retryable is retryable
        assert error.fallback_eligible is fallback
        assert error.status_code == status
        assert error.model == MODEL
        assert error.request_id == REQUEST_ID
    assert gemini_error.provider == "gemini"
    assert groq_error.provider == "groq"
    # Same failure, same semantics: the provider dimension must not leak through.
    assert (gemini_error.category, gemini_error.retryable, gemini_error.fallback_eligible) == (
        groq_error.category,
        groq_error.retryable,
        groq_error.fallback_eligible,
    )


@pytest.mark.parametrize("provider", ["gemini", "groq"])
def test_transport_failures_converge(provider: str) -> None:
    timeout: dict[str, Callable[[], Exception]] = {
        "gemini": lambda: httpx.ReadTimeout("timed out"),
        "groq": lambda: httpx.ReadTimeout("timed out"),
    }
    network: dict[str, Callable[[], Exception]] = {
        "gemini": lambda: httpx.ConnectError("refused"),
        "groq": lambda: httpx.ConnectError("refused"),
    }
    timed_out = _classify(timeout[provider], provider=provider)
    assert timed_out.category is ErrorCategory.TIMEOUT
    assert timed_out.retryable is True
    assert timed_out.fallback_eligible is True

    unreachable = _classify(network[provider], provider=provider)
    assert unreachable.category is ErrorCategory.NETWORK
    assert unreachable.retryable is True
    assert unreachable.fallback_eligible is True


@pytest.mark.parametrize("provider", ["gemini", "groq"])
def test_unmapped_failures_stay_conservative(provider: str) -> None:
    error = _classify(lambda: RuntimeError("something strange"), provider=provider)
    assert error.category is ErrorCategory.UNKNOWN
    assert error.retryable is False
    assert error.fallback_eligible is True


@pytest.mark.parametrize("provider", ["gemini", "groq"])
def test_retry_after_survives_normalization(provider: str) -> None:
    request = httpx.Request("POST", "https://x")
    if provider == "gemini":
        response = httpx.Response(429, headers={"retry-after": "12"}, request=request)
        failure: Exception = genai_errors.ClientError(
            429, {"error": {"code": 429, "message": "quota"}}, response
        )
    else:
        response = httpx.Response(429, headers={"retry-after": "12"}, request=request)
        failure = groq.RateLimitError("quota", response=response, body=None)
    error = _classify(lambda: failure, provider=provider)
    assert isinstance(error, RateLimitError)
    assert error.retry_after_s == 12.0


@pytest.mark.parametrize("provider", ["gemini", "groq"])
def test_secrets_never_reach_public_error_surfaces(provider: str) -> None:
    secret = "gsk_" + "s3cr3t" * 5
    if provider == "gemini":
        failure: Exception = genai_errors.ClientError(
            401, {"error": {"code": 401, "message": f"bad key {secret}", "status": "ERR"}}
        )
    else:
        request = httpx.Request("POST", "https://x")
        response = httpx.Response(401, request=request)
        failure = groq.AuthenticationError(
            f"bad key {secret}", response=response, body={"message": secret}
        )
    error = _classify(lambda: failure, provider=provider)
    assert secret not in str(error)
    assert secret not in error.safe_message
    assert secret not in repr(error.args)
    snapshot = ErrorInfo.from_error(error)
    assert secret not in snapshot.model_dump_json()
    assert snapshot.category is ErrorCategory.AUTHENTICATION


def test_error_info_preserves_classification_semantics() -> None:
    gemini_error = _classify(lambda: _gemini_api_error(429), provider="gemini")
    groq_error = _classify(lambda: _groq_api_error(429), provider="groq")
    for error in (gemini_error, groq_error):
        snapshot = ErrorInfo.from_error(error)
        assert snapshot.category is ErrorCategory.RATE_LIMIT
        assert snapshot.retryable is True
        assert snapshot.fallback_eligible is True
        assert snapshot.status_code == 429
        assert snapshot.model == MODEL


def test_decisions_are_category_driven_not_string_driven() -> None:
    first = RateLimitError("quota exceeded for tenant A")
    second = RateLimitError("completely different wording here")
    assert first.category is second.category is ErrorCategory.RATE_LIMIT
    assert (first.retryable, first.fallback_eligible) == (
        second.retryable,
        second.fallback_eligible,
    )
