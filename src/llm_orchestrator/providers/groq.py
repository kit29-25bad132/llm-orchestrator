"""Groq provider adapter.

This is the only module in the library that is allowed to import the Groq SDK.
It implements the Phase 3 ``LLMProvider`` protocol and translates between the
provider-neutral contracts and Groq, so no SDK type ever escapes past this
boundary.

Design notes:

- The adapter performs exactly **one** attempt. It never retries, never falls
  back, and never substitutes a different model; those are orchestration concerns.
- Configuration enters explicitly via ``api_key`` (and an optional injected
  client). The provider never reads the environment; the future configuration
  layer is responsible for supplying the credential.
- The Groq client is created lazily on first use and reused for the provider's
  lifetime, then released by :meth:`GroqProvider.aclose`.
- The SDK's own automatic retries are disabled (``max_retries=0``) so the
  single-attempt contract holds: a transient failure is classified and surfaced,
  and the caller decides whether to retry.
"""

from __future__ import annotations

import time
from typing import Any

try:
    import groq
    import httpx
    from groq.types.chat import ChatCompletion
    from groq.types.completion_usage import CompletionUsage
except ImportError as exc:
    # The Groq SDK is an optional extra; never fail with an opaque ImportError.
    from ..errors import ConfigurationError

    raise ConfigurationError(
        "GroqProvider requires the optional 'groq' SDK. "
        'Install it with: pip install "llm-orchestrator[groq]"'
    ) from exc

from ..contracts import (
    FinishReason,
    LLMRequest,
    LLMResponse,
    MessageRole,
    ModelConfig,
    Usage,
)
from ..errors import (
    AuthenticationError,
    AuthorizationError,
    ConfigurationError,
    InvalidRequestError,
    LLMError,
    LLMTimeoutError,
    MalformedResponseError,
    ModelUnavailableError,
    NetworkError,
    ProviderUnavailableError,
    RateLimitError,
    UnknownProviderError,
    require_timeout_s,
)

PROVIDER_ID = "groq"

#: Groq finish reasons that map onto the provider-neutral taxonomy. The Groq
#: chat API reports ``stop`` for a natural stop, ``length`` when the token
#: budget was reached, and ``tool_calls``/``function_call`` when the model
#: requested a tool invocation instead of text.
_FINISH_REASON_MAP: dict[str, FinishReason] = {
    "stop": FinishReason.STOP,
    "length": FinishReason.LENGTH,
    "tool_calls": FinishReason.TOOL_CALL,
    "function_call": FinishReason.TOOL_CALL,
}


def _as_token_count(value: object) -> int | None:
    """Coerce a usage value to a non-negative int, or ``None`` if unavailable."""
    if isinstance(value, bool):
        return None
    if isinstance(value, int) and value >= 0:
        return value
    return None


def _translate_messages(request: LLMRequest) -> list[dict[str, str]]:
    """Translate provider-neutral messages into Groq chat messages.

    Groq speaks the OpenAI-compatible ``{"role", "content"}`` shape, so roles
    map one-to-one and ordering is preserved exactly as the caller supplied it.
    """
    translated: list[dict[str, str]] = []
    for message in request.messages:
        if message.role is MessageRole.TOOL:
            raise InvalidRequestError(
                "GroqProvider does not support 'tool' messages yet",
                provider=PROVIDER_ID,
                request_id=request.request_id,
            )
        translated.append({"role": message.role.value, "content": message.content})
    return translated


def _normalize_usage(usage: CompletionUsage | None) -> Usage | None:
    if usage is None:
        return None
    return Usage(
        input_tokens=_as_token_count(getattr(usage, "prompt_tokens", None)),
        output_tokens=_as_token_count(getattr(usage, "completion_tokens", None)),
        total_tokens=_as_token_count(getattr(usage, "total_tokens", None)),
    )


def _normalize_response(
    response: ChatCompletion,
    *,
    request: LLMRequest,
    model: ModelConfig,
    latency_ms: float,
) -> LLMResponse:
    """Translate a Groq chat completion into the normalized ``LLMResponse`` contract."""
    metadata: dict[str, Any] = {}
    response_id = getattr(response, "id", None)
    if response_id:
        metadata["response_id"] = response_id
    fingerprint = getattr(response, "system_fingerprint", None)
    if fingerprint:
        metadata["system_fingerprint"] = fingerprint

    usage = _normalize_usage(getattr(response, "usage", None))

    choices = getattr(response, "choices", None) or []
    if not choices:
        raise MalformedResponseError(
            "Groq returned no choices",
            provider=PROVIDER_ID,
            model=model.model,
            request_id=request.request_id,
        )
    choice = choices[0]
    message = getattr(choice, "message", None)
    content = getattr(message, "content", None) if message is not None else None
    finish_reason = _FINISH_REASON_MAP.get(
        str(getattr(choice, "finish_reason", "") or ""), FinishReason.UNKNOWN
    )
    return LLMResponse(
        request_id=request.request_id,
        provider=PROVIDER_ID,
        model=model.model,
        finish_reason=finish_reason,
        latency_ms=latency_ms,
        content=content,
        usage=usage,
        metadata=metadata,
    )


def _retry_after_s(exc: groq.APIStatusError) -> float | None:
    """Best-effort ``Retry-After`` extraction from the SDK error's response."""
    response = getattr(exc, "response", None)
    headers = getattr(response, "headers", None)
    if headers is None:
        return None
    try:
        raw = headers.get("retry-after")
        return float(raw) if raw is not None else None
    except (TypeError, ValueError):
        return None


def _map_status_error(
    exc: groq.APIStatusError,
    *,
    model: str,
    request_id: str,
) -> LLMError:
    """Map a Groq status error's HTTP status onto the normalized taxonomy."""
    code = getattr(exc, "status_code", None)
    message = getattr(exc, "message", None) or "Groq API error"
    context: dict[str, Any] = {
        "provider": PROVIDER_ID,
        "model": model,
        "request_id": request_id,
        "status_code": code if isinstance(code, int) else None,
        "cause": exc,
    }
    if code == 401:
        return AuthenticationError(message, **context)
    if code == 403:
        return AuthorizationError(message, **context)
    if code == 404:
        return ModelUnavailableError(message, **context)
    if code == 408:
        # A request-timeout status means the operation timed out server-side:
        # retryable, unlike the other 4xx client errors below.
        return LLMTimeoutError(message, **context)
    if code == 429:
        return RateLimitError(message, retry_after_s=_retry_after_s(exc), **context)
    if isinstance(code, int) and 400 <= code < 500:
        return InvalidRequestError(message, **context)
    if isinstance(code, int) and code >= 500:
        return ProviderUnavailableError(message, **context)
    return UnknownProviderError(message, **context)


def _map_exception(exc: Exception, *, model: str, request_id: str) -> LLMError:
    """Map any exception raised by the SDK onto the normalized taxonomy.

    This function is total: it never raises, and an unclassifiable failure
    becomes ``UnknownProviderError`` with the original exception chained.
    """
    # Specific SDK subclasses first; the generic status mapping below would
    # otherwise absorb them (it keys on the same status codes).
    if isinstance(exc, groq.AuthenticationError):
        return _map_status_error(exc, model=model, request_id=request_id)
    if isinstance(exc, groq.PermissionDeniedError):
        return _map_status_error(exc, model=model, request_id=request_id)
    if isinstance(exc, groq.NotFoundError):
        return _map_status_error(exc, model=model, request_id=request_id)
    if isinstance(exc, groq.RateLimitError):
        return _map_status_error(exc, model=model, request_id=request_id)
    if isinstance(
        exc,
        groq.BadRequestError | groq.UnprocessableEntityError | groq.ConflictError,
    ):
        return _map_status_error(exc, model=model, request_id=request_id)
    if isinstance(exc, groq.InternalServerError):
        return _map_status_error(exc, model=model, request_id=request_id)
    if isinstance(exc, groq.APIStatusError):
        return _map_status_error(exc, model=model, request_id=request_id)

    context: dict[str, Any] = {
        "provider": PROVIDER_ID,
        "model": model,
        "request_id": request_id,
        "cause": exc,
    }
    # Check timeouts before the broader transport family (a timeout IS a transport error).
    if isinstance(exc, groq.APITimeoutError | httpx.TimeoutException | TimeoutError):
        return LLMTimeoutError("Groq request timed out", **context)
    if isinstance(exc, groq.APIConnectionError | httpx.TransportError | ConnectionError):
        return NetworkError("Groq network error", **context)
    return UnknownProviderError(f"unmapped Groq failure: {type(exc).__name__}", **context)


class GroqProvider:
    """Concrete :class:`~llm_orchestrator.providers.base.LLMProvider` for Groq."""

    provider_id = PROVIDER_ID

    def __init__(self, api_key: str | None = None, *, client: Any | None = None) -> None:
        """Create the provider.

        ``api_key`` is required unless a pre-built ``client`` is injected. The
        key is held privately and never appears in errors, metadata, or ``repr``.
        ``client`` exists so tests (and advanced callers) can supply an SDK
        client without the provider constructing one.
        """
        if client is None and (api_key is None or not api_key.strip()):
            raise ConfigurationError(
                "GroqProvider requires an explicit api_key "
                "(set GROQ_API_KEY in the environment and pass it in; "
                "the provider never reads the environment itself)"
            )
        self._api_key = api_key
        self._client: Any | None = client

    def __repr__(self) -> str:
        return f"GroqProvider(provider_id={PROVIDER_ID!r}, api_key=[REDACTED])"

    def _groq_client(self) -> Any:
        """Return the SDK client, creating it lazily on first use."""
        if self._client is None:
            # Disable the SDK's own retries: the adapter performs exactly one
            # attempt and lets the caller decide about retry/fallback.
            self._client = groq.AsyncGroq(api_key=self._api_key, max_retries=0)
        return self._client

    async def generate(
        self,
        request: LLMRequest,
        model: ModelConfig,
        *,
        timeout_s: float | None = None,
    ) -> LLMResponse:
        """Perform exactly one Groq generation attempt."""
        require_timeout_s(timeout_s)
        if model.provider != PROVIDER_ID:
            raise InvalidRequestError(
                f"GroqProvider cannot serve provider {model.provider!r}",
                provider=PROVIDER_ID,
                model=model.model,
                request_id=request.request_id,
            )
        if request.response_schema is not None:
            raise InvalidRequestError(
                "structured output is not implemented by GroqProvider yet",
                provider=PROVIDER_ID,
                model=model.model,
                request_id=request.request_id,
            )

        messages = _translate_messages(request)
        call_kwargs: dict[str, Any] = {"model": model.model, "messages": messages}
        if request.temperature is not None:
            call_kwargs["temperature"] = request.temperature
        if request.max_tokens is not None:
            call_kwargs["max_completion_tokens"] = request.max_tokens
        if request.stop is not None:
            call_kwargs["stop"] = list(request.stop)
        effective_timeout = timeout_s if timeout_s is not None else request.timeout_s
        if effective_timeout is not None:
            call_kwargs["timeout"] = effective_timeout

        start = time.perf_counter()
        try:
            response = await self._groq_client().chat.completions.create(**call_kwargs)
        except LLMError:
            raise
        except Exception as exc:
            raise _map_exception(exc, model=model.model, request_id=request.request_id) from exc
        latency_ms = (time.perf_counter() - start) * 1000.0

        return _normalize_response(response, request=request, model=model, latency_ms=latency_ms)

    async def aclose(self) -> None:
        """Close the underlying SDK client, if one was created or injected."""
        client = self._client
        if client is None:
            return
        close = getattr(client, "close", None)
        if callable(close):
            result = close()
            if hasattr(result, "__await__"):
                await result
        self._client = None
