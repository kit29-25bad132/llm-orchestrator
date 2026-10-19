"""Gemini provider adapter.

This is the only module in the library that is allowed to import the Google
GenAI SDK. It implements the Phase 3 ``LLMProvider`` protocol and translates
between the provider-neutral contracts and Gemini, so no SDK type ever escapes
past this boundary.

Design notes:

- The adapter performs exactly **one** attempt. It never retries, never falls
  back, and never substitutes a different model; those are orchestration concerns.
- Configuration enters explicitly via ``api_key`` (and an optional injected
  client). The provider never reads the environment; the future configuration
  layer is responsible for supplying the credential.
- The Gemini client is created lazily on first use and reused for the provider's
  lifetime, then released by :meth:`GeminiProvider.aclose`.
"""

from __future__ import annotations

import time
from typing import Any

try:
    import httpx
    from google import genai
    from google.genai import errors as genai_errors
    from google.genai import types as genai_types
except ImportError as exc:
    # The Gemini SDK is an optional extra; never fail with an opaque ImportError.
    from ..errors import ConfigurationError

    raise ConfigurationError(
        "GeminiProvider requires the optional 'google-genai' SDK. "
        'Install it with: pip install "llm-orchestrator[gemini]"'
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

PROVIDER_ID = "gemini"

#: Provider-neutral role for each Gemini content role.
_CONTENT_ROLE = {MessageRole.USER: "user", MessageRole.ASSISTANT: "model"}

#: Gemini finish reasons that mean the output was withheld by a safety filter.
_FILTERED_FINISH_REASONS = frozenset(
    {
        "SAFETY",
        "RECITATION",
        "BLOCKLIST",
        "PROHIBITED_CONTENT",
        "SPII",
        "IMAGE_SAFETY",
        "IMAGE_PROHIBITED_CONTENT",
        "IMAGE_RECITATION",
        "NO_IMAGE",
    }
)

_FINISH_REASON_MAP: dict[str, FinishReason] = {
    "STOP": FinishReason.STOP,
    "MAX_TOKENS": FinishReason.LENGTH,
    **{name: FinishReason.CONTENT_FILTER for name in _FILTERED_FINISH_REASONS},
    "MALFORMED_FUNCTION_CALL": FinishReason.ERROR,
    "UNEXPECTED_TOOL_CALL": FinishReason.ERROR,
    "TOO_MANY_TOOL_CALLS": FinishReason.ERROR,
}

_UNSPECIFIED_BLOCK_REASONS = frozenset({"BLOCK_REASON_UNSPECIFIED", "UNSPECIFIED"})


def _enum_name(value: object) -> str | None:
    """Return the member name of an SDK enum (or the string itself)."""
    if value is None:
        return None
    name = getattr(value, "name", None)
    if isinstance(name, str):
        return name
    if isinstance(value, str):
        return value
    return str(value)


def _as_token_count(value: object) -> int | None:
    """Coerce a usage value to a non-negative int, or ``None`` if unavailable."""
    if isinstance(value, bool):
        return None
    if isinstance(value, int) and value >= 0:
        return value
    return None


def _split_messages(request: LLMRequest) -> tuple[str | None, list[genai_types.Content]]:
    """Translate provider-neutral messages into Gemini system instruction + contents.

    System turns become the ``system_instruction`` (joined), and consecutive
    same-role turns are merged into a single ``Content`` so Gemini's expected
    user/model alternation is preserved.
    """
    system_parts: list[str] = []
    grouped: list[tuple[str, list[str]]] = []
    for message in request.messages:
        if message.role is MessageRole.SYSTEM:
            system_parts.append(message.content)
            continue
        api_role = _CONTENT_ROLE.get(message.role)
        if api_role is None:
            raise InvalidRequestError(
                f"GeminiProvider does not support {message.role.value!r} messages yet",
                provider=PROVIDER_ID,
                request_id=request.request_id,
            )
        if grouped and grouped[-1][0] == api_role:
            grouped[-1][1].append(message.content)
        else:
            grouped.append((api_role, [message.content]))

    system_instruction = "\n\n".join(system_parts) if system_parts else None
    contents = [
        genai_types.Content(role=role, parts=[genai_types.Part(text=text) for text in texts])
        for role, texts in grouped
    ]
    return system_instruction, contents


def _build_config(
    request: LLMRequest,
    system_instruction: str | None,
    timeout_s: float | None,
) -> genai_types.GenerateContentConfig:
    """Map the supported generic generation parameters onto the SDK config."""
    kwargs: dict[str, Any] = {}
    if system_instruction is not None:
        kwargs["system_instruction"] = system_instruction
    if request.temperature is not None:
        kwargs["temperature"] = request.temperature
    if request.max_tokens is not None:
        kwargs["max_output_tokens"] = request.max_tokens
    if request.stop is not None:
        kwargs["stop_sequences"] = list(request.stop)
    if timeout_s is not None:
        # The SDK expresses HTTP timeouts in milliseconds.
        kwargs["http_options"] = genai_types.HttpOptions(timeout=int(timeout_s * 1000))
    return genai_types.GenerateContentConfig(**kwargs)


def _extract_text(candidate: genai_types.Candidate) -> str | None:
    """Concatenate text parts of a candidate, or ``None`` when there is no text."""
    content = getattr(candidate, "content", None)
    if content is None:
        return None
    parts = getattr(content, "parts", None) or []
    chunks = [part.text for part in parts if getattr(part, "text", None)]
    if not chunks:
        return None
    return "".join(chunks)


def _block_reason(response: genai_types.GenerateContentResponse) -> str | None:
    feedback = getattr(response, "prompt_feedback", None)
    reason = _enum_name(getattr(feedback, "block_reason", None))
    if reason is None or reason in _UNSPECIFIED_BLOCK_REASONS:
        return None
    return reason


def _normalize_usage(
    usage_metadata: genai_types.GenerateContentResponseUsageMetadata | None,
) -> Usage | None:
    if usage_metadata is None:
        return None
    return Usage(
        input_tokens=_as_token_count(getattr(usage_metadata, "prompt_token_count", None)),
        output_tokens=_as_token_count(getattr(usage_metadata, "candidates_token_count", None)),
        total_tokens=_as_token_count(getattr(usage_metadata, "total_token_count", None)),
    )


def _normalize_response(
    response: genai_types.GenerateContentResponse,
    *,
    request: LLMRequest,
    model: ModelConfig,
    latency_ms: float,
) -> LLMResponse:
    """Translate a Gemini response into the normalized ``LLMResponse`` contract."""
    metadata: dict[str, Any] = {}
    response_id = getattr(response, "response_id", None)
    if response_id:
        metadata["response_id"] = response_id
    model_version = getattr(response, "model_version", None)
    if model_version:
        metadata["model_version"] = model_version
    usage = _normalize_usage(getattr(response, "usage_metadata", None))
    block_reason = _block_reason(response)

    candidates = getattr(response, "candidates", None) or []
    if not candidates:
        if block_reason is not None:
            metadata["block_reason"] = block_reason
            return LLMResponse(
                request_id=request.request_id,
                provider=PROVIDER_ID,
                model=model.model,
                finish_reason=FinishReason.CONTENT_FILTER,
                latency_ms=latency_ms,
                content=None,
                usage=usage,
                metadata=metadata,
            )
        raise MalformedResponseError(
            "Gemini returned no candidates",
            provider=PROVIDER_ID,
            model=model.model,
            request_id=request.request_id,
        )

    candidate = candidates[0]
    finish_reason = _FINISH_REASON_MAP.get(
        _enum_name(getattr(candidate, "finish_reason", None)) or "", FinishReason.UNKNOWN
    )
    content = _extract_text(candidate)
    if block_reason is not None:
        metadata["block_reason"] = block_reason
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


def _retry_after_s(exc: genai_errors.APIError) -> float | None:
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


def _map_api_error(
    exc: genai_errors.APIError,
    *,
    model: str,
    request_id: str,
) -> LLMError:
    """Map a Gemini API error's HTTP status onto the normalized taxonomy."""
    code = getattr(exc, "code", None)
    message = getattr(exc, "message", None) or "Gemini API error"
    context: dict[str, Any] = {
        "provider": PROVIDER_ID,
        "model": model,
        "request_id": request_id,
        "status_code": code if isinstance(code, int) else None,
        "cause": exc,
    }
    if code == 400:
        return InvalidRequestError(message, **context)
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
    if isinstance(exc, genai_errors.APIError):
        return _map_api_error(exc, model=model, request_id=request_id)

    context: dict[str, Any] = {
        "provider": PROVIDER_ID,
        "model": model,
        "request_id": request_id,
        "cause": exc,
    }
    # Check timeouts before the broader transport family (a timeout IS a transport error).
    if isinstance(exc, httpx.TimeoutException | TimeoutError):
        return LLMTimeoutError("Gemini request timed out", **context)
    if isinstance(exc, httpx.TransportError | ConnectionError):
        return NetworkError("Gemini network error", **context)
    return UnknownProviderError(f"unmapped Gemini failure: {type(exc).__name__}", **context)


class GeminiProvider:
    """Concrete :class:`~llm_orchestrator.providers.base.LLMProvider` for Google Gemini."""

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
                "GeminiProvider requires an explicit api_key "
                "(set GEMINI_API_KEY in the environment and pass it in; "
                "the provider never reads the environment itself)"
            )
        self._api_key = api_key
        self._client: Any | None = client

    def __repr__(self) -> str:
        return f"GeminiProvider(provider_id={PROVIDER_ID!r}, api_key=[REDACTED])"

    def _genai_client(self) -> Any:
        """Return the SDK client, creating it lazily on first use."""
        if self._client is None:
            self._client = genai.Client(api_key=self._api_key)
        return self._client

    async def generate(
        self,
        request: LLMRequest,
        model: ModelConfig,
        *,
        timeout_s: float | None = None,
    ) -> LLMResponse:
        """Perform exactly one Gemini generation attempt."""
        require_timeout_s(timeout_s)
        if model.provider != PROVIDER_ID:
            raise InvalidRequestError(
                f"GeminiProvider cannot serve provider {model.provider!r}",
                provider=PROVIDER_ID,
                model=model.model,
                request_id=request.request_id,
            )
        if request.response_schema is not None:
            raise InvalidRequestError(
                "structured output is not implemented by GeminiProvider yet",
                provider=PROVIDER_ID,
                model=model.model,
                request_id=request.request_id,
            )

        system_instruction, contents = _split_messages(request)
        if not contents:
            raise InvalidRequestError(
                "GeminiProvider requires at least one non-system message",
                provider=PROVIDER_ID,
                model=model.model,
                request_id=request.request_id,
            )
        effective_timeout = timeout_s if timeout_s is not None else request.timeout_s
        config = _build_config(request, system_instruction, effective_timeout)

        start = time.perf_counter()
        try:
            response = await self._genai_client().aio.models.generate_content(
                model=model.model,
                contents=contents,
                config=config,
            )
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
        aio = getattr(client, "aio", None)
        aclose = getattr(aio, "aclose", None)
        if callable(aclose):
            await aclose()
        self._client = None
