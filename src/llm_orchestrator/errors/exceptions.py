"""Concrete normalized error types, one per :class:`ErrorCategory`.

Each class fixes its category; retryability and fallback eligibility default from
the category but can still be overridden per instance.
"""

from __future__ import annotations

from typing import Any

from .base import LLMError
from .categories import ErrorCategory


class AuthenticationError(LLMError):
    """Provider rejected the credentials (invalid/expired key)."""

    category = ErrorCategory.AUTHENTICATION


class AuthorizationError(LLMError):
    """Credentials are valid but lack permission/scope for the operation."""

    category = ErrorCategory.AUTHORIZATION


class RateLimitError(LLMError):
    """Provider throttled the request."""

    category = ErrorCategory.RATE_LIMIT

    def __init__(self, message: str, *, retry_after_s: float | None = None, **kwargs: Any) -> None:
        super().__init__(message, **kwargs)
        self.retry_after_s = retry_after_s


class LLMTimeoutError(LLMError):
    """The request exceeded its time budget. Named to avoid shadowing ``TimeoutError``."""

    category = ErrorCategory.TIMEOUT


class NetworkError(LLMError):
    """Connection-level failure (DNS, TLS, reset, unreachable host)."""

    category = ErrorCategory.NETWORK


class ProviderUnavailableError(LLMError):
    """Provider-side outage or 5xx that is expected to be temporary."""

    category = ErrorCategory.PROVIDER_UNAVAILABLE


class ModelUnavailableError(LLMError):
    """The requested model does not exist, was deprecated, or is overloaded."""

    category = ErrorCategory.MODEL_UNAVAILABLE


class InvalidRequestError(LLMError):
    """The request is invalid for this provider/model (bad params, context too long)."""

    category = ErrorCategory.INVALID_REQUEST


class MalformedResponseError(LLMError):
    """The provider returned a response that could not be parsed/normalized."""

    category = ErrorCategory.MALFORMED_RESPONSE


class StructuredOutputValidationError(LLMError):
    """Generated output did not satisfy the requested schema."""

    category = ErrorCategory.STRUCTURED_OUTPUT_VALIDATION


class ConfigurationError(LLMError):
    """Misconfiguration (missing credential, bad policy). Never retried."""

    category = ErrorCategory.CONFIGURATION


class ProviderNotRegisteredError(ConfigurationError):
    """No provider is registered under the requested identity.

    A lookup miss is an operator/programmer mistake, so it inherits
    ``CONFIGURATION`` semantics (never retried, never fallen back).
    """


class UnknownProviderError(LLMError):
    """Fallback classification for an unmapped provider failure."""

    category = ErrorCategory.UNKNOWN
