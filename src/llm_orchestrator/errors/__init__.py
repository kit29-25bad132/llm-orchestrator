"""Normalized, provider-independent error taxonomy.

Public surface for the error layer:

    from llm_orchestrator.errors import LLMError, ErrorCategory, RateLimitError
"""

from __future__ import annotations

from .base import LLMError, sanitize_message
from .categories import ErrorCategory
from .exceptions import (
    AuthenticationError,
    AuthorizationError,
    ConfigurationError,
    InvalidRequestError,
    LLMTimeoutError,
    MalformedResponseError,
    ModelUnavailableError,
    NetworkError,
    ProviderUnavailableError,
    RateLimitError,
    StructuredOutputValidationError,
    UnknownProviderError,
)

__all__ = [
    "AuthenticationError",
    "AuthorizationError",
    "ConfigurationError",
    "ErrorCategory",
    "InvalidRequestError",
    "LLMError",
    "LLMTimeoutError",
    "MalformedResponseError",
    "ModelUnavailableError",
    "NetworkError",
    "ProviderUnavailableError",
    "RateLimitError",
    "StructuredOutputValidationError",
    "UnknownProviderError",
    "sanitize_message",
]
