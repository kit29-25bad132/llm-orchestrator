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
    ProviderNotRegisteredError,
    ProviderUnavailableError,
    RateLimitError,
    StructuredOutputValidationError,
    UnknownProviderError,
)
from .exceptions import (
    require_timeout_s as require_timeout_s,
)

# ``require_timeout_s`` is explicitly re-exported for use by the execution
# seams (retry, fallback, orchestration, adapters) but is deliberately not
# listed in ``__all__``: it is internal plumbing, not advertised public API.

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
    "ProviderNotRegisteredError",
    "ProviderUnavailableError",
    "RateLimitError",
    "StructuredOutputValidationError",
    "UnknownProviderError",
    "sanitize_message",
]
