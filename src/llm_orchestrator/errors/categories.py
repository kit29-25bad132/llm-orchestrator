"""Normalized error categories and their default policy semantics.

This module is intentionally free of third-party dependencies so the taxonomy can
be imported anywhere. The category is the single source of truth for whether an
error is retryable / fallback-eligible by default; concrete error classes and
per-consumer policies may still override those flags explicitly.
"""

from __future__ import annotations

from enum import StrEnum


class ErrorCategory(StrEnum):
    """Provider-independent classification of an LLM failure."""

    AUTHENTICATION = "authentication"
    AUTHORIZATION = "authorization"
    RATE_LIMIT = "rate_limit"
    TIMEOUT = "timeout"
    NETWORK = "network"
    PROVIDER_UNAVAILABLE = "provider_unavailable"
    MODEL_UNAVAILABLE = "model_unavailable"
    INVALID_REQUEST = "invalid_request"
    MALFORMED_RESPONSE = "malformed_response"
    STRUCTURED_OUTPUT_VALIDATION = "structured_output_validation"
    CONFIGURATION = "configuration"
    UNKNOWN = "unknown"

    @property
    def retryable_by_default(self) -> bool:
        """Whether retrying the *same* candidate is sensible by default."""
        return self in _RETRYABLE_BY_DEFAULT

    @property
    def fallback_eligible_by_default(self) -> bool:
        """Whether moving to a *different* candidate is sensible by default."""
        return self in _FALLBACK_ELIGIBLE_BY_DEFAULT


# Transient conditions worth another attempt on the same candidate.
_RETRYABLE_BY_DEFAULT: frozenset[ErrorCategory] = frozenset(
    {
        ErrorCategory.RATE_LIMIT,
        ErrorCategory.TIMEOUT,
        ErrorCategory.NETWORK,
        ErrorCategory.PROVIDER_UNAVAILABLE,
        ErrorCategory.MALFORMED_RESPONSE,
        ErrorCategory.STRUCTURED_OUTPUT_VALIDATION,
    }
)

# Conditions where trying a different model/provider may succeed. Deliberately
# excludes CONFIGURATION (programmer/operator error) and INVALID_REQUEST
# (deterministic for this request, though a policy may still opt in).
_FALLBACK_ELIGIBLE_BY_DEFAULT: frozenset[ErrorCategory] = frozenset(
    {
        ErrorCategory.AUTHENTICATION,
        ErrorCategory.AUTHORIZATION,
        ErrorCategory.RATE_LIMIT,
        ErrorCategory.TIMEOUT,
        ErrorCategory.NETWORK,
        ErrorCategory.PROVIDER_UNAVAILABLE,
        ErrorCategory.MODEL_UNAVAILABLE,
        ErrorCategory.MALFORMED_RESPONSE,
        ErrorCategory.STRUCTURED_OUTPUT_VALIDATION,
        ErrorCategory.UNKNOWN,
    }
)
