"""Normalized base error and message sanitization.

``LLMError`` is the single base type every provider maps its native failures onto.
It carries enough normalized context for future retry/fallback/routing decisions
without leaking provider SDK objects or secrets.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any, ClassVar

from .categories import ErrorCategory

# Best-effort redaction of common credential shapes. Applied to the message that
# becomes ``safe_message`` (and thus ``str(error)``), so secrets do not reach
# logs by accident. This is defense in depth, not a substitute for never putting
# secrets in messages in the first place.
#
# Each pattern may define at most one capturing group. When present, that group
# is a context prefix kept in the output (for example ``Bearer ``); the
# credential itself must never be captured.
_SECRET_PATTERNS: tuple[re.Pattern[str], ...] = (
    # ``Bearer <token>`` -> keep ``Bearer ``.
    re.compile(r"(?i)(bearer\s+)[A-Za-z0-9._\-]+"),
    # ``key = <value>`` style assignments -> keep ``key = ``.
    re.compile(r"(?i)((?:api[_-]?key|key|token|secret|password|authorization)\s*[:=]\s*)\S+"),
    # Bare, provider-shaped API keys -> nothing is kept.
    re.compile(r"\b(?:AIza[0-9A-Za-z_\-]{8,}|sk-[A-Za-z0-9]{8,}|gsk_[A-Za-z0-9]{8,})\b"),
)


def sanitize_message(message: str) -> str:
    """Return ``message`` with common credential patterns replaced by ``[REDACTED]``."""
    sanitized = message
    for pattern in _SECRET_PATTERNS:
        sanitized = pattern.sub(_redact_match, sanitized)
    return sanitized


def _redact_match(match: re.Match[str]) -> str:
    prefix = match.group(1) if match.lastindex == 1 else ""
    return f"{prefix}[REDACTED]"


class LLMError(Exception):
    """Provider-independent failure.

    Subclasses declare their ``category``; the default ``retryable`` and
    ``fallback_eligible`` flags are derived from that category and may be
    overridden per instance when a specific call warrants it.

    The exception's message is sanitized (``safe_message``), and ``str(error)``
    returns the sanitized form, so accidental logging cannot leak credentials.
    """

    category: ClassVar[ErrorCategory] = ErrorCategory.UNKNOWN

    def __init__(
        self,
        message: str,
        *,
        provider: str | None = None,
        model: str | None = None,
        request_id: str | None = None,
        attempt: int | None = None,
        status_code: int | None = None,
        cause: BaseException | None = None,
        retryable: bool | None = None,
        fallback_eligible: bool | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        self.message = message
        self.safe_message = sanitize_message(message)
        self.provider = provider
        self.model = model
        self.request_id = request_id
        self.attempt = attempt
        self.status_code = status_code
        self.cause = cause
        self.metadata: Mapping[str, Any] = dict(metadata) if metadata is not None else {}
        self.retryable = self.category.retryable_by_default if retryable is None else retryable
        self.fallback_eligible = (
            self.category.fallback_eligible_by_default
            if fallback_eligible is None
            else fallback_eligible
        )
        super().__init__(self.safe_message)

    def __str__(self) -> str:
        return self.safe_message
