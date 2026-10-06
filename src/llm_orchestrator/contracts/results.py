"""Orchestration outcome contracts.

``LLMResponse`` is the result of one successful invocation; ``OrchestrationResult``
is the outcome of the whole run. The result carries a serializable
:class:`ErrorInfo` rather than a live exception so results stay JSON-friendly.
"""

from __future__ import annotations

import math

from pydantic import BaseModel, field_validator, model_validator

from ..errors import ErrorCategory, LLMError
from ._config import CONTRACT_MODEL_CONFIG
from .models import CandidateRef
from .responses import LLMResponse


class AttemptRecord(BaseModel):
    """One attempt against one candidate."""

    model_config = CONTRACT_MODEL_CONFIG

    provider: str
    model: str
    attempt: int
    succeeded: bool
    latency_ms: float | None = None
    error_category: ErrorCategory | None = None

    @field_validator("provider", "model")
    @classmethod
    def _validate_identifiers(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("AttemptRecord provider/model must not be blank")
        return value

    @field_validator("attempt")
    @classmethod
    def _validate_attempt(cls, value: int) -> int:
        if value < 1:
            raise ValueError("AttemptRecord.attempt must be >= 1")
        return value

    @field_validator("latency_ms")
    @classmethod
    def _validate_latency(cls, value: float | None) -> float | None:
        if value is not None:
            if not math.isfinite(value):
                raise ValueError("AttemptRecord.latency_ms must be finite")
            if value < 0:
                raise ValueError("AttemptRecord.latency_ms must be >= 0")
        return value

    @model_validator(mode="after")
    def _validate_outcome(self) -> AttemptRecord:
        if self.succeeded and self.error_category is not None:
            raise ValueError("a successful attempt must not carry an error_category")
        if not self.succeeded and self.error_category is None:
            raise ValueError("a failed attempt must carry an error_category")
        return self


class ErrorInfo(BaseModel):
    """Serializable, sanitized snapshot of an :class:`LLMError`."""

    model_config = CONTRACT_MODEL_CONFIG

    category: ErrorCategory
    message: str
    retryable: bool
    fallback_eligible: bool
    provider: str | None = None
    model: str | None = None
    request_id: str | None = None
    attempt: int | None = None
    status_code: int | None = None

    @classmethod
    def from_error(cls, error: LLMError) -> ErrorInfo:
        """Build an ``ErrorInfo`` from a live error, using its sanitized message."""
        return cls(
            category=error.category,
            message=error.safe_message,
            retryable=error.retryable,
            fallback_eligible=error.fallback_eligible,
            provider=error.provider,
            model=error.model,
            request_id=error.request_id,
            attempt=error.attempt,
            status_code=error.status_code,
        )


class OrchestrationResult(BaseModel):
    """The complete, observable outcome of one orchestration run."""

    model_config = CONTRACT_MODEL_CONFIG

    request_id: str
    succeeded: bool
    response: LLMResponse | None = None
    error: ErrorInfo | None = None
    candidates_tried: tuple[CandidateRef, ...] = ()
    attempts: tuple[AttemptRecord, ...] = ()
    total_latency_ms: float | None = None

    @field_validator("request_id")
    @classmethod
    def _validate_request_id(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("OrchestrationResult.request_id must not be blank")
        return value

    @field_validator("total_latency_ms")
    @classmethod
    def _validate_total_latency(cls, value: float | None) -> float | None:
        if value is not None:
            if not math.isfinite(value):
                raise ValueError("OrchestrationResult.total_latency_ms must be finite")
            if value < 0:
                raise ValueError("OrchestrationResult.total_latency_ms must be >= 0")
        return value

    @model_validator(mode="after")
    def _validate_outcome(self) -> OrchestrationResult:
        if self.succeeded:
            if self.response is None:
                raise ValueError("a succeeded result must include a response")
            if self.error is not None:
                raise ValueError("a succeeded result must not include an error")
        else:
            if self.response is not None:
                raise ValueError("a failed result must not include a response")
            if self.error is None:
                raise ValueError("a failed result must include an error")
        return self

    @property
    def fallback_count(self) -> int:
        """Number of times execution moved to a new candidate."""
        return max(0, len(self.candidates_tried) - 1)

    @property
    def retry_count(self) -> int:
        """Number of attempts beyond the first attempt per candidate."""
        return max(0, len(self.attempts) - len(self.candidates_tried))
