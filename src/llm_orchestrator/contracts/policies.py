"""Policy contracts.

Phase 2 defines only the *shape* of policies. No retry loops, backoff math,
fallback execution, or routing algorithms are implemented here; those belong to
later phases and consume these contracts.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from enum import StrEnum
from typing import Any, Protocol

from pydantic import BaseModel, Field, field_validator, model_validator

from ..errors import ErrorCategory
from ._config import CONTRACT_MODEL_CONFIG
from .models import CandidateRef, ModelConfig
from .requests import LLMRequest


class JitterStrategy(StrEnum):
    """How randomness is applied to backoff delays."""

    NONE = "none"
    FULL = "full"
    EQUAL = "equal"
    DECORRELATED = "decorrelated"


class RetryPolicy(BaseModel):
    """Whether and how the *same* candidate is attempted again.

    ``retry_on`` overrides the taxonomy defaults; ``None`` means "use each
    error's own retryability".
    """

    model_config = CONTRACT_MODEL_CONFIG

    max_attempts: int = 1
    initial_backoff_s: float = 0.5
    multiplier: float = 2.0
    max_backoff_s: float = 30.0
    jitter: JitterStrategy = JitterStrategy.FULL
    retry_on: frozenset[ErrorCategory] | None = None

    @field_validator("max_attempts")
    @classmethod
    def _validate_attempts(cls, value: int) -> int:
        if value < 1:
            raise ValueError("RetryPolicy.max_attempts must be >= 1")
        return value

    @field_validator("initial_backoff_s", "max_backoff_s")
    @classmethod
    def _validate_delays(cls, value: float) -> float:
        if not math.isfinite(value):
            raise ValueError("backoff delays must be finite")
        if value < 0:
            raise ValueError("backoff delays must be >= 0")
        return value

    @field_validator("multiplier")
    @classmethod
    def _validate_multiplier(cls, value: float) -> float:
        if not math.isfinite(value):
            raise ValueError("RetryPolicy.multiplier must be finite")
        if value < 1:
            raise ValueError("RetryPolicy.multiplier must be >= 1")
        return value

    @model_validator(mode="after")
    def _validate_backoff_window(self) -> RetryPolicy:
        if self.max_backoff_s < self.initial_backoff_s:
            raise ValueError("RetryPolicy.max_backoff_s must be >= initial_backoff_s")
        return self


class FallbackPolicy(BaseModel):
    """Whether execution moves to a *different* candidate.

    Candidate *ordering* is routing's responsibility; ``candidates`` here is an
    explicit override list, not a selection strategy.
    """

    model_config = CONTRACT_MODEL_CONFIG

    max_fallbacks: int = 0
    candidates: tuple[CandidateRef, ...] | None = None
    fallback_on: frozenset[ErrorCategory] | None = None
    include_validation_failures: bool = True
    cross_provider: bool = True

    @field_validator("max_fallbacks")
    @classmethod
    def _validate_max_fallbacks(cls, value: int) -> int:
        if value < 0:
            raise ValueError("FallbackPolicy.max_fallbacks must be >= 0")
        return value

    @field_validator("candidates")
    @classmethod
    def _validate_candidates(
        cls, value: tuple[CandidateRef, ...] | None
    ) -> tuple[CandidateRef, ...] | None:
        if value is not None and len(value) == 0:
            raise ValueError("FallbackPolicy.candidates must not be empty when provided")
        return value


class RoutingContext(BaseModel):
    """Signals available to routing.

    Empty in V1 (static/explicit ordering). Health, latency, and cost signals
    will populate ``metadata`` in a later phase without changing the interface.
    """

    model_config = CONTRACT_MODEL_CONFIG

    metadata: Mapping[str, Any] = Field(default_factory=dict)


class RoutingPolicy(Protocol):
    """Produces an ordered list of candidates, best first.

    Implementations must not perform retries, fallbacks, or provider calls.
    """

    def select(
        self,
        request: LLMRequest,
        candidates: Sequence[ModelConfig],
        context: RoutingContext,
    ) -> tuple[CandidateRef, ...]: ...
