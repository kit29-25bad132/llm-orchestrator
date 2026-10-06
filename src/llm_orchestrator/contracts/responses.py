"""Normalized response and usage contracts."""

from __future__ import annotations

import math
from collections.abc import Mapping
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field, field_validator

from ._config import CONTRACT_MODEL_CONFIG


class FinishReason(StrEnum):
    """Provider-neutral reason a generation stopped."""

    STOP = "stop"
    LENGTH = "length"
    TOOL_CALL = "tool_call"
    CONTENT_FILTER = "content_filter"
    ERROR = "error"
    UNKNOWN = "unknown"


class Usage(BaseModel):
    """Token accounting for one call.

    Every field is optional because providers do not uniformly report usage.
    Values are never estimated here; unknown values stay ``None``.
    """

    model_config = CONTRACT_MODEL_CONFIG

    input_tokens: int | None = None
    output_tokens: int | None = None
    total_tokens: int | None = None

    @field_validator("input_tokens", "output_tokens", "total_tokens")
    @classmethod
    def _validate_non_negative(cls, value: int | None) -> int | None:
        if value is not None and value < 0:
            raise ValueError("usage token counts must be >= 0")
        return value


class LLMResponse(BaseModel):
    """The normalized successful result of one model invocation.

    Provider quirks are erased here; provenance (``provider``/``model``) is kept
    so behavior stays auditable. Raw provider SDK objects are deliberately not
    exposed.
    """

    model_config = CONTRACT_MODEL_CONFIG

    request_id: str
    provider: str
    model: str
    finish_reason: FinishReason
    latency_ms: float

    content: str | None = None
    structured: Mapping[str, Any] | None = None
    usage: Usage | None = None
    metadata: Mapping[str, Any] = Field(default_factory=dict)

    @field_validator("request_id", "provider", "model")
    @classmethod
    def _validate_identifiers(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("identifier fields must not be blank")
        return value

    @field_validator("latency_ms")
    @classmethod
    def _validate_latency(cls, value: float) -> float:
        if not math.isfinite(value):
            raise ValueError("LLMResponse.latency_ms must be finite")
        if value < 0:
            raise ValueError("LLMResponse.latency_ms must be >= 0")
        return value
