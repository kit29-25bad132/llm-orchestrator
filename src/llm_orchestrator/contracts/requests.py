"""Provider-neutral request contract."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field, field_validator

from ._config import CONTRACT_MODEL_CONFIG
from .messages import Message


def _new_request_id() -> str:
    """Generate a default request identifier."""
    return uuid4().hex


class LLMRequest(BaseModel):
    """A logical generation request, independent of who serves it.

    The request expresses *intent* only: it never names a provider or model and
    never carries credentials. Choosing the candidate is the routing layer's job
    (a routing tag may ride in ``metadata``).
    """

    model_config = CONTRACT_MODEL_CONFIG

    messages: tuple[Message, ...]
    request_id: str = Field(default_factory=_new_request_id)

    # Generation intent (all provider-neutral).
    temperature: float | None = None
    max_tokens: int | None = None
    stop: tuple[str, ...] | None = None

    # Structured output target as a JSON Schema document. Adapters convert a
    # consumer's own schema (for example a Pydantic model) into this form.
    response_schema: Mapping[str, Any] | None = None

    timeout_s: float | None = None
    metadata: Mapping[str, Any] = Field(default_factory=dict)

    @field_validator("messages")
    @classmethod
    def _validate_messages(cls, value: tuple[Message, ...]) -> tuple[Message, ...]:
        if not value:
            raise ValueError("LLMRequest.messages must contain at least one message")
        return value

    @field_validator("request_id")
    @classmethod
    def _validate_request_id(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("LLMRequest.request_id must not be blank")
        return value

    @field_validator("temperature")
    @classmethod
    def _validate_temperature(cls, value: float | None) -> float | None:
        if value is not None:
            if not math.isfinite(value):
                raise ValueError("LLMRequest.temperature must be finite")
            if value < 0:
                raise ValueError("LLMRequest.temperature must be >= 0")
        return value

    @field_validator("max_tokens")
    @classmethod
    def _validate_max_tokens(cls, value: int | None) -> int | None:
        if value is not None and value <= 0:
            raise ValueError("LLMRequest.max_tokens must be a positive integer")
        return value

    @field_validator("timeout_s")
    @classmethod
    def _validate_timeout(cls, value: float | None) -> float | None:
        if value is not None:
            if not math.isfinite(value):
                raise ValueError("LLMRequest.timeout_s must be finite")
            if value <= 0:
                raise ValueError("LLMRequest.timeout_s must be > 0")
        return value

    @field_validator("stop")
    @classmethod
    def _validate_stop(cls, value: tuple[str, ...] | None) -> tuple[str, ...] | None:
        if value is not None:
            if len(value) == 0:
                raise ValueError("LLMRequest.stop must not be empty when provided")
            if any(not item for item in value):
                raise ValueError("LLMRequest.stop sequences must not be empty strings")
        return value
