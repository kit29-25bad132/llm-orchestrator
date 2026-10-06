"""Provider-neutral conversational message contract."""

from __future__ import annotations

from collections.abc import Mapping
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field, field_validator

from ._config import CONTRACT_MODEL_CONFIG


class MessageRole(StrEnum):
    """Roles a message can take in a provider-neutral conversation."""

    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


class Message(BaseModel):
    """One conversational turn.

    The system instruction, when present, is a message with role ``SYSTEM``
    rather than a separate field, so there is a single source of truth for the
    conversation. Provider adapters translate this shape into their own format.
    """

    model_config = CONTRACT_MODEL_CONFIG

    role: MessageRole
    content: str
    name: str | None = None
    metadata: Mapping[str, Any] = Field(default_factory=dict)

    @field_validator("content")
    @classmethod
    def _validate_content(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Message.content must not be blank")
        return value

    @field_validator("name")
    @classmethod
    def _validate_name(cls, value: str | None) -> str | None:
        if value is not None and not value.strip():
            raise ValueError("Message.name must not be blank when provided")
        return value
