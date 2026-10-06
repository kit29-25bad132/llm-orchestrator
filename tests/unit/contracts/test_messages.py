"""Tests for the Message contract."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from llm_orchestrator.contracts import Message, MessageRole


def test_valid_message() -> None:
    message = Message(role=MessageRole.USER, content="hello")
    assert message.role is MessageRole.USER
    assert message.content == "hello"
    assert message.name is None
    assert dict(message.metadata) == {}


def test_role_accepts_string_values() -> None:
    assert Message(role="system", content="be terse").role is MessageRole.SYSTEM


def test_all_roles_are_supported() -> None:
    for role in ("system", "user", "assistant", "tool"):
        assert Message(role=role, content="x").role == role


def test_invalid_role_rejected() -> None:
    with pytest.raises(ValidationError):
        Message(role="wizard", content="x")


@pytest.mark.parametrize("content", ["", "   ", "\n\t"])
def test_blank_content_rejected(content: str) -> None:
    with pytest.raises(ValidationError):
        Message(role=MessageRole.USER, content=content)


def test_content_is_not_stripped() -> None:
    message = Message(role=MessageRole.USER, content="  spaced  ")
    assert message.content == "  spaced  "


def test_blank_name_rejected() -> None:
    with pytest.raises(ValidationError):
        Message(role=MessageRole.USER, content="x", name="  ")


def test_unknown_field_rejected() -> None:
    with pytest.raises(ValidationError):
        Message(role=MessageRole.USER, content="x", gemini_parts=[1, 2, 3])


def test_message_is_immutable() -> None:
    message = Message(role=MessageRole.USER, content="x")
    with pytest.raises(ValidationError):
        message.content = "y"
