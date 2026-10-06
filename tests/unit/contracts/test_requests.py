"""Tests for the LLMRequest contract."""

from __future__ import annotations

import math

import pytest
from pydantic import ValidationError

from llm_orchestrator.contracts import LLMRequest, Message, MessageRole


def _message() -> Message:
    return Message(role=MessageRole.USER, content="hello")


def test_minimal_request_generates_a_request_id() -> None:
    request = LLMRequest(messages=[_message()])
    assert request.request_id
    assert isinstance(request.request_id, str)
    assert request.temperature is None
    assert request.max_tokens is None
    assert request.stop is None
    assert request.response_schema is None
    assert request.timeout_s is None
    assert dict(request.metadata) == {}


def test_messages_are_coerced_to_an_immutable_tuple() -> None:
    request = LLMRequest(messages=[_message(), Message(role=MessageRole.SYSTEM, content="sys")])
    assert isinstance(request.messages, tuple)
    assert len(request.messages) == 2


def test_message_order_is_preserved() -> None:
    first = Message(role=MessageRole.SYSTEM, content="first")
    second = Message(role=MessageRole.USER, content="second")
    request = LLMRequest(messages=[first, second])
    assert request.messages[0].content == "first"
    assert request.messages[1].content == "second"


def test_request_id_can_be_supplied() -> None:
    request = LLMRequest(messages=[_message()], request_id="explicit-id")
    assert request.request_id == "explicit-id"


def test_empty_messages_rejected() -> None:
    with pytest.raises(ValidationError):
        LLMRequest(messages=[])


@pytest.mark.parametrize("request_id", ["", "   "])
def test_blank_request_id_rejected(request_id: str) -> None:
    with pytest.raises(ValidationError):
        LLMRequest(messages=[_message()], request_id=request_id)


def test_negative_temperature_rejected() -> None:
    with pytest.raises(ValidationError):
        LLMRequest(messages=[_message()], temperature=-0.1)


def test_non_finite_temperature_rejected() -> None:
    with pytest.raises(ValidationError):
        LLMRequest(messages=[_message()], temperature=math.inf)


def test_high_temperature_is_not_arbitrarily_capped() -> None:
    # Providers define their own ranges; the generic contract must not impose one.
    request = LLMRequest(messages=[_message()], temperature=5.0)
    assert request.temperature == 5.0


@pytest.mark.parametrize("max_tokens", [0, -1])
def test_non_positive_max_tokens_rejected(max_tokens: int) -> None:
    with pytest.raises(ValidationError):
        LLMRequest(messages=[_message()], max_tokens=max_tokens)


@pytest.mark.parametrize("timeout_s", [0.0, -2.0])
def test_non_positive_timeout_rejected(timeout_s: float) -> None:
    with pytest.raises(ValidationError):
        LLMRequest(messages=[_message()], timeout_s=timeout_s)


def test_empty_stop_rejected() -> None:
    with pytest.raises(ValidationError):
        LLMRequest(messages=[_message()], stop=())


def test_blank_stop_sequence_rejected() -> None:
    with pytest.raises(ValidationError):
        LLMRequest(messages=[_message()], stop=["STOP", ""])


def test_response_schema_is_stored_as_a_mapping() -> None:
    schema = {"type": "object", "properties": {"summary": {"type": "string"}}}
    request = LLMRequest(messages=[_message()], response_schema=schema)
    assert dict(request.response_schema or {}) == schema


def test_unknown_field_rejected() -> None:
    with pytest.raises(ValidationError):
        LLMRequest(messages=[_message()], gemini_generation_config={"temperature": 1})


def test_request_is_immutable() -> None:
    request = LLMRequest(messages=[_message()])
    with pytest.raises(ValidationError):
        request.temperature = 1.0
