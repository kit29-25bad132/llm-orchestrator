"""Behavior tests for structured-output validation.

All schemas below are deliberately application-neutral: the library owns no
domain model and must accept whatever Pydantic schema a caller supplies.
"""

from __future__ import annotations

import json
from enum import StrEnum
from typing import Any

import pytest
from pydantic import BaseModel, ConfigDict, ValidationError

from llm_orchestrator.contracts import ErrorInfo, FinishReason, LLMResponse
from llm_orchestrator.errors import ErrorCategory, StructuredOutputValidationError
from llm_orchestrator.structured import validate_structured_output


class Color(StrEnum):
    RED = "red"
    GREEN = "green"


class Address(BaseModel):
    street: str
    zip_code: str


class UserProfile(BaseModel):
    name: str
    age: int
    nickname: str | None = None
    address: Address | None = None
    tags: list[str] = []
    active: bool = True
    favorite_color: Color = Color.RED


class StrictProfile(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    age: int


def _response(
    content: str | None,
    *,
    provider: str = "acme",
    model: str = "m-1",
    request_id: str = "req-9",
) -> LLMResponse:
    return LLMResponse(
        request_id=request_id,
        provider=provider,
        model=model,
        finish_reason=FinishReason.STOP,
        latency_ms=1.0,
        content=content,
    )


def _validate(content: str | None, **kwargs: Any) -> UserProfile:
    return validate_structured_output(_response(content, **kwargs), UserProfile)


# --- happy path -------------------------------------------------------------


def test_valid_object_validates() -> None:
    profile = _validate('{"name": "Ada", "age": 36}')
    assert isinstance(profile, UserProfile)
    assert type(profile) is UserProfile
    assert (profile.name, profile.age) == ("Ada", 36)


def test_optional_and_defaulted_fields() -> None:
    profile = _validate('{"name": "Ada", "age": 36, "nickname": "a"}')
    assert profile.nickname == "a"
    assert profile.tags == []
    assert profile.active is True
    assert profile.favorite_color is Color.RED


def test_nested_model_lists_booleans_and_enums() -> None:
    profile = _validate(
        '{"name": "Ada", "age": 36, "tags": ["x", "y"], "active": false,'
        ' "favorite_color": "green",'
        ' "address": {"street": "Main", "zip_code": "12345"}}'
    )
    assert profile.address == Address(street="Main", zip_code="12345")
    assert profile.tags == ["x", "y"]
    assert profile.active is False
    assert profile.favorite_color is Color.GREEN


def test_surrounding_whitespace_is_ignored() -> None:
    assert _validate('  \n {"name": "Ada", "age": 1} \t\n').age == 1


def test_fenced_json_block_is_accepted() -> None:
    profile = _validate('```json\n{"name": "Ada", "age": 7}\n```')
    assert (profile.name, profile.age) == ("Ada", 7)


def test_bare_and_uppercase_fences_are_accepted() -> None:
    assert _validate('```\n{"name": "A", "age": 1}\n```').age == 1
    assert _validate('```JSON\n{"name": "A", "age": 2}\n```').age == 2


@pytest.mark.parametrize("provider", ["acme", "gemini", "groq"])
def test_validation_is_provider_agnostic(provider: str) -> None:
    profile = _validate('{"name": "Ada", "age": 3}', provider=provider)
    assert profile.age == 3


# --- JSON failures ----------------------------------------------------------


@pytest.mark.parametrize("content", [None, "", "   \n\t  "])
def test_missing_or_blank_content_fails_at_extract(content: str | None) -> None:
    with pytest.raises(StructuredOutputValidationError) as excinfo:
        _validate(content)
    assert excinfo.value.metadata["stage"] == "extract"


@pytest.mark.parametrize(
    "content",
    [
        '{"name": "Ada", "age": 36',  # truncated
        "The user is Ada and she is 36.",  # prose
        "{name: Ada}",  # not JSON at all
        '```json\n{"name": "Ada"}\n',  # unclosed fence
        '```python\n{"name": "Ada", "age": 1}\n```',  # wrong fence tag
        'Here you go:\n```json\n{"name": "Ada", "age": 1}\n```',  # fence not leading
    ],
)
def test_unparseable_content_fails(content: str) -> None:
    with pytest.raises(StructuredOutputValidationError) as excinfo:
        _validate(content)
    error = excinfo.value
    assert error.category is ErrorCategory.STRUCTURED_OUTPUT_VALIDATION
    assert error.metadata["stage"] in ("extract", "parse")


@pytest.mark.parametrize("content", ["[1, 2, 3]", "42", '"just a string"', "true", "null"])
def test_non_object_json_fails_schema_validation(content: str) -> None:
    with pytest.raises(StructuredOutputValidationError) as excinfo:
        _validate(content)
    assert excinfo.value.metadata["stage"] == "validate"


# --- schema failures --------------------------------------------------------


def test_missing_required_field() -> None:
    with pytest.raises(StructuredOutputValidationError) as excinfo:
        _validate('{"name": "Ada"}')
    assert "age" in str(excinfo.value)


def test_wrong_field_type() -> None:
    with pytest.raises(StructuredOutputValidationError) as excinfo:
        _validate('{"name": "Ada", "age": "thirty-six"}')
    message = str(excinfo.value)
    assert "age" in message
    assert "integer" in message


def test_invalid_nested_object() -> None:
    with pytest.raises(StructuredOutputValidationError) as excinfo:
        _validate('{"name": "Ada", "age": 1, "address": {"street": "Main"}}')
    assert "zip_code" in str(excinfo.value)


def test_invalid_list_element() -> None:
    with pytest.raises(StructuredOutputValidationError) as excinfo:
        _validate('{"name": "Ada", "age": 1, "tags": ["ok", 7]}')
    assert "tags" in str(excinfo.value)


def test_invalid_enum_value() -> None:
    with pytest.raises(StructuredOutputValidationError) as excinfo:
        _validate('{"name": "Ada", "age": 1, "favorite_color": "blue"}')
    assert "favorite_color" in str(excinfo.value)


def test_extra_fields_follow_the_schema_configuration() -> None:
    # Pydantic's default ignores unexpected fields.
    assert _validate('{"name": "Ada", "age": 1, "zzz": 2}').age == 1
    # ... while a forbidding schema rejects them.
    with pytest.raises(StructuredOutputValidationError):
        validate_structured_output(_response('{"name": "Ada", "age": 1, "zzz": 2}'), StrictProfile)


# --- error behavior ---------------------------------------------------------


def test_error_carries_taxonomy_semantics() -> None:
    with pytest.raises(StructuredOutputValidationError) as excinfo:
        _validate("not json")
    error = excinfo.value
    assert error.category is ErrorCategory.STRUCTURED_OUTPUT_VALIDATION
    assert error.retryable is ErrorCategory.STRUCTURED_OUTPUT_VALIDATION.retryable_by_default
    assert (
        error.fallback_eligible
        is ErrorCategory.STRUCTURED_OUTPUT_VALIDATION.fallback_eligible_by_default
    )


def test_error_preserves_provenance() -> None:
    with pytest.raises(StructuredOutputValidationError) as excinfo:
        _validate("not json", provider="acme", model="m-1", request_id="req-9")
    error = excinfo.value
    assert (error.provider, error.model, error.request_id) == ("acme", "m-1", "req-9")
    assert error.metadata["schema"] == "UserProfile"
    assert error.metadata["stage"] == "parse"


def test_error_metadata_stays_small_for_huge_outputs() -> None:
    with pytest.raises(StructuredOutputValidationError) as excinfo:
        _validate("x" * 200_000)
    error = excinfo.value
    assert len(error.metadata["content_preview"]) <= 500
    assert "x" * 10_000 not in str(error)


def test_parse_failure_chains_json_error() -> None:
    with pytest.raises(StructuredOutputValidationError) as excinfo:
        _validate('{"name":')
    assert isinstance(excinfo.value.cause, json.JSONDecodeError)


def test_schema_failure_chains_pydantic_error() -> None:
    with pytest.raises(StructuredOutputValidationError) as excinfo:
        _validate('{"name": "Ada"}')
    assert isinstance(excinfo.value.cause, ValidationError)


def test_non_model_schema_is_a_programmer_error() -> None:
    response = _response('{"name": "Ada", "age": 1}')
    with pytest.raises(TypeError):
        validate_structured_output(response, dict)  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        validate_structured_output(response, None)  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        validate_structured_output(response, UserProfile(name="Ada", age=1))  # type: ignore[arg-type]


# --- security ---------------------------------------------------------------


def test_secrets_never_reach_error_surfaces() -> None:
    key = "sk-" + "s3cr3t" * 4
    token = "Bearer abc.def.ghi"
    content = f'{{"nickname": "{key}", "tags": ["{token}"]}}'
    with pytest.raises(StructuredOutputValidationError) as excinfo:
        _validate(content)
    error = excinfo.value
    assert key not in str(error)
    assert key not in error.safe_message
    assert token not in str(error)
    assert all(key not in str(value) for value in error.metadata.values())
    snapshot = ErrorInfo.from_error(error).model_dump_json()
    assert key not in snapshot
    assert token not in snapshot


def test_hostile_payload_shapes_are_inert_data() -> None:
    # Suspicious keys are plain ignored data under the default config — never
    # evaluated, never executed.
    profile = _validate('{"name": "__class__", "age": 1, "__import__": "os"}')
    assert profile.name == "__class__"
    # ... and still rejected where the schema forbids surprises.
    with pytest.raises(StructuredOutputValidationError):
        validate_structured_output(
            _response('{"name": "Ada", "age": 1, "__import__": "os"}'), StrictProfile
        )
    with pytest.raises(StructuredOutputValidationError):
        _validate('{"name": "Ada", "age": 1, "tags": [{"$where": "evil()"}]}')
