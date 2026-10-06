"""Offline tests for the Groq provider adapter.

These exercise the real adapter code (translation, normalization, error mapping)
against a fake SDK *client*, so no network access or API key is required. The
real ``groq`` types are used for responses and errors so the tests cover the
actual translation and normalization logic.
"""

from __future__ import annotations

import asyncio
from typing import Any

import groq
import httpx
import pytest

from llm_orchestrator.contracts import (
    FinishReason,
    LLMRequest,
    LLMResponse,
    Message,
    MessageRole,
    ModelConfig,
)
from llm_orchestrator.errors import (
    AuthenticationError,
    AuthorizationError,
    ConfigurationError,
    ErrorCategory,
    InvalidRequestError,
    LLMError,
    LLMTimeoutError,
    MalformedResponseError,
    ModelUnavailableError,
    NetworkError,
    ProviderUnavailableError,
    RateLimitError,
    UnknownProviderError,
)
from llm_orchestrator.providers import LLMProvider
from llm_orchestrator.providers.groq import PROVIDER_ID, GroqProvider

FAKE_KEY = "gsk_test-key-abcdef1234567890"


class _FakeCompletions:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []
        self.result: Any | None = None
        self.error: Exception | None = None

    async def create(self, **kwargs: Any) -> Any:
        self.calls.append(dict(kwargs))
        if self.error is not None:
            raise self.error
        assert self.result is not None
        return self.result


class _FakeChat:
    def __init__(self) -> None:
        self.completions = _FakeCompletions()


class _FakeClient:
    def __init__(self) -> None:
        self.chat = _FakeChat()
        self.closed = False

    async def close(self) -> None:
        self.closed = True


def _provider() -> tuple[GroqProvider, _FakeClient]:
    fake = _FakeClient()
    return GroqProvider(api_key=FAKE_KEY, client=fake), fake


def _request(
    *messages: Message,
    **kwargs: Any,
) -> LLMRequest:
    return LLMRequest(
        messages=list(messages) or [Message(role=MessageRole.USER, content="hi")],
        **kwargs,
    )


def _model(model: str = "groq-test-model", provider: str = PROVIDER_ID) -> ModelConfig:
    return ModelConfig(provider=provider, model=model)


def _groq_response(
    *,
    text: str | None = "hello",
    finish_reason: str = "stop",
    with_choices: bool = True,
    with_usage: bool = True,
    response_id: str | None = "chatcmpl-1",
) -> Any:
    from groq.types.chat import ChatCompletion
    from groq.types.chat.chat_completion import Choice
    from groq.types.chat.chat_completion_message import ChatCompletionMessage
    from groq.types.completion_usage import CompletionUsage

    kwargs: dict[str, Any] = {
        "id": response_id or "chatcmpl-1",
        "created": 1700000000,
        "model": "groq-test-model",
        "object": "chat.completion",
    }
    if with_choices:
        kwargs["choices"] = [
            Choice(
                index=0,
                finish_reason=finish_reason,  # type: ignore[arg-type]
                message=ChatCompletionMessage(role="assistant", content=text),
            )
        ]
    else:
        kwargs["choices"] = []
    if with_usage:
        kwargs["usage"] = CompletionUsage(prompt_tokens=3, completion_tokens=5, total_tokens=8)
    return ChatCompletion(**kwargs)


def _run(
    provider: GroqProvider, request: LLMRequest, model: ModelConfig, **kwargs: Any
) -> LLMResponse:
    return asyncio.run(provider.generate(request, model, **kwargs))


# --- A. construction & identity -------------------------------------------------


def test_provider_satisfies_the_protocol() -> None:
    provider, _ = _provider()
    assert isinstance(provider, LLMProvider)
    assert provider.provider_id == "groq"


def test_missing_api_key_is_a_configuration_error() -> None:
    with pytest.raises(ConfigurationError):
        GroqProvider()


def test_blank_api_key_is_a_configuration_error() -> None:
    with pytest.raises(ConfigurationError):
        GroqProvider("   ")


def test_injected_client_does_not_require_an_api_key() -> None:
    provider = GroqProvider(client=_FakeClient())
    assert provider.provider_id == "groq"


# --- B. message translation -----------------------------------------------------


def test_user_message_is_passed_through_in_order() -> None:
    provider, fake = _provider()
    fake.chat.completions.result = _groq_response()
    _run(provider, _request(Message(role=MessageRole.USER, content="hello")), _model())
    messages = fake.chat.completions.calls[0]["messages"]
    assert messages == [{"role": "user", "content": "hello"}]


def test_system_assistant_history_is_preserved_verbatim() -> None:
    provider, fake = _provider()
    fake.chat.completions.result = _groq_response()
    request = _request(
        Message(role=MessageRole.SYSTEM, content="be terse"),
        Message(role=MessageRole.USER, content="q"),
        Message(role=MessageRole.ASSISTANT, content="a"),
        Message(role=MessageRole.USER, content="q2"),
    )
    _run(provider, request, _model())
    messages = fake.chat.completions.calls[0]["messages"]
    assert messages == [
        {"role": "system", "content": "be terse"},
        {"role": "user", "content": "q"},
        {"role": "assistant", "content": "a"},
        {"role": "user", "content": "q2"},
    ]


def test_multiple_system_messages_are_all_forwarded() -> None:
    provider, fake = _provider()
    fake.chat.completions.result = _groq_response()
    request = _request(
        Message(role=MessageRole.SYSTEM, content="one"),
        Message(role=MessageRole.SYSTEM, content="two"),
        Message(role=MessageRole.USER, content="hi"),
    )
    _run(provider, request, _model())
    messages = fake.chat.completions.calls[0]["messages"]
    assert [m["role"] for m in messages] == ["system", "system", "user"]


def test_tool_message_role_is_rejected_explicitly() -> None:
    provider, fake = _provider()
    request = _request(Message(role=MessageRole.TOOL, content="tool output"))
    with pytest.raises(InvalidRequestError):
        _run(provider, request, _model())
    assert fake.chat.completions.calls == []


# --- C. generation parameters & timeout -----------------------------------------


def test_model_is_propagated_from_model_config() -> None:
    provider, fake = _provider()
    fake.chat.completions.result = _groq_response()
    _run(provider, _request(), _model("llama-3.3-70b-versatile"))
    assert fake.chat.completions.calls[0]["model"] == "llama-3.3-70b-versatile"


def test_generation_parameters_are_mapped() -> None:
    provider, fake = _provider()
    fake.chat.completions.result = _groq_response()
    request = _request(temperature=0.3, max_tokens=256, stop=("END",))
    _run(provider, request, _model())
    call = fake.chat.completions.calls[0]
    assert call["temperature"] == 0.3
    assert call["max_completion_tokens"] == 256
    assert list(call["stop"]) == ["END"]


def test_absent_generation_parameters_are_not_forwarded() -> None:
    provider, fake = _provider()
    fake.chat.completions.result = _groq_response()
    _run(provider, _request(), _model())
    call = fake.chat.completions.calls[0]
    assert "temperature" not in call
    assert "max_completion_tokens" not in call
    assert "stop" not in call


def test_explicit_timeout_is_forwarded() -> None:
    provider, fake = _provider()
    fake.chat.completions.result = _groq_response()
    _run(provider, _request(), _model(), timeout_s=2.5)
    assert fake.chat.completions.calls[0]["timeout"] == 2.5


def test_request_level_timeout_is_used_when_no_override() -> None:
    provider, fake = _provider()
    fake.chat.completions.result = _groq_response()
    _run(provider, _request(timeout_s=1.0), _model())
    assert fake.chat.completions.calls[0]["timeout"] == 1.0


def test_explicit_timeout_overrides_request_timeout() -> None:
    provider, fake = _provider()
    fake.chat.completions.result = _groq_response()
    _run(provider, _request(timeout_s=1.0), _model(), timeout_s=3.0)
    assert fake.chat.completions.calls[0]["timeout"] == 3.0


def test_structured_output_is_explicitly_unsupported() -> None:
    provider, fake = _provider()
    request = _request(response_schema={"type": "object"})
    with pytest.raises(InvalidRequestError):
        _run(provider, request, _model())
    assert fake.chat.completions.calls == []


# --- D. response normalization --------------------------------------------------


def test_normal_response_is_normalized() -> None:
    provider, fake = _provider()
    fake.chat.completions.result = _groq_response(text="generated text")
    response = _run(provider, _request(request_id="req-9"), _model("groq-x"))
    assert isinstance(response, LLMResponse)
    assert response.request_id == "req-9"
    assert response.provider == "groq"
    assert response.model == "groq-x"
    assert response.content == "generated text"
    assert response.finish_reason is FinishReason.STOP
    assert response.latency_ms >= 0


def test_usage_is_normalized() -> None:
    provider, fake = _provider()
    fake.chat.completions.result = _groq_response()
    response = _run(provider, _request(), _model())
    assert response.usage is not None
    assert response.usage.input_tokens == 3
    assert response.usage.output_tokens == 5
    assert response.usage.total_tokens == 8


def test_missing_usage_is_none_not_invented() -> None:
    provider, fake = _provider()
    fake.chat.completions.result = _groq_response(with_usage=False)
    response = _run(provider, _request(), _model())
    assert response.usage is None


def test_response_metadata_carries_provenance() -> None:
    provider, fake = _provider()
    fake.chat.completions.result = _groq_response(response_id="chatcmpl-42")
    response = _run(provider, _request(), _model())
    assert response.metadata["response_id"] == "chatcmpl-42"


def test_multiple_choices_use_the_first() -> None:
    provider, fake = _provider()
    result = _groq_response(text="first")
    from groq.types.chat.chat_completion import Choice
    from groq.types.chat.chat_completion_message import ChatCompletionMessage

    result.choices.append(
        Choice(
            index=1,
            finish_reason="stop",
            message=ChatCompletionMessage(role="assistant", content="second"),
        )
    )
    fake.chat.completions.result = result
    response = _run(provider, _request(), _model())
    assert response.content == "first"


@pytest.mark.parametrize(
    ("groq_reason", "expected"),
    [
        ("stop", FinishReason.STOP),
        ("length", FinishReason.LENGTH),
        ("tool_calls", FinishReason.TOOL_CALL),
        ("function_call", FinishReason.TOOL_CALL),
    ],
)
def test_finish_reason_is_normalized(groq_reason: str, expected: FinishReason) -> None:
    provider, fake = _provider()
    fake.chat.completions.result = _groq_response(finish_reason=groq_reason)
    response = _run(provider, _request(), _model())
    assert response.finish_reason is expected


def test_none_content_yields_none_content() -> None:
    provider, fake = _provider()
    fake.chat.completions.result = _groq_response(text=None)
    response = _run(provider, _request(), _model())
    assert response.content is None
    assert response.finish_reason is FinishReason.STOP


def test_response_without_choices_is_malformed() -> None:
    provider, fake = _provider()
    fake.chat.completions.result = _groq_response(with_choices=False)
    with pytest.raises(MalformedResponseError):
        _run(provider, _request(), _model())


# --- E. error normalization -----------------------------------------------------


def _status_error(
    cls: type[groq.APIStatusError],
    code: int,
    message: str = "boom",
    headers: dict[str, str] | None = None,
) -> groq.APIStatusError:
    request = httpx.Request("POST", "https://api.groq.com/openai/v1/chat/completions")
    response = httpx.Response(code, request=request, headers=headers or {})
    return cls(message, response=response, body={"message": message})


@pytest.mark.parametrize(
    ("cls", "code", "expected"),
    [
        (groq.BadRequestError, 400, InvalidRequestError),
        (groq.AuthenticationError, 401, AuthenticationError),
        (groq.PermissionDeniedError, 403, AuthorizationError),
        (groq.NotFoundError, 404, ModelUnavailableError),
        (groq.BadRequestError, 408, LLMTimeoutError),
        (groq.RateLimitError, 429, RateLimitError),
        (groq.UnprocessableEntityError, 422, InvalidRequestError),
        (groq.InternalServerError, 500, ProviderUnavailableError),
        (groq.InternalServerError, 503, ProviderUnavailableError),
    ],
)
def test_api_errors_are_mapped_to_normalized_errors(
    cls: type[groq.APIStatusError], code: int, expected: type[LLMError]
) -> None:
    provider, fake = _provider()
    fake.chat.completions.error = _status_error(cls, code)
    with pytest.raises(expected) as excinfo:
        _run(provider, _request(), _model())
    assert isinstance(excinfo.value, LLMError)
    assert excinfo.value.provider == "groq"
    assert excinfo.value.status_code == code


def test_rate_limit_preserves_retry_after() -> None:
    provider, fake = _provider()
    fake.chat.completions.error = _status_error(
        groq.RateLimitError, 429, message="quota", headers={"retry-after": "12"}
    )
    with pytest.raises(RateLimitError) as excinfo:
        _run(provider, _request(), _model())
    assert excinfo.value.retry_after_s == 12.0


def test_sdk_exception_is_chained_as_cause() -> None:
    provider, fake = _provider()
    original = _status_error(groq.RateLimitError, 429)
    fake.chat.completions.error = original
    with pytest.raises(RateLimitError) as excinfo:
        _run(provider, _request(), _model())
    assert excinfo.value.__cause__ is original
    assert excinfo.value.cause is original


def test_sdk_timeout_becomes_llm_timeout() -> None:
    provider, fake = _provider()
    request = httpx.Request("POST", "https://api.groq.com/openai/v1/chat/completions")
    fake.chat.completions.error = groq.APITimeoutError(request=request)
    with pytest.raises(LLMTimeoutError):
        _run(provider, _request(), _model())


def test_httpx_timeout_becomes_llm_timeout() -> None:
    provider, fake = _provider()
    fake.chat.completions.error = httpx.ReadTimeout("timed out")
    with pytest.raises(LLMTimeoutError):
        _run(provider, _request(), _model())


def test_sdk_connection_error_becomes_network_error() -> None:
    provider, fake = _provider()
    request = httpx.Request("POST", "https://api.groq.com/openai/v1/chat/completions")
    fake.chat.completions.error = groq.APIConnectionError(request=request)
    with pytest.raises(NetworkError):
        _run(provider, _request(), _model())


def test_httpx_transport_error_becomes_network_error() -> None:
    provider, fake = _provider()
    fake.chat.completions.error = httpx.ConnectError("connection refused")
    with pytest.raises(NetworkError):
        _run(provider, _request(), _model())


def test_unclassifiable_failure_becomes_unknown_provider_error() -> None:
    provider, fake = _provider()
    fake.chat.completions.error = RuntimeError("something strange")
    with pytest.raises(UnknownProviderError) as excinfo:
        _run(provider, _request(), _model())
    assert isinstance(excinfo.value, LLMError)
    assert excinfo.value.category is ErrorCategory.UNKNOWN


# --- F. retryability semantics --------------------------------------------------


def test_retryable_categories_are_preserved() -> None:
    provider, fake = _provider()
    fake.chat.completions.error = _status_error(groq.RateLimitError, 429)
    with pytest.raises(RateLimitError) as excinfo:
        _run(provider, _request(), _model())
    assert excinfo.value.retryable is True


def test_non_retryable_categories_are_preserved() -> None:
    provider, fake = _provider()
    fake.chat.completions.error = _status_error(groq.AuthenticationError, 401)
    with pytest.raises(AuthenticationError) as excinfo:
        _run(provider, _request(), _model())
    assert excinfo.value.retryable is False


# --- G. secret safety -----------------------------------------------------------


def test_api_key_never_appears_in_repr() -> None:
    provider = GroqProvider(api_key=FAKE_KEY)
    assert FAKE_KEY not in repr(provider)


def test_api_key_never_appears_in_errors_or_responses() -> None:
    provider, fake = _provider()
    fake.chat.completions.error = _status_error(
        groq.AuthenticationError, 401, message="bad credentials"
    )
    with pytest.raises(AuthenticationError) as excinfo:
        _run(provider, _request(), _model())
    error = excinfo.value
    assert FAKE_KEY not in str(error)
    assert FAKE_KEY not in error.safe_message
    assert FAKE_KEY not in repr(error.metadata)

    fake.chat.completions.error = None
    fake.chat.completions.result = _groq_response()
    response = _run(provider, _request(), _model())
    assert FAKE_KEY not in response.model_dump_json()


# --- H. boundary & no hidden behavior -------------------------------------------


def test_sdk_response_object_does_not_escape() -> None:
    provider, fake = _provider()
    fake.chat.completions.result = _groq_response()
    response = _run(provider, _request(), _model())
    assert type(response) is LLMResponse
    assert not hasattr(response, "choices")


def test_no_retry_and_no_fallback_occur() -> None:
    provider, fake = _provider()
    fake.chat.completions.result = _groq_response()
    _run(provider, _request(), _model("groq-only"))
    assert len(fake.chat.completions.calls) == 1
    assert fake.chat.completions.calls[0]["model"] == "groq-only"


def test_requested_model_is_never_substituted() -> None:
    provider, fake = _provider()
    fake.chat.completions.error = _status_error(groq.NotFoundError, 404)
    with pytest.raises(ModelUnavailableError):
        _run(provider, _request(), _model("groq-missing"))
    assert fake.chat.completions.calls[0]["model"] == "groq-missing"
    assert len(fake.chat.completions.calls) == 1


def test_model_from_another_provider_is_rejected() -> None:
    provider, fake = _provider()
    with pytest.raises(InvalidRequestError):
        _run(provider, _request(), _model(provider="some-other-provider"))
    assert fake.chat.completions.calls == []


# --- I. lifecycle ---------------------------------------------------------------


def test_aclose_closes_the_client() -> None:
    provider, fake = _provider()
    asyncio.run(provider.aclose())
    assert fake.closed is True


def test_aclose_is_idempotent() -> None:
    provider, fake = _provider()
    asyncio.run(provider.aclose())
    asyncio.run(provider.aclose())
    assert fake.closed is True
