"""Offline tests for the Gemini provider adapter.

These exercise the real adapter code (translation, normalization, error mapping)
against a fake SDK *client*, so no network access or API key is required. The
real ``google.genai`` types are used for inputs and outputs so the tests cover
the actual translation and normalization logic.
"""

from __future__ import annotations

import asyncio
from typing import Any

import httpx
import pytest
from google.genai import errors as genai_errors
from google.genai import types as genai_types

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
from llm_orchestrator.providers.gemini import PROVIDER_ID, GeminiProvider

FAKE_KEY = "secret-test-key-abcdef123456"


class _FakeAsyncModels:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []
        self.result: genai_types.GenerateContentResponse | None = None
        self.error: Exception | None = None

    async def generate_content(
        self,
        *,
        model: str,
        contents: Any,
        config: Any = None,
    ) -> genai_types.GenerateContentResponse:
        self.calls.append({"model": model, "contents": contents, "config": config})
        if self.error is not None:
            raise self.error
        assert self.result is not None
        return self.result


class _FakeAio:
    def __init__(self) -> None:
        self.models = _FakeAsyncModels()
        self.closed = False

    async def aclose(self) -> None:
        self.closed = True


class _FakeClient:
    def __init__(self) -> None:
        self.aio = _FakeAio()


def _provider() -> tuple[GeminiProvider, _FakeClient]:
    fake = _FakeClient()
    return GeminiProvider(api_key=FAKE_KEY, client=fake), fake


def _request(
    *messages: Message,
    **kwargs: Any,
) -> LLMRequest:
    return LLMRequest(
        messages=list(messages) or [Message(role=MessageRole.USER, content="hi")],
        **kwargs,
    )


def _model(model: str = "gemini-test-model", provider: str = PROVIDER_ID) -> ModelConfig:
    return ModelConfig(provider=provider, model=model)


def _gemini_response(
    *,
    text: str | None = "hello",
    finish_reason: str = "STOP",
    with_candidates: bool = True,
    with_usage: bool = True,
    block_reason: str | None = None,
    response_id: str | None = "resp-1",
) -> genai_types.GenerateContentResponse:
    kwargs: dict[str, Any] = {"model_version": "gemini-test-v1"}
    if with_candidates:
        content = (
            genai_types.Content(role="model", parts=[genai_types.Part(text=text)])
            if text is not None
            else None
        )
        kwargs["candidates"] = [genai_types.Candidate(content=content, finish_reason=finish_reason)]
    if with_usage:
        kwargs["usage_metadata"] = genai_types.GenerateContentResponseUsageMetadata(
            prompt_token_count=3, candidates_token_count=5, total_token_count=8
        )
    if response_id is not None:
        kwargs["response_id"] = response_id
    if block_reason is not None:
        kwargs["prompt_feedback"] = genai_types.GenerateContentResponsePromptFeedback(
            block_reason=block_reason
        )
    return genai_types.GenerateContentResponse(**kwargs)


def _run(
    provider: GeminiProvider, request: LLMRequest, model: ModelConfig, **kwargs: Any
) -> LLMResponse:
    return asyncio.run(provider.generate(request, model, **kwargs))


# --- A. construction & identity -------------------------------------------------


def test_provider_satisfies_the_protocol() -> None:
    provider, _ = _provider()
    assert isinstance(provider, LLMProvider)
    assert provider.provider_id == "gemini"


def test_missing_api_key_is_a_configuration_error() -> None:
    with pytest.raises(ConfigurationError):
        GeminiProvider()


def test_blank_api_key_is_a_configuration_error() -> None:
    with pytest.raises(ConfigurationError):
        GeminiProvider("   ")


def test_injected_client_does_not_require_an_api_key() -> None:
    provider = GeminiProvider(client=_FakeClient())
    assert provider.provider_id == "gemini"


# --- B. message translation -----------------------------------------------------


def test_user_message_becomes_a_user_content() -> None:
    provider, fake = _provider()
    fake.aio.models.result = _gemini_response()
    _run(provider, _request(Message(role=MessageRole.USER, content="hello")), _model())
    contents = fake.aio.models.calls[0]["contents"]
    assert len(contents) == 1
    assert contents[0].role == "user"
    assert contents[0].parts[0].text == "hello"


def test_assistant_message_maps_to_model_role() -> None:
    provider, fake = _provider()
    fake.aio.models.result = _gemini_response()
    request = _request(
        Message(role=MessageRole.USER, content="q"),
        Message(role=MessageRole.ASSISTANT, content="a"),
        Message(role=MessageRole.USER, content="q2"),
    )
    _run(provider, request, _model())
    roles = [c.role for c in fake.aio.models.calls[0]["contents"]]
    assert roles == ["user", "model", "user"]


def test_system_message_becomes_system_instruction() -> None:
    provider, fake = _provider()
    fake.aio.models.result = _gemini_response()
    request = _request(
        Message(role=MessageRole.SYSTEM, content="be terse"),
        Message(role=MessageRole.USER, content="hi"),
    )
    _run(provider, request, _model())
    call = fake.aio.models.calls[0]
    assert call["config"].system_instruction == "be terse"
    assert [c.role for c in call["contents"]] == ["user"]


def test_multiple_system_messages_are_joined() -> None:
    provider, fake = _provider()
    fake.aio.models.result = _gemini_response()
    request = _request(
        Message(role=MessageRole.SYSTEM, content="one"),
        Message(role=MessageRole.SYSTEM, content="two"),
        Message(role=MessageRole.USER, content="hi"),
    )
    _run(provider, request, _model())
    assert fake.aio.models.calls[0]["config"].system_instruction == "one\n\ntwo"


def test_consecutive_same_role_messages_are_merged() -> None:
    provider, fake = _provider()
    fake.aio.models.result = _gemini_response()
    request = _request(
        Message(role=MessageRole.USER, content="first"),
        Message(role=MessageRole.USER, content="second"),
    )
    _run(provider, request, _model())
    contents = fake.aio.models.calls[0]["contents"]
    assert len(contents) == 1
    assert [p.text for p in contents[0].parts] == ["first", "second"]


def test_tool_message_role_is_rejected_explicitly() -> None:
    provider, fake = _provider()
    request = _request(Message(role=MessageRole.TOOL, content="tool output"))
    with pytest.raises(InvalidRequestError):
        _run(provider, request, _model())
    assert fake.aio.models.calls == []


def test_system_only_request_is_rejected() -> None:
    provider, _ = _provider()
    request = _request(Message(role=MessageRole.SYSTEM, content="only system"))
    with pytest.raises(InvalidRequestError):
        _run(provider, request, _model())


# --- C. generation parameters & timeout -----------------------------------------


def test_generation_parameters_are_mapped() -> None:
    provider, fake = _provider()
    fake.aio.models.result = _gemini_response()
    request = _request(temperature=0.3, max_tokens=256, stop=("END",))
    _run(provider, request, _model())
    config = fake.aio.models.calls[0]["config"]
    assert config.temperature == 0.3
    assert config.max_output_tokens == 256
    assert list(config.stop_sequences) == ["END"]


def test_absent_generation_parameters_are_not_forwarded() -> None:
    provider, fake = _provider()
    fake.aio.models.result = _gemini_response()
    _run(provider, _request(), _model())
    config = fake.aio.models.calls[0]["config"]
    assert config.temperature is None
    assert config.max_output_tokens is None
    assert config.stop_sequences is None


def test_explicit_timeout_is_mapped_to_http_options() -> None:
    provider, fake = _provider()
    fake.aio.models.result = _gemini_response()
    _run(provider, _request(), _model(), timeout_s=2.5)
    assert fake.aio.models.calls[0]["config"].http_options.timeout == 2500


def test_request_level_timeout_is_used_when_no_override() -> None:
    provider, fake = _provider()
    fake.aio.models.result = _gemini_response()
    _run(provider, _request(timeout_s=1.0), _model())
    assert fake.aio.models.calls[0]["config"].http_options.timeout == 1000


def test_explicit_timeout_overrides_request_timeout() -> None:
    provider, fake = _provider()
    fake.aio.models.result = _gemini_response()
    _run(provider, _request(timeout_s=1.0), _model(), timeout_s=3.0)
    assert fake.aio.models.calls[0]["config"].http_options.timeout == 3000


def test_structured_output_is_explicitly_unsupported() -> None:
    provider, fake = _provider()
    request = _request(response_schema={"type": "object"})
    with pytest.raises(InvalidRequestError):
        _run(provider, request, _model())
    assert fake.aio.models.calls == []


# --- D. response normalization --------------------------------------------------


def test_normal_response_is_normalized() -> None:
    provider, fake = _provider()
    fake.aio.models.result = _gemini_response(text="generated text")
    response = _run(provider, _request(request_id="req-9"), _model("gemini-x"))
    assert isinstance(response, LLMResponse)
    assert response.request_id == "req-9"
    assert response.provider == "gemini"
    assert response.model == "gemini-x"
    assert response.content == "generated text"
    assert response.finish_reason is FinishReason.STOP
    assert response.latency_ms >= 0


def test_usage_metadata_is_normalized() -> None:
    provider, fake = _provider()
    fake.aio.models.result = _gemini_response()
    response = _run(provider, _request(), _model())
    assert response.usage is not None
    assert response.usage.input_tokens == 3
    assert response.usage.output_tokens == 5
    assert response.usage.total_tokens == 8


def test_missing_usage_is_none_not_invented() -> None:
    provider, fake = _provider()
    fake.aio.models.result = _gemini_response(with_usage=False)
    response = _run(provider, _request(), _model())
    assert response.usage is None


def test_response_metadata_carries_provenance() -> None:
    provider, fake = _provider()
    fake.aio.models.result = _gemini_response(response_id="resp-42")
    response = _run(provider, _request(), _model())
    assert response.metadata["response_id"] == "resp-42"
    assert response.metadata["model_version"] == "gemini-test-v1"


def test_multiple_candidates_use_the_first() -> None:
    provider, fake = _provider()
    response_obj = _gemini_response(text="first")
    response_obj.candidates.append(
        genai_types.Candidate(
            content=genai_types.Content(role="model", parts=[genai_types.Part(text="second")]),
            finish_reason="STOP",
        )
    )
    fake.aio.models.result = response_obj
    response = _run(provider, _request(), _model())
    assert response.content == "first"


def test_multiple_text_parts_are_concatenated() -> None:
    provider, fake = _provider()
    fake.aio.models.result = genai_types.GenerateContentResponse(
        candidates=[
            genai_types.Candidate(
                content=genai_types.Content(
                    role="model",
                    parts=[genai_types.Part(text="a"), genai_types.Part(text="b")],
                ),
                finish_reason="STOP",
            )
        ]
    )
    response = _run(provider, _request(), _model())
    assert response.content == "ab"


@pytest.mark.parametrize(
    ("gemini_reason", "expected"),
    [
        ("STOP", FinishReason.STOP),
        ("MAX_TOKENS", FinishReason.LENGTH),
        ("SAFETY", FinishReason.CONTENT_FILTER),
        ("RECITATION", FinishReason.CONTENT_FILTER),
        ("PROHIBITED_CONTENT", FinishReason.CONTENT_FILTER),
        ("MALFORMED_FUNCTION_CALL", FinishReason.ERROR),
        ("FINISH_REASON_UNSPECIFIED", FinishReason.UNKNOWN),
        ("OTHER", FinishReason.UNKNOWN),
    ],
)
def test_finish_reason_is_normalized(gemini_reason: str, expected: FinishReason) -> None:
    provider, fake = _provider()
    fake.aio.models.result = _gemini_response(finish_reason=gemini_reason)
    response = _run(provider, _request(), _model())
    assert response.finish_reason is expected


def test_empty_candidate_content_yields_none_content() -> None:
    provider, fake = _provider()
    fake.aio.models.result = _gemini_response(text=None)
    response = _run(provider, _request(), _model())
    assert response.content is None
    assert response.finish_reason is FinishReason.STOP


def test_blocked_response_is_normalized_not_raised() -> None:
    provider, fake = _provider()
    fake.aio.models.result = _gemini_response(
        with_candidates=False, block_reason="SAFETY", with_usage=False
    )
    response = _run(provider, _request(), _model())
    assert response.finish_reason is FinishReason.CONTENT_FILTER
    assert response.content is None
    assert response.metadata["block_reason"] == "SAFETY"


def test_response_without_candidates_and_without_block_is_malformed() -> None:
    provider, fake = _provider()
    fake.aio.models.result = _gemini_response(with_candidates=False)
    with pytest.raises(MalformedResponseError):
        _run(provider, _request(), _model())


# --- E. error normalization -----------------------------------------------------


def _api_error(
    cls: type[genai_errors.APIError], code: int, message: str = "boom"
) -> genai_errors.APIError:
    return cls(code, {"error": {"code": code, "message": message, "status": "ERR"}})


@pytest.mark.parametrize(
    ("cls", "code", "expected"),
    [
        (genai_errors.ClientError, 400, InvalidRequestError),
        (genai_errors.ClientError, 401, AuthenticationError),
        (genai_errors.ClientError, 403, AuthorizationError),
        (genai_errors.ClientError, 404, ModelUnavailableError),
        (genai_errors.ClientError, 429, RateLimitError),
        (genai_errors.ClientError, 422, InvalidRequestError),
        (genai_errors.ServerError, 500, ProviderUnavailableError),
        (genai_errors.ServerError, 503, ProviderUnavailableError),
    ],
)
def test_api_errors_are_mapped_to_normalized_errors(
    cls: type[genai_errors.APIError], code: int, expected: type[LLMError]
) -> None:
    provider, fake = _provider()
    fake.aio.models.error = _api_error(cls, code)
    with pytest.raises(expected) as excinfo:
        _run(provider, _request(), _model())
    assert isinstance(excinfo.value, LLMError)
    assert excinfo.value.provider == "gemini"
    assert excinfo.value.status_code == code


def test_rate_limit_preserves_retry_after() -> None:
    provider, fake = _provider()
    response = httpx.Response(
        429, headers={"retry-after": "12"}, request=httpx.Request("POST", "https://x")
    )
    fake.aio.models.error = genai_errors.ClientError(
        429, {"error": {"code": 429, "message": "quota"}}, response
    )
    with pytest.raises(RateLimitError) as excinfo:
        _run(provider, _request(), _model())
    assert excinfo.value.retry_after_s == 12.0


def test_sdk_exception_is_chained_as_cause() -> None:
    provider, fake = _provider()
    original = _api_error(genai_errors.ClientError, 429)
    fake.aio.models.error = original
    with pytest.raises(RateLimitError) as excinfo:
        _run(provider, _request(), _model())
    assert excinfo.value.__cause__ is original
    assert excinfo.value.cause is original


def test_httpx_timeout_becomes_llm_timeout() -> None:
    provider, fake = _provider()
    fake.aio.models.error = httpx.ReadTimeout("timed out")
    with pytest.raises(LLMTimeoutError):
        _run(provider, _request(), _model())


def test_httpx_transport_error_becomes_network_error() -> None:
    provider, fake = _provider()
    fake.aio.models.error = httpx.ConnectError("connection refused")
    with pytest.raises(NetworkError):
        _run(provider, _request(), _model())


def test_unclassifiable_failure_becomes_unknown_provider_error() -> None:
    provider, fake = _provider()
    fake.aio.models.error = RuntimeError("something strange")
    with pytest.raises(UnknownProviderError) as excinfo:
        _run(provider, _request(), _model())
    assert isinstance(excinfo.value, LLMError)
    assert excinfo.value.category is ErrorCategory.UNKNOWN


# --- F. retryability semantics --------------------------------------------------


def test_retryable_categories_are_preserved() -> None:
    provider, fake = _provider()
    fake.aio.models.error = _api_error(genai_errors.ClientError, 429)
    with pytest.raises(RateLimitError) as excinfo:
        _run(provider, _request(), _model())
    assert excinfo.value.retryable is True


def test_non_retryable_categories_are_preserved() -> None:
    provider, fake = _provider()
    fake.aio.models.error = _api_error(genai_errors.ClientError, 401)
    with pytest.raises(AuthenticationError) as excinfo:
        _run(provider, _request(), _model())
    assert excinfo.value.retryable is False


# --- G. secret safety -----------------------------------------------------------


def test_api_key_never_appears_in_repr() -> None:
    provider = GeminiProvider(api_key=FAKE_KEY)
    assert FAKE_KEY not in repr(provider)


def test_api_key_never_appears_in_errors_or_responses() -> None:
    provider, fake = _provider()
    fake.aio.models.error = _api_error(genai_errors.ClientError, 401, message="bad credentials")
    with pytest.raises(AuthenticationError) as excinfo:
        _run(provider, _request(), _model())
    error = excinfo.value
    assert FAKE_KEY not in str(error)
    assert FAKE_KEY not in error.safe_message
    assert FAKE_KEY not in repr(error.metadata)

    fake.aio.models.error = None
    fake.aio.models.result = _gemini_response()
    response = _run(provider, _request(), _model())
    assert FAKE_KEY not in response.model_dump_json()


# --- H. boundary & no hidden behavior -------------------------------------------


def test_sdk_response_object_does_not_escape() -> None:
    provider, fake = _provider()
    fake.aio.models.result = _gemini_response()
    response = _run(provider, _request(), _model())
    assert type(response) is LLMResponse
    assert not hasattr(response, "candidates")


def test_no_retry_and_no_fallback_occur() -> None:
    provider, fake = _provider()
    fake.aio.models.result = _gemini_response()
    _run(provider, _request(), _model("gemini-only"))
    assert len(fake.aio.models.calls) == 1
    assert fake.aio.models.calls[0]["model"] == "gemini-only"


def test_requested_model_is_never_substituted() -> None:
    provider, fake = _provider()
    fake.aio.models.error = _api_error(genai_errors.ClientError, 404)
    with pytest.raises(ModelUnavailableError):
        _run(provider, _request(), _model("gemini-missing"))
    assert fake.aio.models.calls[0]["model"] == "gemini-missing"
    assert len(fake.aio.models.calls) == 1


def test_model_from_another_provider_is_rejected() -> None:
    provider, fake = _provider()
    with pytest.raises(InvalidRequestError):
        _run(provider, _request(), _model(provider="some-other-provider"))
    assert fake.aio.models.calls == []


# --- I. lifecycle ---------------------------------------------------------------


def test_aclose_closes_the_client() -> None:
    provider, fake = _provider()
    asyncio.run(provider.aclose())
    assert fake.aio.closed is True


def test_aclose_is_idempotent() -> None:
    provider, fake = _provider()
    asyncio.run(provider.aclose())
    asyncio.run(provider.aclose())
    assert fake.aio.closed is True
