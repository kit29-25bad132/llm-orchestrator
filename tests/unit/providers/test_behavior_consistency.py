"""Cross-provider behavior consistency between the Gemini and Groq adapters.

``test_cross_provider_consistency.py`` proves equivalent *failures* classify
identically. This matrix proves the success side and the request side converge
too: equivalent SDK successes normalize to the same generic response contract,
equivalent requests preserve the same normalized fields, and unsupported
capabilities are explicitly rejected by both adapters. Provider-specific
suites keep the unique-behavior tests; only shared guarantees live here.
"""

from __future__ import annotations

from typing import Any

import pytest
from google.genai import types as genai_types

from llm_orchestrator.contracts import (
    FinishReason,
    LLMRequest,
    Message,
    MessageRole,
    ModelConfig,
)
from llm_orchestrator.errors import ErrorCategory, InvalidRequestError
from llm_orchestrator.providers import LLMProvider
from llm_orchestrator.providers.gemini import GeminiProvider
from llm_orchestrator.providers.groq import GroqProvider
from support.fakes import FakeGeminiSdkClient, FakeGroqSdkClient, run_sync

FAKE_GEMINI_KEY = "secret-test-key-abcdef123456"
FAKE_GROQ_KEY = "gsk_test-key-abcdef1234567890"


def _gemini_success(text: str) -> Any:
    content = genai_types.Content(role="model", parts=[genai_types.Part(text=text)])
    return genai_types.GenerateContentResponse(
        model_version="test-v1",
        candidates=[genai_types.Candidate(content=content, finish_reason="STOP")],
        usage_metadata=genai_types.GenerateContentResponseUsageMetadata(
            prompt_token_count=3, candidates_token_count=5, total_token_count=8
        ),
        response_id="resp-1",
    )


def _groq_success(text: str) -> Any:
    from groq.types.chat import ChatCompletion
    from groq.types.chat.chat_completion import Choice
    from groq.types.chat.chat_completion_message import ChatCompletionMessage
    from groq.types.completion_usage import CompletionUsage

    return ChatCompletion(
        id="chatcmpl-1",
        created=1700000000,
        model="groq-test-model",
        object="chat.completion",
        choices=[
            Choice(
                index=0,
                finish_reason="stop",  # type: ignore[arg-type]
                message=ChatCompletionMessage(role="assistant", content=text),
            )
        ],
        usage=CompletionUsage(prompt_tokens=3, completion_tokens=5, total_tokens=8),
    )


def _request() -> LLMRequest:
    return LLMRequest(
        messages=[
            Message(role=MessageRole.SYSTEM, content="be terse"),
            Message(role=MessageRole.USER, content="hello"),
        ],
        request_id="req-1",
    )


@pytest.mark.parametrize("provider_id", ["gemini", "groq"])
def test_providers_satisfy_the_shared_protocol(provider_id: str) -> None:
    provider: LLMProvider
    if provider_id == "gemini":
        provider = GeminiProvider(api_key=FAKE_GEMINI_KEY, client=FakeGeminiSdkClient([None]))
    else:
        provider = GroqProvider(api_key=FAKE_GROQ_KEY, client=FakeGroqSdkClient([None]))
    assert isinstance(provider, LLMProvider)
    assert provider.provider_id == provider_id


def test_equivalent_successes_normalize_to_the_same_contract() -> None:
    """Same text and token counts → same generic fields, differing only by origin."""
    gemini = GeminiProvider(
        api_key=FAKE_GEMINI_KEY, client=FakeGeminiSdkClient([_gemini_success("hello")])
    )
    groq = GroqProvider(api_key=FAKE_GROQ_KEY, client=FakeGroqSdkClient([_groq_success("hello")]))
    gemini_model = ModelConfig(provider="gemini", model="gemini-test-model")
    groq_model = ModelConfig(provider="groq", model="groq-test-model")

    gemini_response = run_sync(gemini.generate(_request(), gemini_model))
    groq_response = run_sync(groq.generate(_request(), groq_model))

    for response in (gemini_response, groq_response):
        assert response.request_id == "req-1"
        assert response.content == "hello"
        assert response.finish_reason is FinishReason.STOP
        assert response.usage is not None
        assert response.usage.total_tokens == 8
    assert gemini_response.provider == "gemini"
    assert groq_response.provider == "groq"
    assert gemini_response.model == "gemini-test-model"
    assert groq_response.model == "groq-test-model"


def test_equivalent_requests_preserve_content_and_order() -> None:
    """Both adapters carry the same user text without invention or reordering."""
    gemini_client = FakeGeminiSdkClient([_gemini_success("hi")])
    groq_client = FakeGroqSdkClient([_groq_success("hi")])
    gemini = GeminiProvider(api_key=FAKE_GEMINI_KEY, client=gemini_client)
    groq = GroqProvider(api_key=FAKE_GROQ_KEY, client=groq_client)

    run_sync(gemini.generate(_request(), ModelConfig(provider="gemini", model="m")))
    run_sync(groq.generate(_request(), ModelConfig(provider="groq", model="m")))

    groq_messages = groq_client.calls[0]["messages"]
    assert [(m["role"], m["content"]) for m in groq_messages] == [
        ("system", "be terse"),
        ("user", "hello"),
    ]
    gemini_contents = gemini_client.calls[0]["contents"]
    assert [part.text for content in gemini_contents for part in content.parts] == ["hello"]
    assert gemini_client.calls[0]["config"].system_instruction == "be terse"


@pytest.mark.parametrize("provider_id", ["gemini", "groq"])
def test_response_schema_rejected_by_both_adapters(provider_id: str) -> None:
    """Native structured modes stay explicitly unsupported on both vendors."""
    if provider_id == "gemini":
        provider: LLMProvider = GeminiProvider(
            api_key=FAKE_GEMINI_KEY, client=FakeGeminiSdkClient([_gemini_success("hi")])
        )
        model = ModelConfig(provider="gemini", model="m")
    else:
        provider = GroqProvider(
            api_key=FAKE_GROQ_KEY, client=FakeGroqSdkClient([_groq_success("hi")])
        )
        model = ModelConfig(provider="groq", model="m")
    schemad = _request().model_copy(update={"response_schema": {"type": "object"}})
    with pytest.raises(InvalidRequestError) as exc_info:
        run_sync(provider.generate(schemad, model))
    assert exc_info.value.category is ErrorCategory.INVALID_REQUEST


@pytest.mark.parametrize("provider_id", ["gemini", "groq"])
def test_wrong_vendor_model_rejected_by_both_adapters(provider_id: str) -> None:
    """Neither adapter silently serves another vendor's model id."""
    other = "groq" if provider_id == "gemini" else "gemini"
    if provider_id == "gemini":
        provider = GeminiProvider(
            api_key=FAKE_GEMINI_KEY, client=FakeGeminiSdkClient([_gemini_success("hi")])
        )
    else:
        provider = GroqProvider(
            api_key=FAKE_GROQ_KEY, client=FakeGroqSdkClient([_groq_success("hi")])
        )
    with pytest.raises(InvalidRequestError):
        run_sync(provider.generate(_request(), ModelConfig(provider=other, model="m")))


@pytest.mark.parametrize("provider_id", ["gemini", "groq"])
def test_tool_messages_rejected_by_both_adapters(provider_id: str) -> None:
    """Tool calling stays explicitly unsupported on both vendors."""
    if provider_id == "gemini":
        provider = GeminiProvider(
            api_key=FAKE_GEMINI_KEY, client=FakeGeminiSdkClient([_gemini_success("hi")])
        )
        model = ModelConfig(provider="gemini", model="m")
    else:
        provider = GroqProvider(
            api_key=FAKE_GROQ_KEY, client=FakeGroqSdkClient([_groq_success("hi")])
        )
        model = ModelConfig(provider="groq", model="m")
    tool_request = LLMRequest(
        messages=[Message(role=MessageRole.TOOL, content="result")],
        request_id="req-1",
    )
    with pytest.raises(InvalidRequestError):
        run_sync(provider.generate(tool_request, model))
