"""Opt-in live Gemini integration test.

This test talks to the real Gemini API and is **never** part of the default
suite. It runs only when explicitly enabled:

    RUN_GEMINI_INTEGRATION=1 GEMINI_API_KEY=... GEMINI_TEST_MODEL=... pytest -m live

No credential is printed, hard-coded, or required by the default run.
"""

from __future__ import annotations

import asyncio
import os

import pytest

from llm_orchestrator.contracts import (
    FinishReason,
    LLMRequest,
    LLMResponse,
    Message,
    MessageRole,
    ModelConfig,
)
from llm_orchestrator.providers.gemini import GeminiProvider

pytestmark = pytest.mark.live


def _live_config() -> tuple[str, str] | None:
    if os.environ.get("RUN_GEMINI_INTEGRATION") != "1":
        return None
    api_key = os.environ.get("GEMINI_API_KEY")
    model = os.environ.get("GEMINI_TEST_MODEL")
    if not api_key or not model:
        return None
    return api_key, model


@pytest.mark.skipif(_live_config() is None, reason="live Gemini integration not enabled")
def test_live_gemini_round_trip() -> None:
    config = _live_config()
    assert config is not None  # narrowed by the skipif above
    api_key, model_name = config

    provider = GeminiProvider(api_key=api_key)
    request = LLMRequest(
        messages=[Message(role=MessageRole.USER, content="Reply with the single word: ok")],
        max_tokens=16,
    )
    model = ModelConfig(provider="gemini", model=model_name)

    response = asyncio.run(provider.generate(request, model))

    assert isinstance(response, LLMResponse)
    assert response.provider == "gemini"
    assert response.model == model_name
    assert response.finish_reason in set(FinishReason)
    assert response.latency_ms >= 0
    assert response.content

    asyncio.run(provider.aclose())
