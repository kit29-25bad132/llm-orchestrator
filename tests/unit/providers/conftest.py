"""Shared fixtures and a test-only fake provider.

``FakeProvider`` lives here, not in the package: it exists to prove the
abstraction is implementable without any SDK. It is deliberately minimal and is
not production code.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import pytest

from llm_orchestrator.contracts import (
    FinishReason,
    LLMRequest,
    LLMResponse,
    ModelConfig,
)
from llm_orchestrator.errors import LLMError


@dataclass
class FakeProvider:
    """A deterministic, offline :class:`LLMProvider` implementation."""

    provider_id: str = "fake"
    latency_ms: float = 1.0
    content_prefix: str = "echo"
    error: LLMError | None = None
    closed: bool = False
    calls: list[tuple[str, str]] = field(default_factory=list)

    async def generate(
        self,
        request: LLMRequest,
        model: ModelConfig,
        *,
        timeout_s: float | None = None,
    ) -> LLMResponse:
        self.calls.append((request.request_id, model.model))
        if self.error is not None:
            raise self.error
        last_content = request.messages[-1].content
        return LLMResponse(
            request_id=request.request_id,
            provider=self.provider_id,
            model=model.model,
            finish_reason=FinishReason.STOP,
            latency_ms=self.latency_ms,
            content=f"{self.content_prefix}:{last_content}",
        )

    async def aclose(self) -> None:
        self.closed = True


@pytest.fixture
def fake_provider() -> FakeProvider:
    return FakeProvider()


@pytest.fixture
def make_provider() -> Callable[..., FakeProvider]:
    def _make(**kwargs: object) -> FakeProvider:
        return FakeProvider(**kwargs)

    return _make
