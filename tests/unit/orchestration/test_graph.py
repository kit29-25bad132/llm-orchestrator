"""Graph-level tests: compile once, invoke repeatedly, stay isolated."""

from __future__ import annotations

import asyncio
import random
from typing import Any

import pytest

from llm_orchestrator.contracts import (
    FinishReason,
    JitterStrategy,
    LLMRequest,
    LLMResponse,
    Message,
    MessageRole,
    ModelConfig,
    OrchestrationResult,
    RetryPolicy,
)
from llm_orchestrator.errors import ErrorCategory, NetworkError, RateLimitError
from llm_orchestrator.orchestration import Orchestrator
from llm_orchestrator.providers import ProviderRegistry
from llm_orchestrator.reliability import RetryEngine


class ScriptedProvider:
    """A fake ``LLMProvider`` replaying scripted outcomes (responses/errors)."""

    def __init__(self, provider_id: str, outcomes: list[LLMResponse | BaseException]) -> None:
        self.provider_id = provider_id
        self._outcomes = list(outcomes)
        self.calls: list[dict[str, Any]] = []

    async def generate(
        self,
        request: LLMRequest,
        model: ModelConfig,
        *,
        timeout_s: float | None = None,
    ) -> LLMResponse:
        self.calls.append({"request": request, "model": model, "timeout_s": timeout_s})
        outcome = self._outcomes.pop(0) if len(self._outcomes) > 1 else self._outcomes[0]
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome

    async def aclose(self) -> None:
        pass


def _success(
    content: str = "ok", *, provider: str = "fake", request_id: str = "req-1"
) -> LLMResponse:
    return LLMResponse(
        request_id=request_id,
        provider=provider,
        model="m",
        finish_reason=FinishReason.STOP,
        latency_ms=1.0,
        content=content,
    )


def _request(content: str = "hi", *, request_id: str = "req-1") -> LLMRequest:
    return LLMRequest(
        messages=[Message(role=MessageRole.USER, content=content)], request_id=request_id
    )


def _model(provider: str = "fake") -> ModelConfig:
    return ModelConfig(provider=provider, model="m")


def _orchestrator(provider: ScriptedProvider, policy: RetryPolicy | None = None) -> Orchestrator:
    registry = ProviderRegistry()
    registry.register(provider)

    async def instant_sleep(_: float) -> None:
        return None

    engine = RetryEngine(
        policy or RetryPolicy(max_attempts=1, jitter=JitterStrategy.NONE),
        sleep=instant_sleep,
        rng=random.Random(0),
    )
    return Orchestrator(registry, engine)


def test_successful_run_returns_the_response() -> None:
    orchestrator = _orchestrator(ScriptedProvider("fake", [_success("hello")]))
    result = asyncio.run(orchestrator.run(_request(request_id="r-1"), _model()))
    assert isinstance(result, OrchestrationResult)
    assert result.succeeded is True
    assert result.response is not None
    assert result.response.content == "hello"
    assert result.request_id == "r-1"
    assert result.total_latency_ms is not None
    assert result.total_latency_ms >= 0


def test_failed_run_returns_a_normalized_failure_result() -> None:
    orchestrator = _orchestrator(ScriptedProvider("fake", [RateLimitError("slow")]))
    result = asyncio.run(orchestrator.run(_request(request_id="r-2"), _model()))
    assert result.succeeded is False
    assert result.response is None
    assert result.error is not None
    assert result.error.category is ErrorCategory.RATE_LIMIT
    assert result.request_id == "r-2"


def test_graph_delegates_retries_to_the_engine() -> None:
    provider = ScriptedProvider("fake", [NetworkError("down"), _success("recovered")])
    orchestrator = _orchestrator(provider, RetryPolicy(max_attempts=3, jitter=JitterStrategy.NONE))
    result = asyncio.run(orchestrator.run(_request(), _model()))
    assert result.succeeded is True
    assert result.response is not None
    assert result.response.content == "recovered"
    assert len(provider.calls) == 2
    assert result.attempts[0].attempt == 2


def test_exhaustion_through_the_graph_keeps_normalized_error() -> None:
    orchestrator = _orchestrator(
        ScriptedProvider("fake", [NetworkError("down")]),
        RetryPolicy(max_attempts=2, jitter=JitterStrategy.NONE),
    )
    result = asyncio.run(orchestrator.run(_request(), _model()))
    assert result.succeeded is False
    assert result.error is not None
    assert result.error.category is ErrorCategory.NETWORK
    assert result.error.attempt == 2


def test_unexpected_exceptions_escape_the_graph() -> None:
    orchestrator = _orchestrator(ScriptedProvider("fake", [ValueError("bug")]))
    with pytest.raises(ValueError, match="bug"):
        asyncio.run(orchestrator.run(_request(), _model()))


def test_sequential_runs_do_not_leak_state() -> None:
    first = ScriptedProvider("fake", [_success("AAA", request_id="a")])
    orchestrator = _orchestrator(first)
    result_a = asyncio.run(orchestrator.run(_request("a", request_id="a"), _model()))

    second = ScriptedProvider("fake", [NetworkError("down")])
    registry = ProviderRegistry()
    registry.register(second)
    orchestrator = Orchestrator(registry, orchestrator.retry_engine)
    result_b = asyncio.run(orchestrator.run(_request("b", request_id="b"), _model()))

    assert result_a.succeeded is True
    assert result_a.response is not None
    assert result_a.response.content == "AAA"
    assert result_b.succeeded is False
    assert result_b.request_id == "b"
    assert result_b.response is None
    assert result_b.error is not None
    assert "AAA" not in result_b.error.message


def test_concurrent_runs_stay_isolated() -> None:
    async def main() -> list[OrchestrationResult]:
        providers = [
            ScriptedProvider(f"p{i}", [_success(f"content-{i}", request_id=f"r-{i}")])
            for i in range(3)
        ]
        orchestrators = [_orchestrator(provider) for provider in providers]
        return await asyncio.gather(
            *(
                orchestrator.run(_request(f"q-{i}", request_id=f"r-{i}"), _model(f"p{i}"))
                for i, orchestrator in enumerate(orchestrators)
            )
        )

    results = asyncio.run(main())
    assert [r.request_id for r in results] == ["r-0", "r-1", "r-2"]
    assert [r.response.content if r.response else None for r in results] == [
        "content-0",
        "content-1",
        "content-2",
    ]


def test_concurrent_runs_on_one_orchestrator_stay_isolated() -> None:
    async def main() -> list[OrchestrationResult]:
        provider = ScriptedProvider("fake", [_success("shared")])
        orchestrator = _orchestrator(provider)
        return await asyncio.gather(
            *(orchestrator.run(_request(f"q-{i}", request_id=f"r-{i}"), _model()) for i in range(5))
        )

    results = asyncio.run(main())
    assert [r.request_id for r in results] == [f"r-{i}" for i in range(5)]
    assert all(r.succeeded for r in results)


def test_provider_errors_do_not_cross_between_concurrent_runs() -> None:
    async def main() -> tuple[OrchestrationResult, OrchestrationResult]:
        good = _orchestrator(ScriptedProvider("good", [_success("fine")]))
        bad = _orchestrator(
            ScriptedProvider("bad", [NetworkError("down", provider="bad", model="m")])
        )
        return await asyncio.gather(
            good.run(_request("g", request_id="g"), _model("good")),
            bad.run(_request("b", request_id="b"), _model("bad")),
        )

    ok_result, err_result = asyncio.run(main())
    assert ok_result.succeeded is True
    assert err_result.succeeded is False
    assert err_result.error is not None
    assert err_result.error.provider == "bad"


def test_cancelled_provider_call_propagates_cancellation() -> None:
    orchestrator = _orchestrator(ScriptedProvider("fake", [asyncio.CancelledError()]))
    with pytest.raises(asyncio.CancelledError):
        asyncio.run(orchestrator.run(_request(), _model()))


def test_mid_flight_cancellation_is_not_a_result() -> None:
    async def main() -> None:
        started = asyncio.Event()
        release = asyncio.Event()

        class BlockingProvider(ScriptedProvider):
            async def generate(
                self,
                request: LLMRequest,
                model: ModelConfig,
                *,
                timeout_s: float | None = None,
            ) -> LLMResponse:
                started.set()
                await release.wait()
                return _success()

        orchestrator = _orchestrator(BlockingProvider("fake", [_success()]))
        task = asyncio.create_task(orchestrator.run(_request(), _model()))
        await started.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(main())
