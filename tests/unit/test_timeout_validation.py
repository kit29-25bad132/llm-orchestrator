"""Invalid per-call ``timeout_s`` values must fail at every public seam.

Covers the approved Stage 1 boundary-validation scope: zero, negative, NaN,
and infinite ``timeout_s`` values raise one value-free ``ConfigurationError``
*before* any provider attempt, SDK call, backoff wait, or fallback planning
happens. Every seam (retry engine, fallback executor, orchestrator, and both
adapters) shares a single validator, so rules and wording stay identical.
"""

from __future__ import annotations

import pytest

from llm_orchestrator.contracts import RetryPolicy
from llm_orchestrator.errors import ConfigurationError, require_timeout_s
from llm_orchestrator.orchestration import Orchestrator
from llm_orchestrator.providers.gemini import GeminiProvider
from llm_orchestrator.providers.groq import GroqProvider
from llm_orchestrator.reliability import FallbackExecutor, RetryEngine
from support.fakes import (
    ScriptedProvider,
    make_model,
    make_registry,
    make_request,
    make_response,
    run_sync,
)

#: The single value-free rejection message asserted at every seam.
EXPECTED_MESSAGE = "timeout_s must be a positive, finite number of seconds."

#: The approved invalid inputs: zero, negative, NaN, and both infinities.
INVALID_TIMEOUTS = [
    0.0,
    -0.5,
    -1.0,
    float("nan"),
    float("inf"),
    float("-inf"),
]


def _scripted() -> ScriptedProvider:
    return ScriptedProvider("scripted", [make_response("scripted")])


@pytest.mark.parametrize("bad_timeout", INVALID_TIMEOUTS)
def test_retry_engine_rejects_invalid_timeout_before_any_attempt(
    bad_timeout: float,
) -> None:
    scripted = _scripted()
    engine = RetryEngine(RetryPolicy(max_attempts=3))
    with pytest.raises(ConfigurationError) as excinfo:
        run_sync(
            engine.generate(scripted, make_request(), make_model("scripted"), timeout_s=bad_timeout)
        )
    assert str(excinfo.value) == EXPECTED_MESSAGE
    assert not excinfo.value.retryable
    assert scripted.calls == []


@pytest.mark.parametrize("bad_timeout", INVALID_TIMEOUTS)
def test_fallback_executor_rejects_invalid_timeout_before_planning(
    bad_timeout: float,
) -> None:
    scripted = _scripted()
    executor = FallbackExecutor(make_registry(scripted), RetryEngine(RetryPolicy(max_attempts=1)))
    with pytest.raises(ConfigurationError) as excinfo:
        run_sync(executor.run(make_request(), [make_model("scripted")], timeout_s=bad_timeout))
    assert str(excinfo.value) == EXPECTED_MESSAGE
    assert scripted.calls == []


@pytest.mark.parametrize("bad_timeout", INVALID_TIMEOUTS)
def test_orchestrator_rejects_invalid_timeout_before_graph_invoke(
    bad_timeout: float,
) -> None:
    scripted = _scripted()
    orchestrator = Orchestrator(make_registry(scripted))
    with pytest.raises(ConfigurationError) as excinfo:
        run_sync(orchestrator.run(make_request(), make_model("scripted"), timeout_s=bad_timeout))
    assert str(excinfo.value) == EXPECTED_MESSAGE
    assert scripted.calls == []


@pytest.mark.parametrize("bad_timeout", INVALID_TIMEOUTS)
def test_groq_adapter_rejects_invalid_timeout_before_sdk_call(
    bad_timeout: float,
) -> None:
    # The inert sentinel client proves no SDK path is touched: without the
    # boundary check this call would surface an AttributeError instead.
    provider = GroqProvider(client=object())
    with pytest.raises(ConfigurationError) as excinfo:
        run_sync(provider.generate(make_request(), make_model("groq"), timeout_s=bad_timeout))
    assert str(excinfo.value) == EXPECTED_MESSAGE


@pytest.mark.parametrize("bad_timeout", INVALID_TIMEOUTS)
def test_gemini_adapter_rejects_invalid_timeout_before_sdk_call(
    bad_timeout: float,
) -> None:
    provider = GeminiProvider(client=object())
    with pytest.raises(ConfigurationError) as excinfo:
        run_sync(provider.generate(make_request(), make_model("gemini"), timeout_s=bad_timeout))
    assert str(excinfo.value) == EXPECTED_MESSAGE


@pytest.mark.parametrize("bad_timeout", [*INVALID_TIMEOUTS, False, True, "5", object()])
def test_shared_validator_rejects_invalid_values_with_value_free_message(
    bad_timeout: object,
) -> None:
    with pytest.raises(ConfigurationError) as excinfo:
        require_timeout_s(bad_timeout)  # type: ignore[arg-type]
    assert str(excinfo.value) == EXPECTED_MESSAGE


@pytest.mark.parametrize("ok_timeout", [None, 1, 0.5, 30.0])
def test_shared_validator_accepts_none_positive_finite_numbers(
    ok_timeout: float | None,
) -> None:
    require_timeout_s(ok_timeout)


def test_valid_integer_timeout_reaches_the_provider_unchanged() -> None:
    scripted = _scripted()
    engine = RetryEngine()
    response = run_sync(
        engine.generate(scripted, make_request(), make_model("scripted"), timeout_s=5)
    )
    assert response.content == "ok"
    assert scripted.calls[0]["timeout_s"] == 5
