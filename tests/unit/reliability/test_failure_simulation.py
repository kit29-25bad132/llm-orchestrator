"""Deterministic failure simulation across routing → retry → fallback.

This suite complements ``test_retry_engine.py`` and ``test_fallback.py``: those
prove each layer in depth, while these tests prove the composed pipeline
behaves correctly under realistic success and failure sequences. Every test
uses the shared :mod:`support.fakes` doubles and deterministic scripts; any
backoff wait is recorded instead of performed — fully offline, never real time.

For each scenario the test asserts the complete observable outcome: which
provider/model was selected, invocation counts and attempt order, whether
retry and fallback occurred, the final response or error, attempt-history
details, and secret safety where applicable.
"""

from __future__ import annotations

from llm_orchestrator.contracts import (
    CandidateRef,
    FallbackPolicy,
    JitterStrategy,
    RetryPolicy,
)
from llm_orchestrator.errors import (
    AuthenticationError,
    ErrorCategory,
    InvalidRequestError,
    LLMTimeoutError,
    NetworkError,
    ProviderUnavailableError,
    RateLimitError,
    StructuredOutputValidationError,
)
from llm_orchestrator.reliability import FallbackExecutor, RetryEngine
from support.fakes import (
    RecordingSleep,
    ScriptedProvider,
    attempt_sequence,
    make_model,
    make_registry,
    make_request,
    make_response,
    run_sync,
)

ALLOW = FallbackPolicy(max_fallbacks=5)


def _engine(
    max_attempts: int, recorder: RecordingSleep, *, initial_backoff_s: float = 0.0
) -> RetryEngine:
    return RetryEngine(
        RetryPolicy(
            max_attempts=max_attempts,
            initial_backoff_s=initial_backoff_s,
            jitter=JitterStrategy.NONE,
        ),
        sleep=recorder.sleep,
    )


def test_transient_failure_retries_same_candidate_without_fallback() -> None:
    """A transient blip is absorbed by retry; fallback never engages."""
    recorder = RecordingSleep()
    first = ScriptedProvider("pa", [NetworkError("blip"), make_response("pa")])
    second = ScriptedProvider("pb", [make_response("pb")])
    executor = FallbackExecutor(make_registry(first, second), _engine(3, recorder), ALLOW)

    result = run_sync(executor.run(make_request(), [make_model("pa"), make_model("pb")]))

    assert result.succeeded is True
    assert result.response is not None and result.response.provider == "pa"
    assert len(first.calls) == 2
    assert second.calls == []
    assert result.fallback_count == 0
    assert result.retry_count == 1
    assert attempt_sequence(result) == [("pa", "m", 1, False), ("pa", "m", 2, True)]
    assert len(recorder.delays) == 1


def test_retry_exhaustion_falls_back_to_next_candidate() -> None:
    """Each candidate spends its own retry budget before the plan advances."""
    recorder = RecordingSleep()
    first = ScriptedProvider("pa", [NetworkError("down")])
    second = ScriptedProvider("pb", [make_response("pb")])
    executor = FallbackExecutor(make_registry(first, second), _engine(2, recorder), ALLOW)

    result = run_sync(executor.run(make_request(), [make_model("pa"), make_model("pb")]))

    assert result.succeeded is True
    assert result.response is not None and result.response.provider == "pb"
    assert len(first.calls) == 2
    assert len(second.calls) == 1
    assert result.fallback_count == 1
    assert result.retry_count == 1
    assert attempt_sequence(result) == [
        ("pa", "m", 1, False),
        ("pa", "m", 2, False),
        ("pb", "m", 1, True),
    ]
    assert [a.error_category for a in result.attempts[:2]] == [ErrorCategory.NETWORK] * 2


def test_rate_limit_retry_after_capped_at_max_backoff() -> None:
    """An excessive ``retry_after_s`` never exceeds ``max_backoff_s``."""
    recorder = RecordingSleep()
    first = ScriptedProvider(
        "pa",
        [RateLimitError("slow", retry_after_s=3600.0), make_response("pa")],
    )
    engine = RetryEngine(
        RetryPolicy(max_attempts=2, initial_backoff_s=0.5, max_backoff_s=4.0),
        sleep=recorder.sleep,
    )
    executor = FallbackExecutor(make_registry(first), engine, ALLOW)

    result = run_sync(executor.run(make_request(), [make_model("pa")]))

    assert result.succeeded is True
    assert recorder.delays == [4.0]
    assert result.fallback_count == 0


def test_timeout_retries_then_falls_back_with_timeout_propagated() -> None:
    """Timeouts follow the normalized retry/fallback flags on every candidate."""
    recorder = RecordingSleep()
    first = ScriptedProvider("pa", [LLMTimeoutError("slow")])
    second = ScriptedProvider("pb", [make_response("pb")])
    executor = FallbackExecutor(make_registry(first, second), _engine(2, recorder), ALLOW)

    result = run_sync(
        executor.run(make_request(), [make_model("pa"), make_model("pb")], timeout_s=7.5)
    )

    assert result.succeeded is True
    assert result.response is not None and result.response.provider == "pb"
    assert len(first.calls) == 2
    assert [call["timeout_s"] for call in first.calls] == [7.5, 7.5]
    assert second.calls[0]["timeout_s"] == 7.5
    assert [a.error_category for a in result.attempts[:2]] == [ErrorCategory.TIMEOUT] * 2
    assert result.fallback_count == 1


def test_provider_unavailable_selects_next_eligible_candidate() -> None:
    """A provider outage moves execution to the next candidate in routing order."""
    first = ScriptedProvider("pa", [ProviderUnavailableError("outage")])
    second = ScriptedProvider("pb", [make_response("pb")])
    executor = FallbackExecutor(make_registry(first, second), policy=ALLOW)

    result = run_sync(executor.run(make_request(), [make_model("pa"), make_model("pb")]))

    assert result.succeeded is True
    assert result.candidates_tried == (
        CandidateRef(provider="pa", model="m"),
        CandidateRef(provider="pb", model="m"),
    )
    assert result.fallback_count == 1
    assert result.attempts[0].error_category is ErrorCategory.PROVIDER_UNAVAILABLE


def test_authentication_failure_never_retries_but_still_falls_back() -> None:
    """Non-retryable auth failures skip sleeping yet remain fallback-eligible."""
    recorder = RecordingSleep()
    first = ScriptedProvider("pa", [AuthenticationError("bad key")])
    second = ScriptedProvider("pb", [make_response("pb")])
    executor = FallbackExecutor(make_registry(first, second), _engine(3, recorder), ALLOW)

    result = run_sync(executor.run(make_request(), [make_model("pa"), make_model("pb")]))

    assert result.succeeded is True
    assert len(first.calls) == 1
    assert recorder.delays == []
    assert result.fallback_count == 1
    assert result.retry_count == 0


def test_invalid_request_is_terminal_without_retry_or_fallback() -> None:
    """Deterministic request errors stop the plan at the first candidate."""
    recorder = RecordingSleep()
    first = ScriptedProvider("pa", [InvalidRequestError("bad params")])
    second = ScriptedProvider("pb", [make_response("pb")])
    executor = FallbackExecutor(make_registry(first, second), _engine(3, recorder), ALLOW)

    result = run_sync(executor.run(make_request(), [make_model("pa"), make_model("pb")]))

    assert result.succeeded is False
    assert result.error is not None
    assert result.error.category is ErrorCategory.INVALID_REQUEST
    assert len(first.calls) == 1
    assert recorder.delays == []
    assert second.calls == []
    assert result.candidates_tried == (CandidateRef(provider="pa", model="m"),)


def test_validation_failure_retries_then_falls_back_when_included() -> None:
    """Validation failures are retried per candidate, then fall back by default."""
    recorder = RecordingSleep()
    first = ScriptedProvider("pa", [StructuredOutputValidationError("mismatch")])
    second = ScriptedProvider("pb", [make_response("pb")])
    executor = FallbackExecutor(make_registry(first, second), _engine(2, recorder), ALLOW)

    result = run_sync(executor.run(make_request(), [make_model("pa"), make_model("pb")]))

    assert result.succeeded is True
    assert len(first.calls) == 2
    assert result.fallback_count == 1
    assert [a.error_category for a in result.attempts[:2]] == [
        ErrorCategory.STRUCTURED_OUTPUT_VALIDATION
    ] * 2


def test_validation_failure_stops_plan_when_excluded() -> None:
    """``include_validation_failures=False`` makes validation terminal per plan."""
    recorder = RecordingSleep()
    first = ScriptedProvider("pa", [StructuredOutputValidationError("mismatch")])
    second = ScriptedProvider("pb", [make_response("pb")])
    executor = FallbackExecutor(make_registry(first, second), _engine(2, recorder))
    policy = FallbackPolicy(max_fallbacks=5, include_validation_failures=False)

    result = run_sync(
        executor.run(make_request(), [make_model("pa"), make_model("pb")], policy=policy)
    )

    assert result.succeeded is False
    assert result.error is not None
    assert result.error.category is ErrorCategory.STRUCTURED_OUTPUT_VALIDATION
    assert len(first.calls) == 2
    assert second.calls == []
    assert len(result.candidates_tried) == 1


def test_all_candidates_exhausted_reports_last_error_with_full_history() -> None:
    """Total exhaustion yields the last terminal error plus per-call history."""
    recorder = RecordingSleep()
    first = ScriptedProvider("pa", [NetworkError("first-down")])
    second = ScriptedProvider("pb", [ProviderUnavailableError("second-down")])
    executor = FallbackExecutor(make_registry(first, second), _engine(2, recorder), ALLOW)

    result = run_sync(executor.run(make_request(), [make_model("pa"), make_model("pb")]))

    assert result.succeeded is False
    assert result.response is None
    assert result.error is not None
    assert result.error.category is ErrorCategory.PROVIDER_UNAVAILABLE
    assert result.error.message == "second-down"
    assert attempt_sequence(result) == [
        ("pa", "m", 1, False),
        ("pa", "m", 2, False),
        ("pb", "m", 1, False),
        ("pb", "m", 2, False),
    ]
    assert result.fallback_count == 1
    assert result.retry_count == 2
    assert result.request_id == "req-1"


def test_duplicates_collapse_and_budget_truncates_plan() -> None:
    """Dedup, routing order, and ``max_fallbacks`` jointly bound the plan."""
    failing_a = ScriptedProvider("pa", [NetworkError("a")])
    failing_b = ScriptedProvider("pb", [NetworkError("b")])
    never = ScriptedProvider("pc", [make_response("pc")])
    executor = FallbackExecutor(make_registry(failing_a, failing_b, never))
    policy = FallbackPolicy(max_fallbacks=1)

    result = run_sync(
        executor.run(
            make_request(),
            [make_model("pa"), make_model("pb"), make_model("pa"), make_model("pc")],
            policy=policy,
        )
    )

    assert result.succeeded is False
    assert [c.provider for c in result.candidates_tried] == ["pa", "pb"]
    assert len(failing_a.calls) == 1
    assert never.calls == []


def test_allowlist_excludes_disallowed_candidates_in_routing_order() -> None:
    """The policy allowlist drops disallowed candidates while routing order is kept."""
    failing_a = ScriptedProvider("pa", [NetworkError("a")])
    failing_b = ScriptedProvider("pb", [NetworkError("b")])
    recovered = ScriptedProvider("pc", [make_response("pc")])
    executor = FallbackExecutor(make_registry(failing_a, failing_b, recovered))
    policy = FallbackPolicy(
        max_fallbacks=5,
        candidates=(
            CandidateRef(provider="pc", model="m"),
            CandidateRef(provider="pa", model="m"),
        ),
    )

    result = run_sync(
        executor.run(
            make_request(), [make_model("pa"), make_model("pb"), make_model("pc")], policy=policy
        )
    )

    assert result.succeeded is True
    assert [c.provider for c in result.candidates_tried] == ["pa", "pc"]
    assert failing_b.calls == []


def test_equivalent_adapter_failures_drive_identical_pipeline_behavior() -> None:
    """Equal normalized failures from Gemini/Groq take the same retry/fallback path."""
    from llm_orchestrator.providers import gemini as gemini_adapter
    from llm_orchestrator.providers import groq as groq_adapter

    def classify_gemini() -> object:
        from google.genai import errors as genai_errors

        return gemini_adapter._map_exception(
            genai_errors.ClientError(429, {"error": {"code": 429, "message": "quota"}}),
            model="m",
            request_id="req-1",
        )

    def classify_groq() -> object:
        import groq
        import httpx

        request = httpx.Request("POST", "https://api.groq.com/openai/v1/chat/completions")
        response = httpx.Response(429, request=request)
        failure = groq.RateLimitError("quota", response=response, body=None)
        return groq_adapter._map_exception(failure, model="m", request_id="req-1")

    gemini_failure = classify_gemini()
    groq_failure = classify_groq()
    assert isinstance(gemini_failure, RateLimitError)
    assert isinstance(groq_failure, RateLimitError)

    outcomes: list[RateLimitError] = [gemini_failure, groq_failure]
    for failure in outcomes:
        recorder = RecordingSleep()
        first = ScriptedProvider("pa", [failure])
        second = ScriptedProvider("pb", [make_response("pb")])
        executor = FallbackExecutor(make_registry(first, second), _engine(2, recorder), ALLOW)

        result = run_sync(executor.run(make_request(), [make_model("pa"), make_model("pb")]))

        assert result.succeeded is True
        assert len(first.calls) == 2
        assert recorder.delays == [0.0]
        assert result.fallback_count == 1
        assert result.attempts[0].error_category is ErrorCategory.RATE_LIMIT


def test_terminal_error_messages_never_expose_secrets() -> None:
    """Credential-shaped tokens in failures are redacted in the failed result."""
    secret = "Bearer sk-test-" + "s3cr3t" * 4
    first = ScriptedProvider("pa", [NetworkError(f"call failed: {secret}")])
    executor = FallbackExecutor(make_registry(first), policy=ALLOW)

    result = run_sync(executor.run(make_request(), [make_model("pa")]))

    assert result.succeeded is False
    assert result.error is not None
    assert secret not in result.error.message
    assert "[REDACTED]" in result.error.message
    assert secret not in result.model_dump_json()
