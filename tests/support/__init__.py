"""Shared test-only helpers for the offline suite.

This package is test infrastructure, not library code: nothing here is
exported from ``llm_orchestrator``. New tests should reuse :mod:`support.fakes`
instead of redefining another scripted provider. Existing suites keep their
local doubles untouched to avoid churn.
"""

from support.fakes import (
    FakeGeminiSdkClient,
    FakeGroqSdkClient,
    RecordingSleep,
    ScriptedProvider,
    attempt_sequence,
    make_model,
    make_registry,
    make_request,
    make_response,
    run_sync,
)

__all__ = [
    "FakeGeminiSdkClient",
    "FakeGroqSdkClient",
    "RecordingSleep",
    "ScriptedProvider",
    "attempt_sequence",
    "make_model",
    "make_registry",
    "make_request",
    "make_response",
    "run_sync",
]
