"""Tests for the public export surface."""

from __future__ import annotations

import llm_orchestrator
import llm_orchestrator.contracts as contracts
import llm_orchestrator.errors as errors


def test_top_level_package_remains_identity_only() -> None:
    assert llm_orchestrator.__all__ == ["__version__"]
    assert isinstance(llm_orchestrator.__version__, str)
    assert llm_orchestrator.__version__


def test_contract_exports_are_importable() -> None:
    expected = {
        "AttemptRecord",
        "CandidateRef",
        "ErrorInfo",
        "FallbackPolicy",
        "FinishReason",
        "JitterStrategy",
        "LLMRequest",
        "LLMResponse",
        "Message",
        "MessageRole",
        "ModelCapabilities",
        "ModelConfig",
        "ModelPricing",
        "OrchestrationResult",
        "ProviderConfig",
        "RetryPolicy",
        "RoutingContext",
        "RoutingPolicy",
        "Usage",
    }
    assert set(contracts.__all__) == expected
    for name in expected:
        assert getattr(contracts, name) is not None


def test_error_exports_are_importable() -> None:
    expected = {
        "AuthenticationError",
        "AuthorizationError",
        "ConfigurationError",
        "ErrorCategory",
        "InvalidRequestError",
        "LLMError",
        "LLMTimeoutError",
        "MalformedResponseError",
        "ModelUnavailableError",
        "NetworkError",
        "ProviderNotRegisteredError",
        "ProviderUnavailableError",
        "RateLimitError",
        "StructuredOutputValidationError",
        "UnknownProviderError",
        "sanitize_message",
    }
    assert set(errors.__all__) == expected


def test_public_exports_do_not_leak_private_names() -> None:
    for name in (*contracts.__all__, *errors.__all__):
        assert not name.startswith("_"), f"private name exported: {name}"


def test_contracts_do_not_re_export_provider_implementations() -> None:
    # Guard against accidentally exposing concrete providers via the contract API.
    for forbidden in ("GeminiProvider", "GroqProvider", "BaseProvider"):
        assert not hasattr(contracts, forbidden)
