"""Settings-model tests: explicit construction, validation, and empty config.

Direct construction must never touch the environment; every test here builds
settings from explicit values only.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from llm_orchestrator.config import OrchestratorSettings
from llm_orchestrator.contracts import (
    FallbackPolicy,
    ModelConfig,
    ProviderConfig,
    RetryPolicy,
)
from llm_orchestrator.routing import StaticRoutingPolicy


def _providers() -> tuple[ProviderConfig, ...]:
    return (
        ProviderConfig(name="gemini", credential_env_var="GEMINI_API_KEY"),
        ProviderConfig(name="groq", credential_env_var="GROQ_API_KEY"),
    )


def _models() -> tuple[ModelConfig, ...]:
    return (
        ModelConfig(provider="gemini", model="gemini-test-model"),
        ModelConfig(provider="groq", model="groq-test-model"),
    )


def test_explicit_construction_with_valid_configuration() -> None:
    settings = OrchestratorSettings(
        providers=_providers(),
        models=_models(),
        retry=RetryPolicy(max_attempts=3),
        fallback=FallbackPolicy(max_fallbacks=1),
        routing=StaticRoutingPolicy(preferred_provider="gemini"),
        default_timeout_s=30.0,
    )
    assert settings.providers[0].name == "gemini"
    assert settings.models[1].model == "groq-test-model"
    assert settings.retry.max_attempts == 3
    assert settings.fallback.max_fallbacks == 1
    assert settings.routing.preferred_provider == "gemini"
    assert settings.default_timeout_s == 30.0


def test_defaults_are_conservative_and_empty() -> None:
    settings = OrchestratorSettings()
    assert settings.providers == ()
    assert settings.models == ()
    assert settings.retry == RetryPolicy()
    assert settings.fallback == FallbackPolicy()
    assert settings.routing == StaticRoutingPolicy()
    assert settings.default_timeout_s is None


def test_empty_configuration_is_permitted_and_documented() -> None:
    """Empty means "nothing configured": registry builds empty, selection fails later."""
    settings = OrchestratorSettings(providers=(), models=())
    assert settings.providers == ()
    assert settings.models == ()


@pytest.mark.parametrize("timeout", [0.0, -1.0, -0.5])
def test_non_positive_default_timeout_is_rejected(timeout: float) -> None:
    with pytest.raises(ValidationError):
        OrchestratorSettings(default_timeout_s=timeout)


@pytest.mark.parametrize("timeout", [float("inf"), float("-inf"), float("nan")])
def test_non_finite_default_timeout_is_rejected(timeout: float) -> None:
    with pytest.raises(ValidationError):
        OrchestratorSettings(default_timeout_s=timeout)


def test_unknown_fields_are_rejected() -> None:
    with pytest.raises(ValidationError):
        OrchestratorSettings(api_key="gsk_test_should_not_exist_12345678")  # type: ignore[call-arg]


def test_settings_are_frozen() -> None:
    settings = OrchestratorSettings()
    with pytest.raises(ValidationError):
        settings.default_timeout_s = 5.0  # type: ignore[misc]


def test_settings_hold_no_secret_values_by_construction() -> None:
    """Settings name credentials via ``credential_env_var``; values never live here."""
    settings = OrchestratorSettings(providers=_providers(), models=_models())
    dumped = settings.model_dump_json()
    assert "credential_env_var" in dumped
    assert "GEMINI_API_KEY" in dumped  # the *name* is configuration, not a secret
    assert "gsk_" not in dumped
    assert "AIza" not in dumped
    assert "gsk_" not in repr(settings)
    assert "AIza" not in repr(settings)
