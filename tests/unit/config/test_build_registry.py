"""Registry-factory tests: construction, credentials, injection, and isolation.

All tests are offline. Injected clients are inert sentinels: if construction
tried to use them as live SDK clients (or contacted any API), these tests
would fail. No real API keys are used anywhere.
"""

from __future__ import annotations

import pytest

from llm_orchestrator.config import OrchestratorSettings, build_registry
from llm_orchestrator.contracts import ModelConfig, ProviderConfig
from llm_orchestrator.errors import ConfigurationError
from llm_orchestrator.providers import ProviderRegistry

GEMINI_ENV = "GEMINI_API_KEY"
GROQ_ENV = "GROQ_API_KEY"
# Fake values shaped like real keys so redaction coverage is meaningful.
FAKE_GEMINI_KEY = "AIzaTestFakeKey0123456789abcdef"
FAKE_GROQ_KEY = "gsk_test_fake_value_0123456789abcdef"


def _settings(*providers: ProviderConfig) -> OrchestratorSettings:
    return OrchestratorSettings(providers=providers)


def test_builds_gemini_and_groq_with_injected_clients() -> None:
    gemini_client = object()
    groq_client = object()
    settings = _settings(
        ProviderConfig(name="gemini", credential_env_var=GEMINI_ENV),
        ProviderConfig(name="groq", credential_env_var=GROQ_ENV),
    )
    registry = build_registry(
        settings,
        env={GEMINI_ENV: FAKE_GEMINI_KEY, GROQ_ENV: FAKE_GROQ_KEY},
        client_factory={"gemini": gemini_client, "groq": groq_client},
    )
    assert isinstance(registry, ProviderRegistry)
    assert registry.provider_ids() == ("gemini", "groq")
    assert registry.get("gemini").provider_id == "gemini"
    assert registry.get("groq").provider_id == "groq"


def test_callable_client_factory_receives_provider_names() -> None:
    seen: list[str] = []
    sentinels = {"gemini": object(), "groq": object()}

    def factory(name: str) -> object | None:
        seen.append(name)
        return sentinels[name]

    settings = _settings(
        ProviderConfig(name="gemini", credential_env_var=GEMINI_ENV),
        ProviderConfig(name="groq", credential_env_var=GROQ_ENV),
    )
    registry = build_registry(
        settings,
        env={GEMINI_ENV: FAKE_GEMINI_KEY, GROQ_ENV: FAKE_GROQ_KEY},
        client_factory=factory,
    )
    assert seen == ["gemini", "groq"]
    assert len(registry) == 2


def test_no_provider_calls_occur_during_construction() -> None:
    """Sentinel clients would explode if treated as live SDK clients."""

    class _ExplodingClient:
        def __getattr__(self, name: str) -> object:
            raise AssertionError(f"client must not be touched during build ({name})")

    settings = _settings(
        ProviderConfig(name="gemini", credential_env_var=GEMINI_ENV),
        ProviderConfig(name="groq", credential_env_var=GROQ_ENV),
    )
    registry = build_registry(
        settings,
        env={GEMINI_ENV: FAKE_GEMINI_KEY, GROQ_ENV: FAKE_GROQ_KEY},
        client_factory={"gemini": _ExplodingClient(), "groq": _ExplodingClient()},
    )
    assert registry.provider_ids() == ("gemini", "groq")


def test_missing_credential_raises_safe_error() -> None:
    settings = _settings(ProviderConfig(name="gemini", credential_env_var=GEMINI_ENV))
    with pytest.raises(ConfigurationError) as exc_info:
        build_registry(settings, env={})
    message = str(exc_info.value)
    assert "gemini" in message
    assert GEMINI_ENV in message


def test_blank_credential_raises_safe_error() -> None:
    settings = _settings(ProviderConfig(name="groq", credential_env_var=GROQ_ENV))
    with pytest.raises(ConfigurationError) as exc_info:
        build_registry(settings, env={GROQ_ENV: "   "})
    message = str(exc_info.value)
    assert "groq" in message
    assert GROQ_ENV in message


def test_credential_values_never_appear_in_errors_or_settings() -> None:
    settings = _settings(
        ProviderConfig(name="gemini", credential_env_var=GEMINI_ENV),
        ProviderConfig(name="groq", credential_env_var=GROQ_ENV),
    )
    # Fail the groq credential while gemini's (fake, key-shaped) value is present.
    with pytest.raises(ConfigurationError) as exc_info:
        build_registry(settings, env={GEMINI_ENV: FAKE_GEMINI_KEY})
    assert FAKE_GEMINI_KEY not in str(exc_info.value)
    assert FAKE_GROQ_KEY not in str(exc_info.value)
    assert FAKE_GEMINI_KEY not in repr(settings)
    assert FAKE_GROQ_KEY not in repr(settings)
    assert FAKE_GEMINI_KEY not in settings.model_dump_json()
    assert "AIza" not in settings.model_dump_json()
    assert "gsk_" not in settings.model_dump_json()


def test_injected_client_needs_no_credential() -> None:
    settings = _settings(ProviderConfig(name="gemini", credential_env_var=None))
    registry = build_registry(settings, env={}, client_factory={"gemini": object()})
    assert registry.provider_ids() == ("gemini",)


def test_missing_credential_without_client_fails_at_provider_boundary() -> None:
    settings = _settings(ProviderConfig(name="gemini", credential_env_var=None))
    with pytest.raises(ConfigurationError):
        build_registry(settings, env={})


def test_unsupported_provider_fails_safely() -> None:
    settings = _settings(ProviderConfig(name="acme", credential_env_var="ACME_API_KEY"))
    with pytest.raises(ConfigurationError) as exc_info:
        build_registry(settings, env={"ACME_API_KEY": "not-a-real-key"}, client_factory={})
    assert "acme" in str(exc_info.value)


def test_disabled_providers_are_skipped_before_credential_resolution() -> None:
    settings = _settings(
        ProviderConfig(name="gemini", enabled=False, credential_env_var=GEMINI_ENV),
    )
    registry = build_registry(settings, env={})
    assert len(registry) == 0


def test_empty_settings_builds_an_empty_registry() -> None:
    assert len(build_registry(OrchestratorSettings(), env={})) == 0


def test_duplicate_provider_names_fail() -> None:
    settings = _settings(
        ProviderConfig(name="gemini", credential_env_var=GEMINI_ENV),
        ProviderConfig(name="gemini", credential_env_var=GEMINI_ENV),
    )
    with pytest.raises(ConfigurationError):
        build_registry(settings, env={GEMINI_ENV: FAKE_GEMINI_KEY})


def test_registries_are_independent_instances() -> None:
    settings = _settings(ProviderConfig(name="gemini", credential_env_var=GEMINI_ENV))
    env = {GEMINI_ENV: FAKE_GEMINI_KEY}
    first = build_registry(settings, env=env, client_factory={"gemini": object()})
    second = build_registry(settings, env=env, client_factory={"gemini": object()})
    assert first is not second
    assert first.get("gemini") is not second.get("gemini")
    first.unregister("gemini")
    assert second.provider_ids() == ("gemini",)


def test_settings_models_flow_into_selection_unchanged() -> None:
    """Configured models stay usable by the existing router without conversion."""
    from llm_orchestrator.contracts import LLMRequest, Message, MessageRole
    from llm_orchestrator.routing import ModelRouter

    settings = OrchestratorSettings(
        providers=(ProviderConfig(name="gemini"),),
        models=(ModelConfig(provider="gemini", model="gemini-test-model"),),
    )
    request = LLMRequest(messages=[Message(role=MessageRole.USER, content="hi")])
    selected = ModelRouter(settings.routing).select(request, settings.models)
    assert (selected.provider, selected.model) == ("gemini", "gemini-test-model")
