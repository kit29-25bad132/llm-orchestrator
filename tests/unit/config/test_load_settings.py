"""Environment-loading tests: mapping, precedence, and malformed values.

The environment is always injected as a plain dictionary, except for the two
tests that deliberately exercise the ``os.environ`` fallback via monkeypatch.
"""

from __future__ import annotations

import pytest

from llm_orchestrator.config import OrchestratorSettings, load_settings
from llm_orchestrator.config.loader import DEFAULT_TIMEOUT_ENV_VAR
from llm_orchestrator.contracts import RetryPolicy
from llm_orchestrator.errors import ConfigurationError


def test_no_environment_yields_defaults() -> None:
    settings = load_settings(env={})
    assert settings == OrchestratorSettings()
    assert settings.default_timeout_s is None


def test_environment_timeout_is_applied_without_explicit_settings() -> None:
    settings = load_settings(env={DEFAULT_TIMEOUT_ENV_VAR: "12.5"})
    assert settings.default_timeout_s == 12.5


def test_blank_environment_timeout_is_ignored() -> None:
    settings = load_settings(env={DEFAULT_TIMEOUT_ENV_VAR: "   "})
    assert settings.default_timeout_s is None


@pytest.mark.parametrize("raw", ["not-a-number", "0", "-3", "inf", "nan", "-inf"])
def test_malformed_environment_timeout_raises_without_echoing_value(raw: str) -> None:
    with pytest.raises(ConfigurationError) as exc_info:
        load_settings(env={DEFAULT_TIMEOUT_ENV_VAR: raw})
    message = str(exc_info.value)
    assert DEFAULT_TIMEOUT_ENV_VAR in message
    assert raw not in message


def test_explicit_timeout_wins_over_environment() -> None:
    explicit = OrchestratorSettings(default_timeout_s=7.5)
    result = load_settings(explicit=explicit, env={DEFAULT_TIMEOUT_ENV_VAR: "30"})
    assert result is explicit
    assert result.default_timeout_s == 7.5


def test_environment_fills_only_an_unset_explicit_timeout() -> None:
    explicit = OrchestratorSettings(default_timeout_s=None, retry=RetryPolicy(max_attempts=4))
    result = load_settings(explicit=explicit, env={DEFAULT_TIMEOUT_ENV_VAR: "30"})
    assert result.default_timeout_s == 30.0
    assert result.retry.max_attempts == 4
    # The caller's object is never mutated; a copy carries the merged timeout.
    assert explicit.default_timeout_s is None
    assert result is not explicit


def test_explicit_without_environment_timeout_is_returned_unchanged() -> None:
    explicit = OrchestratorSettings(default_timeout_s=None)
    assert load_settings(explicit=explicit, env={}) is explicit


def test_omitted_and_explicit_none_timeouts_are_equivalent() -> None:
    """`None` means "unset": omitted and explicit `None` both take env fill."""
    env = {DEFAULT_TIMEOUT_ENV_VAR: "30"}
    assert load_settings(env=env) == load_settings(
        explicit=OrchestratorSettings(default_timeout_s=None), env=env
    )
    assert load_settings(explicit=OrchestratorSettings(), env=env).default_timeout_s == 30.0


def test_supplied_env_is_used_exclusively(monkeypatch: pytest.MonkeyPatch) -> None:
    """An injected mapping wins even when the real environment disagrees."""
    monkeypatch.setenv(DEFAULT_TIMEOUT_ENV_VAR, "99")
    settings = load_settings(env={})
    assert settings.default_timeout_s is None


def test_os_environ_is_read_only_inside_the_loader(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(DEFAULT_TIMEOUT_ENV_VAR, "11")
    try:
        settings = load_settings()
    finally:
        monkeypatch.delenv(DEFAULT_TIMEOUT_ENV_VAR, raising=False)
    assert settings.default_timeout_s == 11.0


def test_resolve_timeout_precedence() -> None:
    from llm_orchestrator.config import resolve_timeout_s
    from llm_orchestrator.config.loader import resolve_timeout_s as resolve_direct
    from llm_orchestrator.contracts import LLMRequest, Message, MessageRole, ProviderConfig

    assert resolve_timeout_s is resolve_direct
    settings = OrchestratorSettings(
        providers=(ProviderConfig(name="gemini", default_timeout_s=20.0),),
        default_timeout_s=60.0,
    )
    request = LLMRequest(messages=[Message(role=MessageRole.USER, content="hi")], timeout_s=5.0)
    assert resolve_timeout_s(settings) == 60.0
    assert resolve_timeout_s(settings, provider_name="gemini") == 20.0
    assert resolve_timeout_s(settings, provider_name="unknown") == 60.0
    assert resolve_timeout_s(settings, provider_name="gemini", request=request) == 5.0
    assert (
        resolve_timeout_s(settings, provider_name="gemini", request=request, timeout_s=1.0) == 1.0
    )
    assert resolve_timeout_s(OrchestratorSettings()) is None
