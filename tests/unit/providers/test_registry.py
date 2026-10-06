"""Tests for the provider registry."""

from __future__ import annotations

import pytest

from llm_orchestrator.errors import (
    ConfigurationError,
    LLMError,
    ProviderNotRegisteredError,
)
from llm_orchestrator.providers import ProviderRegistry


def test_register_then_get_returns_the_same_instance(fake_provider) -> None:
    registry = ProviderRegistry()
    registry.register(fake_provider)
    assert registry.get("fake") is fake_provider


def test_has_and_contains_reflect_registration(fake_provider) -> None:
    registry = ProviderRegistry()
    assert registry.has("fake") is False
    assert "fake" not in registry
    registry.register(fake_provider)
    assert registry.has("fake") is True
    assert "fake" in registry


def test_provider_ids_are_sorted_and_deterministic(make_provider) -> None:
    registry = ProviderRegistry()
    registry.register(make_provider(provider_id="zeta"))
    registry.register(make_provider(provider_id="alpha"))
    registry.register(make_provider(provider_id="mid"))
    assert registry.provider_ids() == ("alpha", "mid", "zeta")
    assert len(registry) == 3


def test_duplicate_registration_is_rejected(make_provider) -> None:
    registry = ProviderRegistry()
    registry.register(make_provider(provider_id="dup"))
    with pytest.raises(ConfigurationError):
        registry.register(make_provider(provider_id="dup"))


def test_duplicate_registration_can_be_replaced_explicitly(make_provider) -> None:
    registry = ProviderRegistry()
    first = make_provider(provider_id="dup", content_prefix="first")
    second = make_provider(provider_id="dup", content_prefix="second")
    registry.register(first)
    registry.register(second, replace=True)
    assert registry.get("dup") is second


def test_unknown_provider_raises_a_normalized_error() -> None:
    registry = ProviderRegistry()
    with pytest.raises(ProviderNotRegisteredError) as excinfo:
        registry.get("missing")
    assert isinstance(excinfo.value, ConfigurationError)
    assert isinstance(excinfo.value, LLMError)
    assert excinfo.value.provider == "missing"


def test_unknown_provider_does_not_return_none() -> None:
    registry = ProviderRegistry()
    assert registry.has("missing") is False


def test_unregister_removes_the_provider(fake_provider) -> None:
    registry = ProviderRegistry()
    registry.register(fake_provider)
    registry.unregister("fake")
    assert registry.has("fake") is False
    assert len(registry) == 0


def test_unregister_unknown_provider_raises() -> None:
    registry = ProviderRegistry()
    with pytest.raises(ProviderNotRegisteredError):
        registry.unregister("missing")


def test_registries_are_isolated(make_provider) -> None:
    first = ProviderRegistry()
    second = ProviderRegistry()
    first.register(make_provider(provider_id="only-in-first"))
    assert first.has("only-in-first") is True
    assert second.has("only-in-first") is False


def test_registering_a_non_provider_is_rejected() -> None:
    registry = ProviderRegistry()
    with pytest.raises(TypeError):
        registry.register(object())


def test_registering_a_blank_provider_id_is_rejected(make_provider) -> None:
    registry = ProviderRegistry()
    with pytest.raises(ConfigurationError):
        registry.register(make_provider(provider_id="   "))
