"""Tests for the provider/model descriptor contracts."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from llm_orchestrator.contracts import (
    CandidateRef,
    ModelCapabilities,
    ModelConfig,
    ModelPricing,
    ProviderConfig,
)


def test_minimal_model_config() -> None:
    model = ModelConfig(provider="some-provider", model="some-model")
    assert model.provider == "some-provider"
    assert model.model == "some-model"
    assert model.enabled is True
    assert model.pricing is None
    assert model.quality_tier is None
    assert dict(model.tags) == {}
    assert model.capabilities.structured_output is False
    assert model.capabilities.max_context_tokens is None


@pytest.mark.parametrize(("provider", "model"), [("", "m"), ("   ", "m"), ("p", ""), ("p", "  ")])
def test_blank_model_identifiers_rejected(provider: str, model: str) -> None:
    with pytest.raises(ValidationError):
        ModelConfig(provider=provider, model=model)


def test_provider_and_model_are_opaque_strings() -> None:
    # A future, unknown provider must work without any core change.
    model = ModelConfig(provider="brand-new-vendor", model="vendor/x-9000")
    assert model.provider == "brand-new-vendor"
    assert model.model == "vendor/x-9000"


def test_capabilities() -> None:
    caps = ModelCapabilities(structured_output=True, streaming=True, max_context_tokens=8192)
    assert caps.structured_output is True
    assert caps.streaming is True
    assert caps.tool_calling is False
    assert caps.max_context_tokens == 8192


def test_non_positive_context_rejected() -> None:
    with pytest.raises(ValidationError):
        ModelCapabilities(max_context_tokens=0)


def test_pricing_defaults_and_validation() -> None:
    pricing = ModelPricing(input_per_1k_tokens=0.5, output_per_1k_tokens=1.5, currency="USD")
    assert pricing.input_per_1k_tokens == 0.5
    assert pricing.currency == "USD"
    with pytest.raises(ValidationError):
        ModelPricing(input_per_1k_tokens=-1.0)


def test_model_config_rejects_extra_fields() -> None:
    with pytest.raises(ValidationError):
        ModelConfig(provider="p", model="m", api_key="secret")


def test_minimal_provider_config() -> None:
    provider = ProviderConfig(name="some-provider")
    assert provider.name == "some-provider"
    assert provider.enabled is True
    assert provider.credential_env_var is None
    assert provider.base_url is None
    assert provider.default_timeout_s is None
    assert provider.max_concurrency is None


def test_provider_config_names_an_env_var_but_never_a_secret() -> None:
    provider = ProviderConfig(name="some-provider", credential_env_var="SOME_PROVIDER_API_KEY")
    assert provider.credential_env_var == "SOME_PROVIDER_API_KEY"
    assert not hasattr(provider, "api_key")


def test_blank_provider_name_rejected() -> None:
    with pytest.raises(ValidationError):
        ProviderConfig(name="   ")


def test_provider_config_invalid_values_rejected() -> None:
    with pytest.raises(ValidationError):
        ProviderConfig(name="p", max_concurrency=0)
    with pytest.raises(ValidationError):
        ProviderConfig(name="p", default_timeout_s=0)


def test_candidate_ref() -> None:
    candidate = CandidateRef(provider="p", model="m")
    assert (candidate.provider, candidate.model) == ("p", "m")
    with pytest.raises(ValidationError):
        CandidateRef(provider="", model="m")
