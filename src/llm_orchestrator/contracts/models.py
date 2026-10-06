"""Provider/model descriptor contracts.

These describe *what* a model/provider is. They contain no credentials and no
provider-specific branching; ``provider`` and ``model`` are opaque strings so new
providers and models need no core changes.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

from pydantic import BaseModel, Field, field_validator

from ._config import CONTRACT_MODEL_CONFIG


class ModelCapabilities(BaseModel):
    """Capability flags a routing/orchestration layer can filter on."""

    model_config = CONTRACT_MODEL_CONFIG

    structured_output: bool = False
    streaming: bool = False
    tool_calling: bool = False
    max_context_tokens: int | None = None

    @field_validator("max_context_tokens")
    @classmethod
    def _validate_context(cls, value: int | None) -> int | None:
        if value is not None and value <= 0:
            raise ValueError("ModelCapabilities.max_context_tokens must be positive")
        return value


class ModelPricing(BaseModel):
    """Optional cost metadata, reserved for future cost-aware routing."""

    model_config = CONTRACT_MODEL_CONFIG

    input_per_1k_tokens: float | None = None
    output_per_1k_tokens: float | None = None
    currency: str | None = None

    @field_validator("input_per_1k_tokens", "output_per_1k_tokens")
    @classmethod
    def _validate_rate(cls, value: float | None) -> float | None:
        if value is not None:
            if not math.isfinite(value):
                raise ValueError("ModelPricing rates must be finite")
            if value < 0:
                raise ValueError("ModelPricing rates must be >= 0")
        return value


class ModelConfig(BaseModel):
    """One concrete model offered by one provider."""

    model_config = CONTRACT_MODEL_CONFIG

    provider: str
    model: str
    capabilities: ModelCapabilities = Field(default_factory=ModelCapabilities)
    pricing: ModelPricing | None = None
    quality_tier: int | None = None
    enabled: bool = True
    tags: Mapping[str, str] = Field(default_factory=dict)

    @field_validator("provider", "model")
    @classmethod
    def _validate_identifiers(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("ModelConfig provider/model must not be blank")
        return value


class ProviderConfig(BaseModel):
    """Provider-level, non-secret configuration.

    ``credential_env_var`` names the environment variable holding the credential;
    the credential value itself is never part of a contract.
    """

    model_config = CONTRACT_MODEL_CONFIG

    name: str
    enabled: bool = True
    credential_env_var: str | None = None
    base_url: str | None = None
    default_timeout_s: float | None = None
    max_concurrency: int | None = None
    extra: Mapping[str, Any] = Field(default_factory=dict)

    @field_validator("name")
    @classmethod
    def _validate_name(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("ProviderConfig.name must not be blank")
        return value

    @field_validator("credential_env_var", "base_url")
    @classmethod
    def _validate_optional_strings(cls, value: str | None) -> str | None:
        if value is not None and not value.strip():
            raise ValueError("optional string fields must not be blank when provided")
        return value

    @field_validator("default_timeout_s")
    @classmethod
    def _validate_timeout(cls, value: float | None) -> float | None:
        if value is not None:
            if not math.isfinite(value):
                raise ValueError("ProviderConfig.default_timeout_s must be finite")
            if value <= 0:
                raise ValueError("ProviderConfig.default_timeout_s must be > 0")
        return value

    @field_validator("max_concurrency")
    @classmethod
    def _validate_concurrency(cls, value: int | None) -> int | None:
        if value is not None and value < 1:
            raise ValueError("ProviderConfig.max_concurrency must be >= 1")
        return value


class CandidateRef(BaseModel):
    """A lightweight ``(provider, model)`` pointer used by routing/fallback."""

    model_config = CONTRACT_MODEL_CONFIG

    provider: str
    model: str

    @field_validator("provider", "model")
    @classmethod
    def _validate_identifiers(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("CandidateRef provider/model must not be blank")
        return value
