"""Typed top-level configuration.

An :class:`OrchestratorSettings` is the single validated object an application
builds to describe *what* it wants: which providers are configured, which
model candidates exist, which retry/fallback/routing policies apply, and what
the library-level default timeout is. It reuses the existing contract types
verbatim — no policy shape is duplicated here.

Secrets are never part of settings: providers name their credential via
``ProviderConfig.credential_env_var`` and the value is resolved from the
environment only inside :func:`llm_orchestrator.config.loader.build_registry`,
passed as a plain local to the provider constructor, and never stored.
"""

from __future__ import annotations

import math

from pydantic import BaseModel, Field, field_validator

from ..contracts._config import CONTRACT_MODEL_CONFIG
from ..contracts.models import ModelConfig, ProviderConfig
from ..contracts.policies import FallbackPolicy, RetryPolicy
from ..routing.static import StaticRoutingPolicy


class OrchestratorSettings(BaseModel):
    """The complete, validated configuration for one orchestration setup.

    All fields have defaults so ``OrchestratorSettings()`` is valid: an empty
    configuration means "no providers, no model candidates, default policies,
    no library-level timeout". :func:`build_registry` then yields an empty
    registry, and candidate selection over the empty model set fails with
    ``ConfigurationError`` at use time — configuration itself never invents
    providers or models.

    ``None`` means "unset" everywhere: Pydantic does not distinguish an
    omitted ``default_timeout_s`` from an explicitly supplied ``None``, so
    :func:`load_settings` may fill either from the environment. There is
    intentionally no way to force "no timeout" against an environment
    default — omit the variable instead.
    """

    model_config = CONTRACT_MODEL_CONFIG

    providers: tuple[ProviderConfig, ...] = ()
    models: tuple[ModelConfig, ...] = ()
    retry: RetryPolicy = Field(default_factory=RetryPolicy)
    fallback: FallbackPolicy = Field(default_factory=FallbackPolicy)
    routing: StaticRoutingPolicy = Field(default_factory=StaticRoutingPolicy)
    default_timeout_s: float | None = None

    @field_validator("default_timeout_s")
    @classmethod
    def _validate_default_timeout(cls, value: float | None) -> float | None:
        if value is not None:
            if not math.isfinite(value):
                raise ValueError("OrchestratorSettings.default_timeout_s must be finite")
            if value <= 0:
                raise ValueError("OrchestratorSettings.default_timeout_s must be > 0")
        return value
