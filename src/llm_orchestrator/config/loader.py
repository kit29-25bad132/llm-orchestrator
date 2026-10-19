"""Environment loading and provider-registry construction.

This module holds the only environment-reading and provider-constructing code
in the library outside the adapters themselves:

- :func:`load_settings` derives the library-level default timeout from the
  environment (explicit settings always win; see its docstring for the exact
  precedence). It performs no ``.env`` loading and reads ``os.environ`` only
  inside the function body — never at import time.
- :func:`build_registry` resolves each configured provider's credential from
  the environment and constructs the concrete adapters. It performs no model
  calls, creates no network clients eagerly (adapters create SDK clients
  lazily), keeps no global state, and implements no retry, fallback, routing,
  or LangGraph logic.
- :func:`resolve_timeout_s` is the pure precedence rule for timeouts; it
  performs no I/O.

Only ``GEMINI_API_KEY``, ``GROQ_API_KEY``, and ``LLM_ORCH_DEFAULT_TIMEOUT_S``
belong to the library configuration surface. The ``*_TEST_MODEL`` and
``RUN_*_INTEGRATION`` variables remain test-only.
"""

from __future__ import annotations

import math
import os
from collections.abc import Callable, Mapping
from typing import Any

from ..contracts import LLMRequest
from ..errors import ConfigurationError
from ..providers.registry import ProviderRegistry
from .settings import OrchestratorSettings

#: Environment variable for the optional library-level default timeout.
DEFAULT_TIMEOUT_ENV_VAR = "LLM_ORCH_DEFAULT_TIMEOUT_S"

#: Provider names this factory knows how to construct. Anything else fails
#: with a safe ``ConfigurationError`` rather than a silent skip.
_SUPPORTED_PROVIDERS = ("gemini", "groq")

#: A per-provider injected SDK client (or ``None``), keyed by provider name;
#: alternatively a callable receiving the provider name. Tests use this to
#: stay offline; production callers omit it so adapters build real clients.
ClientFactory = Mapping[str, Any] | Callable[[str], Any | None]


def _resolve_env(env: Mapping[str, str] | None) -> Mapping[str, str]:
    """Return the environment mapping to read, without touching ``os`` at import."""
    return env if env is not None else os.environ


def _parse_default_timeout(raw: str | None) -> float | None:
    """Parse the default-timeout variable, or ``None`` when absent/blank.

    Raises a value-free ``ConfigurationError`` naming only the variable when
    the value is not a positive finite number of seconds.
    """
    if raw is None:
        return None
    text = raw.strip()
    if not text:
        return None
    try:
        value = float(text)
    except ValueError:
        raise ConfigurationError(
            f"Environment variable {DEFAULT_TIMEOUT_ENV_VAR} must be a positive number of seconds."
        ) from None
    if not math.isfinite(value) or value <= 0:
        raise ConfigurationError(
            f"Environment variable {DEFAULT_TIMEOUT_ENV_VAR} must be a positive number of seconds."
        )
    return value


def load_settings(
    explicit: OrchestratorSettings | None = None,
    *,
    env: Mapping[str, str] | None = None,
) -> OrchestratorSettings:
    """Build the effective settings from explicit values and the environment.

    Precedence (highest first):

    1. Every field set on ``explicit`` is preserved as-is — "explicit wins".
    2. When ``explicit`` is given but leaves ``default_timeout_s`` as
       ``None``, a valid ``LLM_ORCH_DEFAULT_TIMEOUT_S`` from the environment
       fills exactly that field (via ``model_copy``; the caller's object is
       never mutated).
    3. When ``explicit`` is ``None``, the result carries library defaults for
       everything except ``default_timeout_s``, which comes from the
       environment when present and valid.

    Providers, models, and policies are never merged from the environment:
    the environment carries no structured configuration, only the scalar
    default timeout (credentials are resolved separately by
    :func:`build_registry` and never stored on settings). If ``env`` is
    supplied it is used exclusively; otherwise ``os.environ`` is read here,
    inside this function only. No ``.env`` file is ever loaded automatically.
    """
    source = _resolve_env(env)
    env_timeout = _parse_default_timeout(source.get(DEFAULT_TIMEOUT_ENV_VAR))
    if explicit is None:
        if env_timeout is None:
            return OrchestratorSettings()
        return OrchestratorSettings(default_timeout_s=env_timeout)
    if explicit.default_timeout_s is not None or env_timeout is None:
        return explicit
    return explicit.model_copy(update={"default_timeout_s": env_timeout})


def resolve_timeout_s(
    settings: OrchestratorSettings,
    *,
    provider_name: str | None = None,
    request: LLMRequest | None = None,
    timeout_s: float | None = None,
) -> float | None:
    """Return the effective timeout without performing any I/O.

    Precedence (highest first), preserving existing provider behavior:

    1. Per-call ``timeout_s``.
    2. ``request.timeout_s``.
    3. The named provider's ``ProviderConfig.default_timeout_s``.
    4. The settings-level ``default_timeout_s``.
    5. ``None``, letting the SDK's existing default apply.

    Callers pass the result into the existing ``timeout_s`` seams
    (``RetryEngine.generate``, ``Orchestrator.run``, provider ``generate``).
    """
    if timeout_s is not None:
        return timeout_s
    if request is not None and request.timeout_s is not None:
        return request.timeout_s
    if provider_name is not None:
        for provider in settings.providers:
            if provider.name == provider_name and provider.default_timeout_s is not None:
                return provider.default_timeout_s
    return settings.default_timeout_s


def _resolve_credential(
    *,
    provider_name: str,
    credential_env_var: str | None,
    source: Mapping[str, str],
) -> str | None:
    """Return the credential for one provider, or ``None`` when none is named.

    The value lives only in this call's locals: it is passed straight to the
    provider constructor and never stored on settings. A missing or blank
    value raises a ``ConfigurationError`` naming the provider and the expected
    variable — never the value.
    """
    if credential_env_var is None:
        return None
    raw = source.get(credential_env_var)
    if raw is None or not raw.strip():
        raise ConfigurationError(
            f"Missing credential for provider {provider_name!r}; "
            f"expected environment variable {credential_env_var}."
        )
    return raw


def _injected_client(client_factory: ClientFactory | None, provider_name: str) -> Any | None:
    """Return the injected client for ``provider_name``, or ``None``."""
    if client_factory is None:
        return None
    if isinstance(client_factory, Mapping):
        return client_factory.get(provider_name)
    return client_factory(provider_name)


def build_registry(
    settings: OrchestratorSettings,
    *,
    env: Mapping[str, str] | None = None,
    client_factory: ClientFactory | None = None,
) -> ProviderRegistry:
    """Construct and populate a ``ProviderRegistry`` from ``settings``.

    Each enabled provider in ``settings.providers`` is constructed with its
    environment-resolved credential (see :func:`_resolve_credential`) and an
    optionally injected client, then registered. Disabled providers are
    skipped. Construction only builds provider objects — it makes no model
    calls, creates no network connections (SDK clients are created lazily by
    the adapters), and keeps no global state: every call returns a fresh,
    independent registry.

    Unknown provider names raise a safe ``ConfigurationError``; duplicate
    names surface the registry's own ``ConfigurationError``. If ``env`` is
    supplied it is used exclusively for credential resolution; otherwise
    ``os.environ`` is read here, inside this function only.
    """
    # Lazy, per-branch imports keep ``import llm_orchestrator.config`` free of
    # provider SDKs and let a consumer install only the extra they configure:
    # a Gemini-only setup never imports the Groq adapter (and vice versa).
    # Importing an adapter without its extra raises the adapter's actionable
    # ``ConfigurationError`` naming the extra to install.
    source = _resolve_env(env)
    registry = ProviderRegistry()
    for provider_config in settings.providers:
        if not provider_config.enabled:
            continue
        if provider_config.name not in _SUPPORTED_PROVIDERS:
            raise ConfigurationError(
                f"Unsupported provider {provider_config.name!r}; "
                f"supported providers are {', '.join(_SUPPORTED_PROVIDERS)}."
            )
        api_key = _resolve_credential(
            provider_name=provider_config.name,
            credential_env_var=provider_config.credential_env_var,
            source=source,
        )
        client = _injected_client(client_factory, provider_config.name)
        if provider_config.name == "gemini":
            from ..providers.gemini import GeminiProvider

            registry.register(GeminiProvider(api_key=api_key, client=client))
        else:
            from ..providers.groq import GroqProvider

            registry.register(GroqProvider(api_key=api_key, client=client))
    return registry
