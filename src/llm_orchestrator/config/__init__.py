"""Secure, typed, injectable configuration.

Applications describe their setup once as an :class:`OrchestratorSettings`
(explicit values, no environment reads), optionally layer the documented
environment variables over it with :func:`load_settings`, and build an
independent :class:`ProviderRegistry` with :func:`build_registry`:

    from llm_orchestrator.config import (
        OrchestratorSettings,
        build_registry,
        load_settings,
    )

    settings = load_settings(explicit=OrchestratorSettings(...), env=os.environ)
    registry = build_registry(settings, env=os.environ)

Importing this package reads no environment, requires no credentials, and
loads no provider SDKs. See ``docs/configuration.md``.
"""

from __future__ import annotations

from .loader import build_registry, load_settings, resolve_timeout_s
from .settings import OrchestratorSettings

__all__ = [
    "OrchestratorSettings",
    "build_registry",
    "load_settings",
    "resolve_timeout_s",
]
