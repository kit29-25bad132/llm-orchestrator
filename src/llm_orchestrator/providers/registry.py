"""Provider registry.

A small, explicit map from provider id to provider instance. The registry knows
nothing about concrete providers: it never branches on ``"gemini"``/``"groq"``
and performs no environment or SDK setup. Callers construct providers and
register them; the registry only resolves them by id.

Lifecycle: the registry is a mutable, non-thread-safe structure intended to be
populated during application setup and read during request handling. No locking
is provided because Phase 3 has no demonstrated concurrent-mutation need.
"""

from __future__ import annotations

from ..errors import ConfigurationError, ProviderNotRegisteredError
from .base import LLMProvider


class ProviderRegistry:
    """Resolves provider instances by :attr:`LLMProvider.provider_id`."""

    def __init__(self) -> None:
        self._providers: dict[str, LLMProvider] = {}

    def register(self, provider: LLMProvider, *, replace: bool = False) -> None:
        """Register ``provider``.

        Duplicate ids are rejected unless ``replace`` is true, so accidental
        configuration mistakes fail loudly rather than silently overwriting.
        """
        if not isinstance(provider, LLMProvider):
            raise TypeError(
                "provider must implement the LLMProvider protocol (provider_id, generate, aclose)"
            )
        provider_id = provider.provider_id
        if not provider_id or not provider_id.strip():
            raise ConfigurationError("provider.provider_id must not be blank")
        if provider_id in self._providers and not replace:
            raise ConfigurationError(
                f"provider {provider_id!r} is already registered; pass replace=True to override it"
            )
        self._providers[provider_id] = provider

    def unregister(self, provider_id: str) -> None:
        """Remove a registered provider."""
        if provider_id not in self._providers:
            raise ProviderNotRegisteredError(
                f"provider {provider_id!r} is not registered", provider=provider_id
            )
        del self._providers[provider_id]

    def get(self, provider_id: str) -> LLMProvider:
        """Return the provider registered under ``provider_id``.

        Raises :class:`ProviderNotRegisteredError` rather than returning
        ``None``, so a missing provider cannot silently become an ``AttributeError``
        deeper in the call stack.
        """
        try:
            return self._providers[provider_id]
        except KeyError:
            raise ProviderNotRegisteredError(
                f"provider {provider_id!r} is not registered", provider=provider_id
            ) from None

    def has(self, provider_id: str) -> bool:
        """Whether a provider is registered under ``provider_id``."""
        return provider_id in self._providers

    def provider_ids(self) -> tuple[str, ...]:
        """Registered provider ids, sorted for deterministic output."""
        return tuple(sorted(self._providers))

    def __len__(self) -> int:
        return len(self._providers)

    def __contains__(self, provider_id: object) -> bool:
        return provider_id in self._providers
