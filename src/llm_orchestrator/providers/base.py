"""The provider abstraction.

Every LLM provider (Gemini, Groq, and any future vendor) implements this
protocol. The abstraction is intentionally tiny: identity, one generation
operation, and lifecycle close. It speaks only in Phase 2 contracts, so SDK
types can never leak past this boundary.

Why a ``Protocol`` rather than an ``ABC``:

- Structural typing keeps providers loosely coupled to the library and makes
  test doubles trivial (no inheritance required).
- There is no shared behavior to inherit yet; a concrete abstract base would be
  an empty shell. A shared base can be introduced later if real shared behavior
  appears, without changing this contract.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from ..contracts import LLMRequest, LLMResponse, ModelConfig


@runtime_checkable
class LLMProvider(Protocol):
    """A provider of one or more LLM models.

    Implementations translate :class:`LLMRequest` intent into a vendor call and
    return a normalized :class:`LLMResponse`. They must not perform retries,
    fallbacks, or routing; those belong to the orchestration/reliability layers.

    The ``provider_id`` attribute is a stable, opaque identity string supplied
    by the implementation (for example a vendor name). It is deliberately not an
    enum, so a new provider can be added without editing core code.
    """

    provider_id: str

    async def generate(
        self,
        request: LLMRequest,
        model: ModelConfig,
        *,
        timeout_s: float | None = None,
    ) -> LLMResponse:
        """Perform exactly one generation attempt against ``model``.

        ``LLMRequest`` carries intent only and never names a provider or model,
        so the target is supplied explicitly. ``timeout_s`` is the effective
        deadline the caller wants enforced; when provided it overrides any
        request-level timeout.

        Implementations must raise a normalized :class:`LLMError` subtype (from
        ``llm_orchestrator.errors``) on failure, never a raw SDK error.
        """
        ...

    async def aclose(self) -> None:
        """Release any resources held by the provider."""
        ...
