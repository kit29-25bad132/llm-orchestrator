"""Provider abstraction layer.

    from llm_orchestrator.providers import LLMProvider, ProviderRegistry

Concrete providers (Gemini, Groq, ...) are implemented in later phases; this
package defines only the boundary they plug into.
"""

from __future__ import annotations

from .base import LLMProvider
from .registry import ProviderRegistry

__all__ = [
    "LLMProvider",
    "ProviderRegistry",
]
