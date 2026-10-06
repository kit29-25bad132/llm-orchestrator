"""Provider-agnostic structured-output validation.

Validates normalized ``LLMResponse`` text against caller-owned Pydantic
schemas — no provider SDKs, no model calls, no repair loops:

    from llm_orchestrator.structured import validate_structured_output
"""

from __future__ import annotations

from .pydantic_adapter import validate_structured_output

__all__ = [
    "validate_structured_output",
]
