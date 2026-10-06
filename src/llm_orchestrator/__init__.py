"""llm-orchestrator: a reusable, provider-agnostic LLM orchestration library.

Phase 1 exposes only package identity. The public contracts, providers, retry,
fallback, routing, structured output, and orchestration layers are implemented
in later phases; this module intentionally exposes no other API yet.
"""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

_DISTRIBUTION_NAME = "llm-orchestrator"

try:
    __version__ = version(_DISTRIBUTION_NAME)
except PackageNotFoundError:  # running from a source tree without installation
    __version__ = "0.0.0"

__all__ = ["__version__"]
