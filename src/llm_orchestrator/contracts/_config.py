"""Shared Pydantic configuration for contract models.

Contracts are frozen (no accidental mutation during orchestration) and reject
unknown fields (typos and provider-specific leakage fail fast). Immutability is
shallow: nested ``Mapping`` values are not deep-frozen.
"""

from __future__ import annotations

from pydantic import ConfigDict

CONTRACT_MODEL_CONFIG = ConfigDict(frozen=True, extra="forbid")
