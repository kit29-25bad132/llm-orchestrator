"""Deterministic routing and model selection.

Routing answers *which model should we attempt first?* It never calls
providers, retries, or advances to another candidate on failure:

    from llm_orchestrator.routing import ModelRouter, StaticRoutingPolicy

    router = ModelRouter(StaticRoutingPolicy(preferred_provider="provider-a"))
    selected = router.select(request, candidates)  # a ModelConfig
"""

from __future__ import annotations

from .router import ModelRouter, RoutingDecision, SelectionReason, select_model
from .static import StaticRoutingPolicy, estimate_cost

__all__ = [
    "ModelRouter",
    "RoutingDecision",
    "SelectionReason",
    "StaticRoutingPolicy",
    "estimate_cost",
    "select_model",
]
