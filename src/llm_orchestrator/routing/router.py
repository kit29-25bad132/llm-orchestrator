"""Deterministic initial-candidate selection.

A :class:`ModelRouter` answers one question — *which model should we attempt
first?* — by applying a declarative policy to an explicit candidate set. It
returns exactly one candidate; it never calls providers, never retries, and
never advances to another candidate on failure. Retry stays in
:mod:`llm_orchestrator.reliability`; moving to another candidate after a
failure belongs to a future fallback phase.
"""

from __future__ import annotations

from collections.abc import Sequence
from enum import StrEnum

from pydantic import BaseModel, Field

from ..contracts import LLMRequest, ModelConfig, RoutingContext
from ..contracts._config import CONTRACT_MODEL_CONFIG
from ..errors import ConfigurationError
from .static import StaticRoutingPolicy


class SelectionReason(StrEnum):
    """Why the winner was chosen. One value per ranking dimension."""

    SOLE_ELIGIBLE = "sole_eligible"
    EXPLICIT_MODEL_MATCH = "explicit_model_match"
    PREFERRED_PROVIDER = "preferred_provider"
    PREFERRED_MODEL = "preferred_model"
    LOWEST_COST = "lowest_cost"
    HIGHEST_PRIORITY = "highest_priority"
    STABLE_TIEBREAK = "stable_tiebreak"


class RoutingDecision(BaseModel):
    """The outcome of one routing invocation, for debugging and tests."""

    model_config = CONTRACT_MODEL_CONFIG

    selected: ModelConfig
    eligible: tuple[ModelConfig, ...] = Field(min_length=1)
    policy: StaticRoutingPolicy
    reason: SelectionReason


def _selection_reason(
    policy: StaticRoutingPolicy,
    winner: ModelConfig,
    runner_up: ModelConfig,
) -> SelectionReason:
    """Name the first ranking dimension that separates winner from runner-up."""
    required = policy.required_model
    if (
        required is not None
        and policy.allow_substitution
        and (winner.provider, winner.model) == (required.provider, required.model)
        and (runner_up.provider, runner_up.model) != (required.provider, required.model)
    ):
        return SelectionReason.EXPLICIT_MODEL_MATCH
    if policy.preferred_provider is not None and (
        (winner.provider == policy.preferred_provider)
        != (runner_up.provider == policy.preferred_provider)
    ):
        return SelectionReason.PREFERRED_PROVIDER
    if policy.preferred_model is not None and (
        (winner.model == policy.preferred_model) != (runner_up.model == policy.preferred_model)
    ):
        return SelectionReason.PREFERRED_MODEL
    winner_cost = policy._candidate_cost(winner)
    runner_cost = policy._candidate_cost(runner_up)
    if winner_cost != runner_cost:
        return SelectionReason.LOWEST_COST
    if winner.quality_tier != runner_up.quality_tier:
        return SelectionReason.HIGHEST_PRIORITY
    return SelectionReason.STABLE_TIEBREAK


def _explain(
    policy: StaticRoutingPolicy,
    ordered: tuple[ModelConfig, ...],
) -> RoutingDecision:
    winner = ordered[0]
    if len(ordered) == 1:
        reason = SelectionReason.SOLE_ELIGIBLE
    else:
        reason = _selection_reason(policy, winner, ordered[1])
    return RoutingDecision(selected=winner, eligible=ordered, policy=policy, reason=reason)


class ModelRouter:
    """Selects the initial candidate. Stateless, synchronous, side-effect free."""

    def __init__(self, policy: StaticRoutingPolicy | None = None) -> None:
        self._policy = policy if policy is not None else StaticRoutingPolicy()

    @property
    def policy(self) -> StaticRoutingPolicy:
        """The default policy used when ``select`` is called without one."""
        return self._policy

    def explain(
        self,
        request: LLMRequest,
        candidates: Sequence[ModelConfig],
        *,
        policy: StaticRoutingPolicy | None = None,
        context: RoutingContext | None = None,
    ) -> RoutingDecision:
        """Return the full routing decision (selected + eligible + reason).

        Raises :class:`ConfigurationError` when no candidate is eligible: that
        is a routing/configuration decision, never a provider outage, so it is
        neither retried nor fallen back from.
        """
        active = policy if policy is not None else self._policy
        ordered = active.ordered(request, candidates, context)
        if not ordered:
            raise ConfigurationError(
                "no eligible model candidate for request "
                f"{request.request_id}: considered {len(list(candidates))} candidate(s), "
                "0 eligible",
                request_id=request.request_id,
            )
        return _explain(active, ordered)

    def select(
        self,
        request: LLMRequest,
        candidates: Sequence[ModelConfig],
        *,
        policy: StaticRoutingPolicy | None = None,
        context: RoutingContext | None = None,
    ) -> ModelConfig:
        """Return the single candidate to attempt first.

        Exactly one routing invocation selects exactly one initial candidate.
        """
        return self.explain(request, candidates, policy=policy, context=context).selected


def select_model(
    request: LLMRequest,
    candidates: Sequence[ModelConfig],
    policy: StaticRoutingPolicy,
    context: RoutingContext | None = None,
) -> ModelConfig:
    """Select the initial candidate without keeping a router instance."""
    return ModelRouter(policy).select(request, candidates, context=context)
