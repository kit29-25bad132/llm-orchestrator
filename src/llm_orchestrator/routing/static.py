"""Declarative static routing policy.

A :class:`StaticRoutingPolicy` is pure configuration: it names *what* the
application prefers (provider, model, capabilities, cost) but contains no
control flow. Eligibility filtering and deterministic ranking live here as
plain in-memory computation over :class:`ModelConfig` candidates — no provider
calls, no network, no randomness, no health checks.

The policy implements the :class:`RoutingPolicy` protocol from
``llm_orchestrator.contracts`` (:meth:`select` returns the eligible candidates
ordered best-first as :class:`CandidateRef`), and additionally exposes
:meth:`ordered`, which returns the same ordering as full configurations for
the single-candidate selector in :mod:`llm_orchestrator.routing.router`.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

from pydantic import BaseModel, Field, field_validator

from ..contracts import CandidateRef, LLMRequest, ModelConfig, RoutingContext
from ..contracts._config import CONTRACT_MODEL_CONFIG

#: Deterministic ordering key for one candidate (lower/earlier is better).
RankKey = tuple[int, int, int, float, float, float, str, str]


def estimate_cost(
    candidate: ModelConfig,
    *,
    input_tokens: int,
    output_tokens: int,
) -> float | None:
    """Estimate the price of one request on ``candidate``.

    ``input_tokens * input_per_1k / 1000 + output_tokens * output_per_1k / 1000``.

    Returns ``None`` when the candidate carries no usable pricing (no
    :class:`ModelPricing`, or either rate missing). Token counts are never
    invented here: callers supply them explicitly (see
    :attr:`StaticRoutingPolicy.estimated_input_tokens`), and actual
    post-generation usage never influences selection.
    """
    pricing = candidate.pricing
    if pricing is None:
        return None
    if pricing.input_per_1k_tokens is None or pricing.output_per_1k_tokens is None:
        return None
    return (
        input_tokens * pricing.input_per_1k_tokens + output_tokens * pricing.output_per_1k_tokens
    ) / 1000.0


class StaticRoutingPolicy(BaseModel):
    """Declarative initial-candidate preferences.

    Every field is a filter or a ranking hint; none triggers I/O. Fields left
    at their defaults mean "no opinion", so the zero-value policy selects the
    lexicographically first enabled candidate deterministically.
    """

    model_config = CONTRACT_MODEL_CONFIG

    # -- identity preferences (ranking only, unless allowlists/required say otherwise)
    preferred_provider: str | None = None
    preferred_model: str | None = None
    allowed_providers: frozenset[str] | None = None
    allowed_models: frozenset[str] | None = None

    # -- explicit model request (filtering unless substitution is allowed)
    required_model: CandidateRef | None = None
    allow_substitution: bool = False

    # -- capability requirements (filtering; exactly the ModelCapabilities axes)
    require_structured_output: bool = False
    require_tool_calling: bool = False
    require_streaming: bool = False
    min_context_tokens: int | None = Field(default=None, gt=0)

    # -- cost preference (ranking only; needs explicit token estimates)
    prefer_lowest_cost: bool = False
    estimated_input_tokens: int | None = Field(default=None, ge=0)
    estimated_output_tokens: int | None = Field(default=None, ge=0)

    @field_validator("preferred_provider", "preferred_model")
    @classmethod
    def _validate_non_blank(cls, value: str | None) -> str | None:
        if value is not None and not value.strip():
            raise ValueError("provider/model preference must not be blank when provided")
        return value

    @field_validator("allowed_providers", "allowed_models")
    @classmethod
    def _validate_allowlist(cls, value: frozenset[str] | None) -> frozenset[str] | None:
        if value is not None:
            if len(value) == 0:
                raise ValueError("allowlists must not be empty when provided")
            if any(not item.strip() for item in value):
                raise ValueError("allowlist entries must not be blank")
        return value

    def _cost_active(self) -> bool:
        """Whether cost participates in ranking.

        Cost comparison requires all three inputs: the policy must ask for it
        and both token estimates must be supplied. Anything less means cost is
        ignored rather than guessed.
        """
        return (
            self.prefer_lowest_cost
            and self.estimated_input_tokens is not None
            and self.estimated_output_tokens is not None
        )

    def _candidate_cost(self, candidate: ModelConfig) -> float | None:
        if not self._cost_active():
            return None
        assert self.estimated_input_tokens is not None
        assert self.estimated_output_tokens is not None
        return estimate_cost(
            candidate,
            input_tokens=self.estimated_input_tokens,
            output_tokens=self.estimated_output_tokens,
        )

    def is_eligible(self, request: LLMRequest, candidate: object) -> bool:
        """Whether ``candidate`` may serve ``request`` under this policy.

        Pure predicate over configuration and request intent. A request that
        carries ``response_schema`` additionally requires the
        ``structured_output`` capability, since such a request cannot succeed
        on a model that cannot produce structured output.
        """
        if not isinstance(candidate, ModelConfig):
            return False
        if not candidate.enabled:
            return False
        if self.allowed_providers is not None and candidate.provider not in self.allowed_providers:
            return False
        if self.allowed_models is not None and candidate.model not in self.allowed_models:
            return False
        if self.required_model is not None and not self.allow_substitution:
            if (
                candidate.provider != self.required_model.provider
                or candidate.model != self.required_model.model
            ):
                return False
        capabilities = candidate.capabilities
        needs_structured = self.require_structured_output or request.response_schema is not None
        if needs_structured and not capabilities.structured_output:
            return False
        if self.require_tool_calling and not capabilities.tool_calling:
            return False
        if self.require_streaming and not capabilities.streaming:
            return False
        if self.min_context_tokens is not None:
            if (
                capabilities.max_context_tokens is None
                or capabilities.max_context_tokens < self.min_context_tokens
            ):
                return False
        return True

    def _rank_key(self, candidate: ModelConfig) -> RankKey:
        """Deterministic ordering key, lower (earlier) is better.

        Dimensions, in precedence order:

        1. required-model match (only discriminates when substitution is allowed;
           otherwise non-matching candidates were already filtered out),
        2. preferred-provider match,
        3. preferred-model match,
        4. estimated cost (only when cost comparison is active; candidates with
           unknown cost sort after candidates with known cost),
        5. quality tier as deterministic priority (higher wins; unknown sorts last),
        6. ``(provider, model)`` lexicographic tiebreak, making the result
           independent of input ordering.
        """
        if self.required_model is not None and self.allow_substitution:
            required_match = (
                0
                if (
                    candidate.provider == self.required_model.provider
                    and candidate.model == self.required_model.model
                )
                else 1
            )
        else:
            required_match = 0
        provider_match = (
            0
            if self.preferred_provider is None or candidate.provider == self.preferred_provider
            else 1
        )
        model_match = (
            0 if self.preferred_model is None or candidate.model == self.preferred_model else 1
        )
        cost = self._candidate_cost(candidate)
        if cost is None:
            cost_key: tuple[float, float] = (1.0, 0.0) if self._cost_active() else (0.0, 0.0)
        else:
            cost_key = (0.0, cost)
        tier = candidate.quality_tier
        tier_key = -float(tier) if tier is not None else math.inf
        return (
            required_match,
            provider_match,
            model_match,
            cost_key[0],
            cost_key[1],
            tier_key,
            candidate.provider,
            candidate.model,
        )

    def ordered(
        self,
        request: LLMRequest,
        candidates: Sequence[ModelConfig],
        context: RoutingContext | None = None,
    ) -> tuple[ModelConfig, ...]:
        """Eligible candidates best-first (empty when none are eligible).

        ``context`` is accepted for protocol symmetry with future adaptive
        policies; static routing does not read it.
        """
        _ = context
        eligible = [c for c in candidates if self.is_eligible(request, c)]
        eligible.sort(key=self._rank_key)
        return tuple(eligible)

    def select(
        self,
        request: LLMRequest,
        candidates: Sequence[ModelConfig],
        context: RoutingContext,
    ) -> tuple[CandidateRef, ...]:
        """Order eligible candidates best-first as lightweight references.

        Implements the :class:`RoutingPolicy` protocol. Empty input or no
        eligible candidate yields ``()``; surfacing that as an error is the
        selector's job (see :mod:`llm_orchestrator.routing.router`).
        """
        return tuple(
            CandidateRef(provider=c.provider, model=c.model)
            for c in self.ordered(request, candidates, context)
        )
