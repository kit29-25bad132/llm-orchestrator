"""Selection behavior of StaticRoutingPolicy + ModelRouter.

Offline and deterministic throughout: no providers are constructed, no network
is touched, and every expectation pins the documented ranking/filtering order.
"""

from __future__ import annotations

import itertools
from typing import Any

import pytest
from pydantic import ValidationError

from llm_orchestrator.contracts import (
    CandidateRef,
    LLMRequest,
    Message,
    MessageRole,
    ModelCapabilities,
    ModelConfig,
    ModelPricing,
    RoutingContext,
)
from llm_orchestrator.errors import ConfigurationError, ErrorCategory
from llm_orchestrator.routing import (
    ModelRouter,
    RoutingDecision,
    SelectionReason,
    StaticRoutingPolicy,
    estimate_cost,
    select_model,
)


def make_request() -> LLMRequest:
    return LLMRequest(messages=[Message(role=MessageRole.USER, content="hello")])


def cfg(provider: str, model: str, **kwargs: Any) -> ModelConfig:
    return ModelConfig(provider=provider, model=model, **kwargs)


REQUEST = make_request()
DEFAULT_POLICY = StaticRoutingPolicy()


# --- candidates: one / many / duplicates / invalid / ordering ---


def test_single_candidate_is_selected() -> None:
    selected = select_model(REQUEST, [cfg("b", "m")], DEFAULT_POLICY)
    assert (selected.provider, selected.model) == ("b", "m")


def test_default_policy_prefers_lexicographic_order() -> None:
    candidates = [cfg("b", "m2"), cfg("a", "m1"), cfg("b", "m1")]
    assert (select_model(REQUEST, candidates, DEFAULT_POLICY).model) == "m1"
    assert (select_model(REQUEST, candidates, DEFAULT_POLICY).provider) == "a"


def test_duplicate_candidates_select_without_error() -> None:
    candidates = [cfg("a", "m"), cfg("a", "m")]
    selected = select_model(REQUEST, candidates, DEFAULT_POLICY)
    assert (selected.provider, selected.model) == ("a", "m")


def test_non_model_entries_are_filtered_not_fatal() -> None:
    candidates = [cfg("a", "m"), "not-a-candidate"]  # type: ignore[list-item]
    assert select_model(REQUEST, candidates, DEFAULT_POLICY).model == "m"


def test_all_invalid_candidates_raise_configuration_error() -> None:
    with pytest.raises(ConfigurationError):
        select_model(REQUEST, ["junk"], DEFAULT_POLICY)  # type: ignore[list-item]


def test_ordered_returns_stable_best_first_ordering() -> None:
    policy = StaticRoutingPolicy()
    candidates = [cfg("b", "m"), cfg("a", "m")]
    first = policy.ordered(REQUEST, candidates)
    second = policy.ordered(REQUEST, list(reversed(candidates)))
    assert [(c.provider, c.model) for c in first] == [("a", "m"), ("b", "m")]
    assert first == second


def test_select_returns_protocol_compatible_references() -> None:
    refs = DEFAULT_POLICY.select(REQUEST, [cfg("b", "m"), cfg("a", "m")], RoutingContext())
    assert refs == (CandidateRef(provider="a", model="m"), CandidateRef(provider="b", model="m"))


def test_select_empty_when_nothing_eligible() -> None:
    refs = DEFAULT_POLICY.select(REQUEST, [cfg("a", "m", enabled=False)], RoutingContext())
    assert refs == ()


# --- explicit model requests ---


def test_explicit_required_model_is_selected() -> None:
    policy = StaticRoutingPolicy(required_model=CandidateRef(provider="b", model="m2"))
    candidates = [cfg("a", "m1"), cfg("b", "m2")]
    assert select_model(REQUEST, candidates, policy).model == "m2"


def test_explicit_model_missing_raises_without_substitution() -> None:
    policy = StaticRoutingPolicy(required_model=CandidateRef(provider="b", model="ghost"))
    with pytest.raises(ConfigurationError):
        select_model(REQUEST, [cfg("a", "m1")], policy)


def test_explicit_model_disabled_raises_without_substitution() -> None:
    policy = StaticRoutingPolicy(required_model=CandidateRef(provider="a", model="m1"))
    with pytest.raises(ConfigurationError):
        select_model(REQUEST, [cfg("a", "m1", enabled=False)], policy)


def test_explicit_request_never_silently_substitutes() -> None:
    policy = StaticRoutingPolicy(
        required_model=CandidateRef(provider="b", model="ghost"),
        allow_substitution=False,
    )
    with pytest.raises(ConfigurationError):
        select_model(REQUEST, [cfg("a", "m1"), cfg("b", "m2")], policy)


def test_allowed_substitution_ranks_exact_match_first() -> None:
    policy = StaticRoutingPolicy(
        required_model=CandidateRef(provider="b", model="m2"),
        allow_substitution=True,
    )
    candidates = [cfg("a", "m1"), cfg("b", "m2")]
    decision = ModelRouter(policy).explain(REQUEST, candidates)
    assert (decision.selected.provider, decision.selected.model) == ("b", "m2")
    assert len(decision.eligible) == 2
    assert decision.reason is SelectionReason.EXPLICIT_MODEL_MATCH


# --- provider preference (not fallback) ---


def test_preferred_provider_wins() -> None:
    policy = StaticRoutingPolicy(preferred_provider="b")
    assert select_model(REQUEST, [cfg("a", "m"), cfg("b", "m")], policy).provider == "b"


def test_preferred_provider_unavailable_falls_back_to_ranking_not_error() -> None:
    policy = StaticRoutingPolicy(preferred_provider="ghost")
    selected = select_model(REQUEST, [cfg("b", "m2"), cfg("a", "m1")], policy)
    assert (selected.provider, selected.model) == ("a", "m1")


def test_nonpreferred_provider_remains_eligible() -> None:
    policy = StaticRoutingPolicy(preferred_provider="b")
    decision = ModelRouter(policy).explain(REQUEST, [cfg("a", "m"), cfg("b", "m")])
    assert [(c.provider, c.model) for c in decision.eligible] == [("b", "m"), ("a", "m")]


def test_provider_preference_selects_exactly_one_candidate() -> None:
    """Preference reorders; it never executes a second candidate on failure."""
    policy = StaticRoutingPolicy(preferred_provider="b")
    router = ModelRouter(policy)
    selected = router.select(REQUEST, [cfg("a", "m"), cfg("b", "m")])
    assert isinstance(selected, ModelConfig)
    assert (selected.provider, selected.model) == ("b", "m")


def test_preferred_model_beats_same_provider_default() -> None:
    policy = StaticRoutingPolicy(preferred_provider="a", preferred_model="m2")
    candidates = [cfg("a", "m1"), cfg("a", "m2")]
    assert select_model(REQUEST, candidates, policy).model == "m2"


# --- priority via quality_tier ---


def test_highest_quality_tier_wins() -> None:
    candidates = [cfg("a", "low", quality_tier=1), cfg("a", "high", quality_tier=9)]
    assert select_model(REQUEST, candidates, DEFAULT_POLICY).model == "high"


def test_unranked_quality_sorts_after_ranked() -> None:
    candidates = [cfg("a", "unranked"), cfg("a", "ranked", quality_tier=0)]
    assert select_model(REQUEST, candidates, DEFAULT_POLICY).model == "ranked"


def test_equal_tier_breaks_tie_deterministically() -> None:
    candidates = [cfg("b", "m", quality_tier=3), cfg("a", "m", quality_tier=3)]
    assert select_model(REQUEST, candidates, DEFAULT_POLICY).provider == "a"


# --- cost-aware routing ---


def priced(provider: str, model: str, *, inp: float, out: float) -> ModelConfig:
    return cfg(
        provider, model, pricing=ModelPricing(input_per_1k_tokens=inp, output_per_1k_tokens=out)
    )


def cost_policy() -> StaticRoutingPolicy:
    return StaticRoutingPolicy(
        prefer_lowest_cost=True, estimated_input_tokens=1000, estimated_output_tokens=500
    )


def test_estimate_cost_math_is_transparent() -> None:
    candidate = priced("a", "m", inp=2.0, out=4.0)
    assert estimate_cost(candidate, input_tokens=1000, output_tokens=500) == pytest.approx(4.0)


def test_estimate_cost_none_without_pricing() -> None:
    assert estimate_cost(cfg("a", "m"), input_tokens=1000, output_tokens=500) is None
    partial = cfg("a", "m", pricing=ModelPricing(input_per_1k_tokens=1.0))
    assert estimate_cost(partial, input_tokens=1000, output_tokens=500) is None


def test_lower_estimated_cost_wins_when_requested() -> None:
    candidates = [priced("a", "pricey", inp=10.0, out=10.0), priced("a", "cheap", inp=1.0, out=1.0)]
    assert select_model(REQUEST, candidates, cost_policy()).model == "cheap"


def test_cost_tie_is_deterministic() -> None:
    candidates = [priced("b", "m", inp=1.0, out=1.0), priced("a", "m", inp=1.0, out=1.0)]
    assert select_model(REQUEST, candidates, cost_policy()).provider == "a"


def test_known_cost_sorts_before_unknown_cost() -> None:
    candidates = [cfg("a", "unknown"), priced("b", "known", inp=50.0, out=50.0)]
    assert select_model(REQUEST, candidates, cost_policy()).model == "known"


def test_cost_ignored_without_token_estimates() -> None:
    policy = StaticRoutingPolicy(prefer_lowest_cost=True)
    candidates = [priced("b", "cheap", inp=1.0, out=1.0), priced("a", "pricey", inp=9.0, out=9.0)]
    # No estimates supplied: no invented token counts; lexicographic order decides.
    assert select_model(REQUEST, candidates, policy).provider == "a"


def test_cost_ignored_when_policy_does_not_ask_for_it() -> None:
    candidates = [priced("b", "cheap", inp=1.0, out=1.0), priced("a", "pricey", inp=9.0, out=9.0)]
    assert select_model(REQUEST, candidates, DEFAULT_POLICY).provider == "a"


# --- capability filtering ---


def capable(provider: str, model: str, **caps: Any) -> ModelConfig:
    return cfg(provider, model, capabilities=ModelCapabilities(**caps))


def test_required_capability_satisfied() -> None:
    policy = StaticRoutingPolicy(require_tool_calling=True)
    candidates = [capable("a", "plain"), capable("b", "tooled", tool_calling=True)]
    assert select_model(REQUEST, candidates, policy).provider == "b"


def test_missing_capability_filtered() -> None:
    policy = StaticRoutingPolicy(require_structured_output=True)
    decision = ModelRouter(policy).explain(
        REQUEST, [capable("a", "plain"), capable("b", "schemad", structured_output=True)]
    )
    assert [(c.provider, c.model) for c in decision.eligible] == [("b", "schemad")]


def test_no_capable_candidate_raises() -> None:
    policy = StaticRoutingPolicy(require_streaming=True)
    with pytest.raises(ConfigurationError):
        select_model(REQUEST, [capable("a", "plain")], policy)


def test_structured_request_requires_capability() -> None:
    schema_request = LLMRequest(
        messages=[Message(role=MessageRole.USER, content="hi")],
        response_schema={"type": "object"},
    )
    candidates = [capable("a", "plain"), capable("b", "schemad", structured_output=True)]
    assert select_model(schema_request, candidates, DEFAULT_POLICY).provider == "b"
    with pytest.raises(ConfigurationError):
        select_model(schema_request, [capable("a", "plain")], DEFAULT_POLICY)


def test_min_context_filters_small_windows() -> None:
    policy = StaticRoutingPolicy(min_context_tokens=128_000)
    candidates = [
        capable("a", "small", max_context_tokens=8_000),
        capable("b", "large", max_context_tokens=200_000),
        capable("c", "unknown"),
    ]
    decision = ModelRouter(policy).explain(REQUEST, candidates)
    assert [(c.provider, c.model) for c in decision.eligible] == [("b", "large")]


def test_allowlist_and_enabled_filtering() -> None:
    policy = StaticRoutingPolicy(allowed_providers=frozenset({"a"}))
    candidates = [cfg("b", "m"), cfg("a", "off", enabled=False), cfg("a", "on")]
    decision = ModelRouter(policy).explain(REQUEST, candidates)
    assert [(c.provider, c.model) for c in decision.eligible] == [("a", "on")]
    policy_models = StaticRoutingPolicy(allowed_models=frozenset({"m2"}))
    assert select_model(REQUEST, [cfg("a", "m1"), cfg("a", "m2")], policy_models).model == "m2"


# --- determinism ---


def test_same_input_same_candidate() -> None:
    candidates = [cfg("b", "m2"), cfg("a", "m1", quality_tier=2)]
    policy = StaticRoutingPolicy(preferred_provider="a")
    first = select_model(REQUEST, candidates, policy)
    for _ in range(5):
        assert select_model(REQUEST, candidates, policy) == first


def test_permutations_select_same_identity() -> None:
    candidates = [cfg("b", "m2"), cfg("a", "m1"), cfg("c", "m3", quality_tier=5)]
    expected = select_model(REQUEST, candidates, cost_policy())
    for perm in itertools.permutations(candidates):
        selected = select_model(REQUEST, list(perm), cost_policy())
        assert (selected.provider, selected.model) == (expected.provider, expected.model)


def test_decision_reason_is_stable() -> None:
    policy = StaticRoutingPolicy(preferred_provider="a")
    candidates = [cfg("b", "m"), cfg("a", "m")]
    reasons = {ModelRouter(policy).explain(REQUEST, candidates).reason for _ in range(3)}
    assert reasons == {SelectionReason.PREFERRED_PROVIDER}


# --- errors ---


def test_no_eligible_candidate_is_configuration_error() -> None:
    with pytest.raises(ConfigurationError) as exc_info:
        select_model(REQUEST, [], DEFAULT_POLICY)
    err = exc_info.value
    assert err.category is ErrorCategory.CONFIGURATION
    assert err.retryable is False
    assert err.fallback_eligible is False


def test_error_message_is_safe_and_counts_candidates() -> None:
    request = make_request()
    with pytest.raises(ConfigurationError) as exc_info:
        select_model(request, [cfg("a", "m", enabled=False)], DEFAULT_POLICY)
    message = str(exc_info.value)
    assert exc_info.value.safe_message == message
    assert "0 eligible" in message
    assert request.request_id in message


def test_error_carries_request_id() -> None:
    request = make_request()
    with pytest.raises(ConfigurationError) as exc_info:
        select_model(request, [], DEFAULT_POLICY)
    assert exc_info.value.request_id == request.request_id


def test_no_provider_sdk_exception_leaks() -> None:
    """Routing failures are always ConfigurationError, never provider errors."""
    try:
        select_model(REQUEST, [], DEFAULT_POLICY)
    except ConfigurationError:
        pass
    except Exception as exc:  # pragma: no cover - documents the guarantee
        pytest.fail(f"unexpected exception type leaked: {type(exc).__name__}")


# --- policy validation ---


def test_blank_preferences_rejected() -> None:
    with pytest.raises(ValidationError):
        StaticRoutingPolicy(preferred_provider="  ")
    with pytest.raises(ValidationError):
        StaticRoutingPolicy(preferred_model="")


def test_empty_allowlist_rejected() -> None:
    with pytest.raises(ValidationError):
        StaticRoutingPolicy(allowed_providers=frozenset())


def test_negative_token_estimates_rejected() -> None:
    with pytest.raises(ValidationError):
        StaticRoutingPolicy(estimated_input_tokens=-1)
    with pytest.raises(ValidationError):
        StaticRoutingPolicy(min_context_tokens=0)


# --- decision object + public API ---


def test_explain_reports_eligible_policy_and_reason() -> None:
    policy = StaticRoutingPolicy(preferred_provider="a")
    decision = ModelRouter(policy).explain(REQUEST, [cfg("b", "m"), cfg("a", "m")])
    assert isinstance(decision, RoutingDecision)
    assert (decision.selected.provider, decision.selected.model) == ("a", "m")
    assert len(decision.eligible) == 2
    assert decision.policy == policy
    assert decision.reason is SelectionReason.PREFERRED_PROVIDER


def test_reason_variants() -> None:
    assert (
        ModelRouter(StaticRoutingPolicy()).explain(REQUEST, [cfg("a", "m")]).reason
        is SelectionReason.SOLE_ELIGIBLE
    )
    assert (
        ModelRouter(StaticRoutingPolicy(preferred_model="m2"))
        .explain(REQUEST, [cfg("a", "m1"), cfg("a", "m2")])
        .reason
        is SelectionReason.PREFERRED_MODEL
    )
    assert (
        ModelRouter(cost_policy())
        .explain(REQUEST, [priced("a", "p", inp=9.0, out=9.0), priced("a", "c", inp=1.0, out=1.0)])
        .reason
        is SelectionReason.LOWEST_COST
    )
    assert (
        ModelRouter(StaticRoutingPolicy())
        .explain(REQUEST, [cfg("a", "lo", quality_tier=1), cfg("a", "hi", quality_tier=2)])
        .reason
        is SelectionReason.HIGHEST_PRIORITY
    )
    assert (
        ModelRouter(StaticRoutingPolicy()).explain(REQUEST, [cfg("b", "m"), cfg("a", "m")]).reason
        is SelectionReason.STABLE_TIEBREAK
    )


def test_router_default_policy_used_when_none_given() -> None:
    router = ModelRouter()
    assert router.policy == StaticRoutingPolicy()
    assert router.select(REQUEST, [cfg("a", "m")]) == cfg("a", "m")


def test_router_per_call_policy_override() -> None:
    router = ModelRouter(StaticRoutingPolicy(preferred_provider="a"))
    override = StaticRoutingPolicy(preferred_provider="b")
    selected = router.select(REQUEST, [cfg("a", "m"), cfg("b", "m")], policy=override)
    assert selected.provider == "b"


def test_selected_candidate_identity_is_from_input() -> None:
    candidates = [cfg("b", "m"), cfg("a", "m")]
    selected = select_model(REQUEST, candidates, DEFAULT_POLICY)
    assert any(selected is original for original in candidates)


def test_reviewiq_shaped_preference_is_representable() -> None:
    """Preferred-vendor + cost/priority selection without hardcoding vendors in src."""
    gemini_flash = priced("gemini", "flash", inp=1.0, out=2.0)
    gemini_pro = ModelConfig(
        provider="gemini",
        model="pro",
        pricing=ModelPricing(input_per_1k_tokens=5.0, output_per_1k_tokens=10.0),
        quality_tier=10,
    )
    groq_fast = priced("groq", "fast", inp=0.5, out=0.5)
    candidates = [groq_fast, gemini_pro, gemini_flash]
    policy = StaticRoutingPolicy(
        preferred_provider="gemini",
        prefer_lowest_cost=True,
        estimated_input_tokens=1000,
        estimated_output_tokens=1000,
    )
    decision = ModelRouter(policy).explain(REQUEST, candidates)
    # Cheapest preferred-vendor model wins; the other vendor stays eligible
    # (preference is initial selection, not fallback execution).
    assert (decision.selected.provider, decision.selected.model) == ("gemini", "flash")
    assert ("groq", "fast") in [(c.provider, c.model) for c in decision.eligible]
