# Routing & model selection

Status: **Phase 10.** `llm_orchestrator.routing` selects the initial model
candidate deterministically. It consumes the Phase 2 contracts
(`ModelConfig`, `RoutingPolicy` protocol, `RoutingContext`) and the Phase 2
error taxonomy (`ConfigurationError`); it changes no earlier phase.

## What routing is

Given an application request and a set of eligible model candidates,
deterministically select the most appropriate candidate according to explicit
routing policy. It answers:

> **Which model should we attempt first?**

```python
from llm_orchestrator.contracts import LLMRequest, Message, MessageRole, ModelConfig
from llm_orchestrator.routing import ModelRouter, StaticRoutingPolicy

request = LLMRequest(messages=[Message(role=MessageRole.USER, content="Hello")])
candidates = [
    ModelConfig(provider="provider-a", model="model-x"),
    ModelConfig(provider="provider-b", model="model-y"),
]

router = ModelRouter(StaticRoutingPolicy(preferred_provider="provider-a"))
selected = router.select(request, candidates)  # a ModelConfig

# The selected candidate flows into the existing execution path unchanged:
# router.select(...) -> RetryEngine.generate(provider, request, selected)
```

## What routing is not

- **Not retry.** Retry re-attempts the *same* candidate (`RetryEngine`).
- **Not fallback.** Moving to another candidate after a failure belongs to
  `FallbackExecutor` (Phase 11, `docs/fallback.md`). The router returns
  exactly one initial candidate and never executes any of them; the fallback
  layer consumes the routing order.
- **Not health-aware.** No pings, no latency probes, no telemetry. Health
  signals may arrive later via `RoutingContext.metadata`; the static policy
  does not read them.
- **Not stochastic.** No random or weighted selection. Same input always
  yields the same candidate.
- **Not an LLM call.** Selection is ordinary in-memory filtering and sorting.

Separation, end to end:

```text
Routing  -> choose the initial candidate (this phase)
Retry    -> retry the chosen candidate when retryable (Phase 6, unchanged)
Fallback -> move to another candidate after an eligible failure (Phase 11)
```

## Candidates

Candidates are `ModelConfig` (no new candidate type was needed): `provider`
and `model` identity, `enabled`, `capabilities`, optional `pricing`,
`quality_tier` as deterministic priority, and `tags`.

## Policy (`StaticRoutingPolicy`)

A frozen Pydantic model — pure configuration, no control flow. Every field
defaults to "no opinion":

| Field | Effect |
| --- | --- |
| `preferred_provider` / `preferred_model` | Ranking boost only; never filters. A missing preference still selects the best remaining candidate. |
| `allowed_providers` / `allowed_models` | Eligibility filter. |
| `required_model` (`CandidateRef`) | With `allow_substitution=False` (default), only that exact `(provider, model)` is eligible — never silently substituted. With `True`, it ranks first but others stay eligible. |
| `require_structured_output` / `require_tool_calling` / `require_streaming` / `min_context_tokens` | Eligibility filter on exactly the `ModelCapabilities` axes. Additionally, any request carrying `response_schema` requires `structured_output`, since such a request cannot succeed elsewhere. |
| `prefer_lowest_cost` + `estimated_input_tokens` + `estimated_output_tokens` | Ranking by estimated cost. All three are required; otherwise cost is ignored rather than guessed. |

## Selection algorithm

Eligibility first, in order: valid `ModelConfig` → `enabled` → provider
allowlist → model allowlist → explicit-model requirement → required
capabilities (including schema-implied structured output) → minimum context.

Ranking of the eligible, best first:

1. required-model match (only when substitution is allowed),
2. preferred-provider match,
3. preferred-model match,
4. lowest estimated cost (only when cost comparison is active; candidates
   with unknown cost sort after candidates with known cost),
5. highest `quality_tier` (unknown sorts last),
6. `(provider, model)` lexicographic tiebreak — independent of input order.

`StaticRoutingPolicy.select` exposes this ordering through the
`RoutingPolicy` protocol as `tuple[CandidateRef, ...]` (`()` when none is
eligible). `ModelRouter.select` returns the single winner as `ModelConfig`;
`ModelRouter.explain` additionally returns the eligible list, the policy,
and a `SelectionReason` (`sole_eligible`, `explicit_model_match`,
`preferred_provider`, `preferred_model`, `lowest_cost`, `highest_priority`,
`stable_tiebreak`).

## Cost model

`estimated_cost = (input_tokens × input_per_1k + output_tokens × output_per_1k) / 1000`
(`estimate_cost`), using only configured `ModelPricing` and the policy's
explicit token estimates.

- Known before the request: configured prices, capabilities, context needs.
- Never estimated: actual token usage (known only after generation), and no
  token counts are invented — without both estimates, cost does not
  participate in ranking.

## Failure behavior

No eligible candidate raises `ConfigurationError` (category
`configuration`: never retried, never fallen back from) with a safe message
counting considered vs. eligible candidates. No new error class was needed:
a routing miss is operator/programmer configuration, not a provider outage.
Registry misses for the *selected* provider still surface at execution time
via `ProviderNotRegisteredError`, unchanged.

## Complexity and guarantees

Selection is O(n) filtering plus O(n log n) sorting over the candidate set —
negligible next to LLM generation. Deterministic: no I/O, no clocks, no
randomness. Provider-agnostic: the package never names a concrete vendor and
never imports provider SDKs, the registry, the retry engine, or LangGraph —
it runs before all of them.
