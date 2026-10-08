# Reliability — Retry + Fallback (Phases 6, 11)

`RetryEngine` (`llm_orchestrator.reliability`) retries a failed provider call
against the **same** provider, model, and request. It is provider-agnostic: it
consumes only the generic `LLMProvider` interface, the `RetryPolicy` contract,
and normalized `LLMError` semantics.

```
LLMRequest + ModelConfig
        ↓
RetryEngine.generate(provider, request, model)   ← up to max_attempts tries
        ↓                                            of the SAME candidate
   LLMResponse
```

Retry means *try again*; fallback (a *different* provider/model) is owned by
`FallbackExecutor` in the same package (Phase 11; see `docs/fallback.md`).
The retry engine itself never switches candidates: exhaustion re-raises the
terminal normalized error, and the fallback layer decides what happens next.

## Fallback (Phase 11)

Candidate-level fallback lives beside retry in `llm_orchestrator.reliability`
but stays strictly separated: `FallbackExecutor.run(request, candidates, …)`
attempts the routing-ordered plan in order, running each candidate through the
injected `RetryEngine` and advancing only on fallback-eligible terminal
failures, bounded by `FallbackPolicy.max_fallbacks`. Full semantics, plan
rules, and guarantees are documented in [`docs/fallback.md`](fallback.md).

---

## Policy semantics

`RetryPolicy` (Phase 2 contract, unchanged in Phase 6):

| Field | Meaning | Default |
| --- | --- | --- |
| `max_attempts` | **Total** provider attempts (`1` = one attempt, zero retries) | `1` |
| `initial_backoff_s` | Wait before the second attempt | `0.5` |
| `multiplier` | Exponential growth per retry (≥ 1) | `2.0` |
| `max_backoff_s` | Hard cap on any single wait (≥ initial) | `30.0` |
| `jitter` | `none` \| `full` \| `equal` \| `decorrelated` | `full` |
| `retry_on` | Category override; `None` = use each error's own flag | `None` |

Invalid policies are rejected by Pydantic validation (`max_attempts ≥ 1`,
finite non-negative delays, `multiplier ≥ 1`, `max_backoff_s ≥
initial_backoff_s`), so a dangerous policy cannot silently reach the loop.
The conservative default (`max_attempts=1`) performs no retries unless the
caller opts in.

## Delay formula

For the `n`-th wait (`n = 0` before attempt 2):

```
base = min(max_backoff_s, initial_backoff_s × multiplierⁿ)
```

then the policy's jitter is applied (`none` → `base`; `full` → uniform in
`[0, base]`; `equal` → half fixed plus half jittered; `decorrelated` →
uniform in `[initial_backoff_s, min(max_backoff_s, 3 × previous)]`). The
result is always clamped to `[0, max_backoff_s]`.

A server-provided `retry_after_s` (normalized onto `RateLimitError` by the
adapters) raises the computed wait — `min(max_backoff_s, max(delay,
retry_after_s))` — so the server's minimum is honored without ever exceeding
the policy cap. Missing, negative, or non-finite values are ignored.

## What is (and is not) retried

| Situation | Behavior |
| --- | --- |
| Retryable `LLMError` (`RateLimitError`, `LLMTimeoutError`, `NetworkError`, `ProviderUnavailableError`, …), attempts left | wait, retry same candidate |
| Non-retryable `LLMError` (auth, invalid request, …) | propagate immediately, no sleep |
| `retry_on` set | category membership decides, overriding taxonomy defaults |
| Unknown non-`LLMError` exception | propagate immediately, never retried |
| `asyncio.CancelledError` (during call or sleep) | propagate immediately, never retried |
| Success | return at once; no further attempts |
| Budget exhausted | re-raise the **original** normalized error with `attempt` stamped |

There is no retry loop around programming bugs, and cancellation is never
swallowed.

## Testability

`RetryEngine(policy, *, sleep=..., rng=...)` injects time and randomness:
tests record sleep calls instead of waiting and seed `random.Random` (or use
`jitter=none`) for deterministic delays. Each `generate` call keeps its state
in locals — no module-level mutable state — so concurrent requests never
interfere. The request object is never mutated.

## Intentionally deferred

- Routing/candidate ordering is owned by Phase 10 (`docs/routing.md`); the
  fallback plan consumes that order and never reselects.
- Structured output (Phase 10), observability events (Phase 12),
  configuration loading (Phase 12).
