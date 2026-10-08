# Fallback execution (Phase 11)

`FallbackExecutor` (`llm_orchestrator.reliability`) moves execution to another
candidate only after the current candidate reaches a *terminal* failure inside
`RetryEngine`. It is a reusable service around the retry engine — not a
LangGraph node — so the single-candidate graph (`orchestration/`) is unchanged.

```
Routing  -> which candidate to try first (an ordered ModelConfig list)
Retry    -> re-attempt the same candidate (RetryEngine.generate)
Fallback -> move to the next candidate (this phase, on eligible failure)
```

```
Candidate A
    ↓
RetryEngine (attempt 1..N on A only)
    ↓
terminal error
    ↓
fallback eligible? -- no --> OrchestrationResult(failed, last error + history)
    |
    yes (and budget remains)
    ↓
Candidate B
    ↓
RetryEngine (attempt 1..N on B only)
    ↓
terminal success/failure
```

## API

```python
from llm_orchestrator.reliability import FallbackExecutor

executor = FallbackExecutor(registry, retry_engine, fallback_policy)
result = await executor.run(request, candidates, initial=selected, policy=policy)
```

- `candidates` is the routing-ordered set; `initial` (default: the first
  candidate, i.e. the Phase 10 selection) is always attempted first.
- `policy` per call overrides the executor default.
- `timeout_s` passes through to every candidate unchanged.

## Plan

Execution order is deterministic: initial candidate first, duplicates collapsed
to their first `(provider, model)` occurrence, an explicit
`FallbackPolicy.candidates` list acting as an allowlist that preserves routing
order (routing owns ordering, never this layer), `cross_provider=False` keeping
only the initial provider, and truncation to `1 + max_fallbacks` entries.

`max_attempts` (retry) and `max_fallbacks` (fallback) are different limits:

| Limit | Meaning |
| --- | --- |
| `max_attempts` | Total provider attempts on **one** candidate (`1` = no retry) |
| `max_fallbacks` | Candidate **transitions** (`0` = initial candidate only) |

So `max_fallbacks=2` allows at most `A → B → C`. The initial candidate is not
a fallback: `A → B` is 1 fallback.

## Eligibility

The normalized `LLMError.fallback_eligible` flag decides, exactly as the error
taxonomy defines it (see `docs/errors.md`). An explicit
`FallbackPolicy.fallback_on` category set overrides the taxonomy, mirroring how
`RetryPolicy.retry_on` works. Structured-output validation failures additionally
honor `include_validation_failures`. Registry misses (`ConfigurationError`)
are terminal per the taxonomy and stop the run.

## Guarantees

- Success returns immediately; later candidates are never touched.
- All-candidate failure returns the **last** candidate's terminal normalized
  error with the complete per-call `AttemptRecord` history
  (`A attempt 1, A attempt 2, B attempt 1, …`), so no failure is discarded.
- The same `request_id` is used for every candidate; each record keeps its own
  provider/model/attempt metadata.
- Cancellation (`asyncio.CancelledError`) and non-`LLMError` programming errors
  propagate untouched and never trigger a transition.
- The layer never sleeps, never computes backoff, never branches on a vendor,
  never imports a provider SDK, and never calls the router: exactly one retry
  implementation (`RetryEngine.generate`) exists.

## What fallback is not

- Not routing (no `select`, no reordering, no health signals).
- Not retry (no attempt loop, no backoff/jitter, no `retry-after` handling).
- Not a LangGraph change (the graph stays `START → execute_model → END`).
- Not cost accounting (but it stops on success and honors the depth bound, so
  no unnecessary invocation ever runs).
