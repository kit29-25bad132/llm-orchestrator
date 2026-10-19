# Orchestration (LangGraph workflow)

`Orchestrator` (`llm_orchestrator.orchestration`, requires the `langgraph`
extra — `pip install "llm-orchestrator[langgraph]"`; without it the import
raises an actionable `ConfigurationError` naming the command) runs one model
candidate
through the existing infrastructure inside a deterministic LangGraph workflow.
LangGraph owns workflow state and transitions; every other concern stays where
it already lives: domain semantics in `contracts`, failure semantics in
`errors`, retries in `RetryEngine`, SDK translation in providers.

```
LLMRequest + ModelConfig
        ↓
LangGraph state (OrchestrationState)
        ↓
execute_model ──→ provider resolved from ProviderRegistry
        ↓
RetryEngine.generate(provider, request, model)   ← retries live here, nowhere else
        ↓
Provider.generate() ──→ LLMResponse ──→ OrchestrationResult(succeeded, response)
                   └──── LLMError ─────→ OrchestrationResult(failed, ErrorInfo)
```

Graph topology (linear by design — branching arrives with fallback):

```
START → execute_model → END
```

---

## Why LangGraph exists here

The graph is the explicit orchestration boundary: typed per-invocation state,
one deterministic transition, a compiled workflow reused across runs, and a
seam where future nodes (fallback, validation) and conditional edges will
attach. It does not wrap a function for ceremony — the state model is what
later phases will extend.

## State design

`OrchestrationState` (a `TypedDict` in `state.py`) threads domain contracts
through the graph without redefining them:

| Field | Role |
| --- | --- |
| `request` / `model` / `timeout_s` | Caller inputs; passed through untouched |
| `response` / `error` | Terminal outcome (exactly one set); the live `LLMError` stays available for future branching |
| `attempts` | Terminal `AttemptRecord`(s) for future observability |
| `result` | The domain-level `OrchestrationResult` |

No `GraphRequest`/`GraphResponse` duplicates exist. State is built fresh per
`run()` (`initial_state`) and never shared between invocations.

## Node responsibilities

`execute_model` (`nodes.py`) — the only node:

1. Resolves the provider from the injected `ProviderRegistry` via
   `model.provider` (a miss is a normalized `ProviderNotRegisteredError`
   failure, not a crash).
2. Invokes `RetryEngine.generate` with the same request, model, and timeout.
3. On success records the response, a terminal success attempt record, and a
   success `OrchestrationResult`.
4. On `LLMError` records the error, a terminal failure record, and a failure
   `OrchestrationResult` built with `ErrorInfo.from_error`.
5. Lets anything else — programming bugs, `CancelledError` — propagate.

It never imports a provider SDK, never sleeps, never loops over attempts,
never switches candidates. Providers are never instantiated or closed here;
their lifecycle stays with whoever registered them.

## Retry integration

Retries are delegated wholesale: the node makes exactly one
`RetryEngine.generate` call, so backoff, jitter, `retry_after_s`, budgets,
and attempt stamping all behave exactly as Phase 6 defines. An attempt-count
record is derived from counted provider calls (success) or the engine-stamped
`error.attempt` (failure) — retry internals are not duplicated or exposed.

## Error handling

Only normalized `LLMError` becomes a failure result, preserving category,
flags, provider, model, attempt, status, and sanitized message. Unexpected
exceptions escape the graph (they indicate bugs, not provider outcomes).
Registry misses are already `LLMError`, so they follow the same path.

## Concurrency and cancellation

- The graph is compiled once (`build_orchestration_graph`) and invoked per
  request with fresh state; concurrent and sequential runs cannot contaminate
  each other (covered by isolation tests).
- Execution is async-only (`await orchestrator.run(...)`); there is no sync
  wrapper and no `asyncio.run` inside the library.
- Outer cancellation surfaces as `asyncio.CancelledError`. When the node body
  itself raises cancellation, LangGraph reports `NodeCancelledError` wrapping
  it; `Orchestrator.run` unwraps that case so callers always observe standard
  cancellation semantics. Cancellation never becomes an `OrchestrationResult`.

## API

```python
from llm_orchestrator.orchestration import Orchestrator

orchestrator = Orchestrator(registry, retry_engine)  # graph compiled once
result = await orchestrator.run(request, model, timeout_s=30.0)
```

`Orchestrator` is the single public entry point; `build_orchestration_graph`,
`execute_model`, `OrchestrationState`, and `initial_state` are available for
advanced use and testing.

## What LangGraph does NOT do yet

```text
Fallback: owned by FallbackExecutor (Phase 11, docs/fallback.md) as a reusable
  service around RetryEngine — the graph itself stays single-candidate by design
Routing: deferred (the candidate arrives as an explicit ModelConfig)
Structured output: deferred (no validation nodes)
Observability: deferred (attempt records are preserved for it)
Multi-agent behavior: never planned (deterministic workflow only)
```

Future extension points: sibling nodes beside `execute_model`, conditional
edges on `error`/validation outcome, and the accumulated `error`/`attempts`
state they will consume.
