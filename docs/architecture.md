# llm-orchestrator — Architecture

Status: **Phase 0 (architecture and foundation only).** This document is design,
not implementation. No provider code, retry logic, fallback logic, or LangGraph
workflows exist yet.

This library is a reusable, provider-agnostic LLM orchestration layer intended
to be consumed by multiple unrelated AI applications. Its first real consumer
will be ReviewIQ, but **no ReviewIQ-specific business logic may live here.**

---

## 1. Repository audit summary (Phase 0 Step 1)

| Check | Result |
| --- | --- |
| Working directory | `llm-orchestrator` |
| Tracked/untracked files | None |
| Hidden files | `.git/`, `.freebuff/project-id` only |
| README | None |
| `pyproject.toml` | None |
| requirements files | None |
| tests / docs | None |
| Git state | Initialized, branch `master`, **no commits yet** |
| Remote | `origin` → `https://github.com/kit29-25bad132/llm-orchestrator.git` |
| Python | 3.14.0 (`python`; `python3` alias points to Windows Store stub) |
| pip | 26.2.1 |
| Packaging tools | No Poetry, no uv |
| Node | v24.13.1 (not used by this project) |

**Conclusion:** the repository is genuinely empty of project material. It
contains only Git metadata and Freebuff's local `.freebuff/` state (already
excluded via `.git/info/exclude`). Phase 0 starts from a clean slate.

> **Environment risk to flag for human approval.** Python 3.14 is very new. Some
> V1 dependencies (LangGraph, Pydantic, provider SDKs) may not yet publish
> wheels for 3.14 and may require a source build. See §14 and §18. This does not
> block Phase 0 but must be resolved in Phase 1 before installing dependencies.

---

## 2. Project boundaries (Phase 0 Step 2)

### A. What `llm-orchestrator` IS

A small, reusable infrastructure library that answers one question:

> *Given a provider-agnostic generation request, how do I reliably obtain a
> validated result from one of several interchangeable LLM providers/models —
> with retries, fallbacks, structured output, and observability — without the
> calling application knowing which provider answered?*

### B. What it IS NOT

- Not an application, chatbot, agent framework, or product.
- Not a home for domain prompts, business rules, or ReviewIQ logic.
- Not a RAG engine, vector store, or memory system.
- Not a multi-agent or autonomous-agent runtime.
- Not a model-training, hosting, or inference-server toolkit.
- Not a database or persistence layer.
- Not a dashboard, UI, or web service.
- Not a hard dependency on Gemini, Groq, or LangGraph.

### C. Responsibilities of the orchestration library

- Normalized request/response contracts.
- Provider and model abstraction + registry.
- Candidate selection (routing).
- Per-candidate retry with exponential backoff and jitter.
- Candidate-level fallback (model and provider).
- Normalized error taxonomy and classification.
- Timeout enforcement.
- Structured-output handling and schema validation (via a pluggable port).
- LangGraph-based orchestration as the control-flow engine.
- Configuration loading (secrets from environment only).
- Observability hooks and structured events.
- Test infrastructure (fakes, offline suites).

### D. Responsibilities of consuming applications

- Business/domain prompts and prompt templates.
- Defining response schemas for their own structured outputs.
- Supplying credentials via environment variables.
- Choosing which models/providers to enable and in what order.
- Deciding what to do with a result or a terminal failure.
- Application-level caching, persistence, and persistence of run history.

### E. Required for V1

Provider abstraction, Gemini + Groq adapters, model selection, timeout handling,
retry (backoff + jitter), fallback (model + provider), error classification,
structured output + schema validation, LangGraph orchestration, configuration
management, basic observability, and a strong offline test suite.

### F. Explicitly postponed

Cost/latency/reliability-aware adaptive routing, streaming, tool/function
calling, multimodal input, embedding/batch endpoints, metrics exporters
(OTel/Prometheus), multiple concurrent orchestration engines, and persistence.

**Guiding rule: no abstraction without a second real consumer.** Interfaces are
introduced only where a concrete second implementation is expected (a second
provider, a second schema validator, a second observer). Everything else stays
concrete until it earns its abstraction.

---

## 3. Architecture overview (Phase 0 Step 3)

### Dependency direction

```
        Application (e.g. ReviewIQ) — owns business logic
                        │
                        ▼
                Orchestrator  (control flow + engine adapter)
                        │
        ┌───────────────┼────────────────┐
        ▼               ▼                ▼
   Routing        Reliability        Validation
   policy         (retry/fallback)   (schema port)
        │               │                │
        └───────────────┼────────────────┘
                        ▼
              Provider abstraction (contracts)
                        │
                        ▼
          Concrete providers (Gemini, Groq, …)
```

Rules:

1. **Dependencies point downward only.** Lower layers never import upper layers.
2. **Contracts form the seam.** The orchestrator depends on the `Provider`
   protocol, never on a concrete provider.
3. **Providers are dumb translators.** They do not decide retry/fallback/routing.
4. **Policies are pure and injectable.** Retry/fallback/routing are data/pure
   functions, so they can be unit-tested without I/O and swapped per consumer.
5. **LangGraph is an engine adapter, not the architecture.** The core
   orchestration semantics (attempt, classify, retry, fallback, validate) are
   engine-agnostic; LangGraph is the V1 implementation of the control flow.

### Conceptual areas → where they live

| Concept | Lives in | Owns |
| --- | --- | --- |
| Contracts | `contracts/` | request/response/config data models + `Provider` protocol |
| Providers | `providers/` | concrete adapters + registry |
| Model registry | `models/` (or config) | known models + capabilities/pricing metadata |
| Routing | `routing/` | candidate ordering |
| Reliability | `reliability/` | retry + fallback policy evaluation, backoff math |
| Error classification | `errors/` | normalized taxonomy + mapping |
| Structured output / validation | `structured/` | schema port + Pydantic adapter |
| Orchestration | `orchestration/` | the run loop / engine adapter + LangGraph graph |
| Configuration | `config/` | settings loading from env |
| Observability | `observability/` | observer protocol + events + sinks |

Not every area needs its own package on day one. Phase 1 creates only the package
skeleton; `contracts/`, `errors/`, and `config/` arrive in Phase 2; the rest
arrive in the phase that needs them (see `docs/roadmap.md`).

---

## 4. Core abstractions (Phase 0 Step 4)

These are the library's public vocabulary, implemented in Phase 2 as
`llm_orchestrator.contracts`, with the error hierarchy in
`llm_orchestrator.errors`.

Two Phase-2 refinements to the text below: `request_id` is a first-class field
on `LLMRequest` (rather than living inside `metadata`), and `LLMResponse`
deliberately omits a raw provider payload so contracts stay provider-neutral and
serializable.

### 4.1 `Message`

- **Purpose:** one provider-neutral conversational turn.
- **Required:** `role` (`system` \| `user` \| `assistant` \| `tool`), `content`.
- **Optional:** `name`, `metadata`.
- **Ownership:** library.
- **Why:** providers disagree on message shapes; one normalized shape is the
  single translation point.
- **Must NOT include:** provider-specific fields (e.g. Gemini `parts`).

### 4.2 `LLMRequest`

- **Purpose:** expresses intent to generate, independent of who serves it.
- **Required:** `messages` (non-empty).
- **Optional:** `response_schema` (structured output target), `max_tokens`,
  `temperature`, `stop`, `timeout_s` (overlay), `metadata` (e.g. `request_id`,
  tags used by routing/observability).
- **Ownership:** library.
- **Why:** one canonical input to every provider and policy.
- **Must NOT include:** provider name, model name, API keys, tenant logic,
  business rules. Choosing *who* serves the request is routing's job, not the
  request's. (An advisory routing tag may ride in `metadata`; the request never
  names a concrete provider.)

### 4.3 `LLMResponse`

- **Purpose:** the normalized successful result.
- **Required:** `request_id`, `provider`, `model`, `finish_reason`, `latency_ms`.
- **Optional:** `content` (text), `structured` (validated object), `usage`,
  `raw` (provider payload, opt-in and excluded from logs by default).
- **Ownership:** library.
- **Why:** callers must not learn provider quirks; provenance is recorded so
  behavior stays auditable.
- **Must NOT include:** business interpretation of the content.

### 4.4 `ModelConfig`

- **Purpose:** describes one concrete model offered by a provider.
- **Required:** `provider`, `model`.
- **Optional:** `capabilities` (structured output, streaming, tool calling,
  context window), `pricing` (input/output per 1K tokens, currency),
  `defaults` (temperature, max_tokens, timeout), `enabled`, `tags`.
- **Ownership:** library schema; values supplied by the consuming app.
- **Why:** routing and capability checks need model facts in one place; pricing
  fields exist for the future cost-aware router.
- **Must NOT include:** credentials.

### 4.5 `ProviderConfig`

- **Purpose:** provider-level, non-secret configuration.
- **Required:** `name`, `credential_env_var` (the **name** of an env var).
- **Optional:** `base_url`, `default_timeout_s`, `max_concurrency`, `extra`
  (non-secret provider options).
- **Ownership:** library schema; values from app config/env.
- **Why:** keeps secrets out of code and centralizes per-provider settings.
- **Must NOT include:** the secret value itself.

### 4.6 `RetryPolicy`

- **Purpose:** decides whether and how the *same candidate* is attempted again.
- **Required:** `max_attempts` (≥1).
- **Optional:** `initial_backoff_s`, `multiplier`, `max_backoff_s`, `jitter`
  (`none` \| `full` \| `equal` \| `decorrelated`), `retry_on`
  (categories; `None` = use each error's default retryability).
- **Ownership:** library (schema + evaluation); values from app config.
- **Why:** reliability is configurable and testable.
- **Must NOT include:** provider SDK details or sleeps that ignore injectable
  clock/sleep (needed for deterministic tests).

### 4.7 `FallbackPolicy`

- **Purpose:** decides whether execution moves to a *different candidate*.
- **Required:** `max_fallbacks` (≥0).
- **Optional:** `candidates` (explicit ordered list; `None` = derive from
  routing), `fallback_on` (categories), `include_validation_failures`,
  `cross_provider` (allow moving across providers).
- **Ownership:** library; values from app config.
- **Why:** separates "try again" from "try elsewhere" (§6).
- **Must NOT include:** candidate selection *order* logic (that is routing).

### 4.8 `RoutingPolicy` (protocol)

- **Purpose:** produce an **ordered** list of candidates for a request.
- **Required input:** `request`, available `ModelConfig`s, and a `RoutingContext`
  (health/latency/cost signals; empty in V1).
- **Output:** `Sequence[CandidateRef]` (`provider` + `model`), best first.
- **Ownership:** library protocol; implementations by library (static/explicit
  in V1) or by consumers.
- **Why:** V1 uses an explicit order; the same seam later supports adaptive,
  cost/latency/reliability-aware selection without touching the orchestrator.
- **Must NOT include:** retry/fallback decisions or provider calls.

### 4.9 `OrchestrationResult`

- **Purpose:** the complete, observable outcome of one orchestration run.
- **Required:** `request_id`, `succeeded`.
- **Optional:** `response`, `error` (an `ErrorInfo` snapshot rather than a live
  exception, so results serialize), `candidates_tried`, `attempts`
  (`AttemptRecord[]`), `total_latency_ms`. Retry/fallback counts are derived
  properties. (`retry_events`, `fallback_events`, and `validation` arrive with
  the observability and structured-output phases.)
- **Ownership:** library.
- **Why:** callers need one object that explains *what happened and why* without
  reading logs.
- **Must NOT include:** app-level decisions or persisted history.

### 4.10 Error hierarchy

See §7. One normalized base (`LLMError`) with a `category`, plus `retryable`
and `fallback_eligible` flags derived from the taxonomy.

---

## 5. Provider architecture (Phase 0 Step 5)

### 5.1 Shape (implemented in Phase 3 as a `Protocol`)

```
LLMProvider (Protocol)          # providers/base.py
├── GeminiProvider              # implemented (providers/gemini.py)
├── GroqProvider                # Phase 5
└── (future providers)          # no core changes
```

Phase 3 chose structural typing over an `ABC`: there is no shared behavior to
inherit yet, and a protocol keeps test doubles trivial. A shared `BaseProvider`
can be added later if real shared behavior appears, without changing the
contract.

### 5.2 Provider interface (contract)

Implemented in `llm_orchestrator.providers.base`:

```python
@runtime_checkable
class LLMProvider(Protocol):
    provider_id: str                          # opaque, not an enum

    async def generate(
        self,
        request: LLMRequest,
        model: ModelConfig,
        *,
        timeout_s: float | None = None,
    ) -> LLMResponse: ...

    async def aclose(self) -> None: ...
```

- `generate` performs exactly **one** attempt. It never loops, retries, or
  falls back. The target `model` is passed explicitly because `LLMRequest`
  deliberately never names a provider or model.
- Adding a provider must require **only** a new `LLMProvider` implementation plus
  a registry entry — never edits to the orchestrator, routing, or reliability
  code.
- The interface speaks only Phase 2 contracts; vendor SDK types never cross it.

### 5.3 Provider capabilities

Model-level (declared in `ModelConfig.capabilities`): `structured_output`,
`streaming`, `tool_calling`, `max_context_tokens`. The orchestrator checks
capabilities before use (e.g. never request structured output from a model that
cannot do it). Capabilities are *data*, not per-provider `if` branches in the
core.

### 5.4 Model identification

A candidate is `(provider, model)`. `model` is the provider's own model id
(e.g. `gemini-2.x`, `llama-3.x`). The core treats it as an opaque string; only
the concrete provider interprets it.

### 5.5 Provider-specific vs normalized errors

Providers translate native SDK exceptions into the normalized taxonomy (§7).
Any unmapped exception becomes `UnknownProviderError` (category `UNKNOWN`) and
its original cause is preserved. Providers must never leak raw SDK exception
types to callers.

`GeminiProvider` (Phase 4) implements this mapping from the SDK's structured
fields (HTTP `code`, `status`, `message`) rather than message string matching,
and it maps only; retry/fallback decisions remain with the reliability layer.

### 5.6 Registration and discovery

- **Phase 3:** an explicit `ProviderRegistry` (`providers/registry.py`) mapping
  provider id → provider instance. Callers construct providers and register
  them; duplicate ids are rejected unless `replace=True`, and unknown ids raise
  `ProviderNotRegisteredError`. The registry knows nothing about concrete
  providers.
- **Later:** optional discovery via Python entry points so third-party packages
  can register providers. The registry API is designed so discovery is additive.

### 5.7 Provider-specific configuration

Provider options that are not secrets go in `ProviderConfig.extra` (e.g.
`base_url`, API version). Secrets live only in the named environment variable.
Configuration is validated at startup; a missing credential produces
`ConfigurationError` (fail fast, not retry).

---

## 6. Retry vs fallback (Phase 0 Step 6)

| | RETRY | FALLBACK |
| --- | --- | --- |
| Moves to | the **same** provider + model | a **different** candidate |
| Triggered by | retryable errors / validation failure | non-retryable errors, exhausted retries, or policy |
| Bounded by | `RetryPolicy.max_attempts` | `FallbackPolicy.max_fallbacks` |
| Ordering | backoff + jitter between attempts | next candidate from routing order |
| Cost profile | cheap if transient, wasteful if not | more likely to succeed, may differ in quality |

They compose: each candidate gets its own retry budget before the run falls
back. Fallback may also be immediate when an error is non-retryable
(e.g. authentication failure).

### Future flow

```
Request
  │
  ▼
Select candidate (RoutingPolicy)
  │
  ▼
Call model (Provider.generate — one attempt)
  │
  ▼
Failure? ── retryable ──► retry (same candidate, backoff+jitter, if budget left)
  │         non-retryable ─► fallback (next candidate) or fail per policy
  │         success ──────► validate
  ▼
Validation failure? ── yes ─► retry (repair) / fallback per policy
  │
  ▼
Success ─► OrchestrationResult(succeeded=True)
No candidates left ─► OrchestrationResult(succeeded=False, error=terminal)
```

### Policy vs provider responsibility

- **Provider:** performs one attempt; translates request/response/errors; reports
  capabilities. It holds **no** control-flow logic.
- **Policy (library):** decides retryable vs fallback vs fail; computes backoff;
  chooses the next candidate; enforces budgets/timeouts. Policies are pure and
  injectable, so behavior is deterministic under test.

---

## 7. Error taxonomy (Phase 0 Step 7)

One base class `LLMError` with:

- `category: ErrorCategory`
- `retryable: bool`, `fallback_eligible: bool` (defaults from the table below)
- `provider`, `model`, `request_id`, `attempt` (context)
- `cause` (original exception, for debugging)
- `safe_message` (sanitized for logs — never contains secrets or raw payloads)

| Category | Retryable | Fallback | Fail fast | Rationale |
| --- | --- | --- | --- | --- |
| `AUTHENTICATION` | No | Yes (prefer cross-provider) | Only if no alternatives | Bad/expired key will not fix itself; another provider may work. |
| `AUTHORIZATION` | No | Yes | Only if no alternatives | Permission/scope problem is deterministic for this provider. |
| `RATE_LIMIT` | **Yes** (honor `Retry-After`) | Yes | No | Usually transient; backoff then move provider if still throttled. |
| `TIMEOUT` | **Yes** | Yes | No | Transient; note generation is non-idempotent, so bound retries. |
| `NETWORK` | **Yes** | Yes | No | Connection blips are transient. |
| `PROVIDER_UNAVAILABLE` | Yes (limited) | **Yes (strongly)** | No | Outages; fallback is the higher-value action. |
| `MODEL_UNAVAILABLE` (not found/deprecated) | No | **Yes** | No | Retrying a missing model is pointless; another model may serve. |
| `INVALID_REQUEST` | No | Conditional | Usually yes | Bad parameters/context-too-long are deterministic; fallback only if a larger-context/capable model could satisfy it. |
| `MALFORMED_RESPONSE` | Yes (bounded) | Yes | No | Provider returned unparseable output; often transient. |
| `STRUCTURED_OUTPUT_VALIDATION` | Yes (bounded repair) | Yes | No | Schema mismatch may be repaired by re-prompting, else fallback. |
| `CONFIGURATION` | No | No | **Yes** | Programmer/operator error; must surface immediately. |
| `UNKNOWN` | No | Yes | No | Conservative: do **not** blindly retry; allow fallback. |

Design notes:

- **Not everything is retryable.** `CONFIGURATION`, `INVALID_REQUEST`, auth
  errors, and unknown errors are not automatically retried.
- **`MODEL_UNAVAILABLE` vs `RATE_LIMIT`:** a model that is *overloaded* maps to
  `MODEL_UNAVAILABLE`/`PROVIDER_UNAVAILABLE` (retryable), whereas a model that
  does not *exist* is non-retryable. Providers disambiguate this at mapping time.
- **Overrides:** `RetryPolicy.retry_on` / `FallbackPolicy.fallback_on` may
  override the defaults per consumer, but the taxonomy supplies safe defaults.
- **Validation errors are first-class**, because structured generation failure is
  a common, recoverable condition — not an implementation bug.

Subclasses (one per category):
`AuthenticationError`, `AuthorizationError`, `RateLimitError`,
`LLMTimeoutError`, `NetworkError`, `ProviderUnavailableError`,
`ModelUnavailableError`, `InvalidRequestError`, `MalformedResponseError`,
`StructuredOutputValidationError`, `ConfigurationError`, `UnknownProviderError`.

(`LLMTimeoutError` avoids shadowing the built-in `TimeoutError`.)

---

## 8. LangGraph role (Phase 0 Step 8)

**LangGraph is the control-flow engine, not the library.** The library's
semantics (§6) exist independently; LangGraph is the V1 way to express them. The
core must remain usable if the engine is replaced.

### State

```python
class OrchestrationState(TypedDict):
    request: LLMRequest
    candidates: list[CandidateRef]
    current_index: int
    attempt_in_candidate: int
    response: LLMResponse | None
    validation: ValidationResult | None
    last_error: LLMError | None
    attempts: list[AttemptRecord]
    retry_events: list[RetryEvent]
    fallback_events: list[FallbackEvent]
    result: OrchestrationResult | None
```

State carries only data needed to make the next decision. Policies are injected
(closures / run config), not stored in state.

### Nodes (each is small and single-purpose)

- `plan_candidates` — apply `RoutingPolicy` → ordered candidates.
- `invoke_candidate` — one `Provider.generate` attempt; records attempt and maps
  errors to the taxonomy.
- `validate_output` — run schema validation when a schema is present.
- `decide` — pure function: given last error/validation + budgets, return
  `RETRY`, `FALLBACK`, `SUCCEED`, or `FAIL`.
- `finalize_success` / `finalize_failure` — build `OrchestrationResult`.

### Conditional edges

- after `invoke_candidate` → `validate_output` (success) or `decide` (error)
- after `validate_output` → `finalize_success` or `decide`
- after `decide`:
  - `RETRY` → increment attempt, backoff, → `invoke_candidate`
  - `FALLBACK` → advance candidate → `invoke_candidate`
  - `SUCCEED` → `finalize_success`
  - `FAIL` → `finalize_failure`

Termination: `finalize_success` / `finalize_failure` are the only terminal nodes.

### Graph vs non-graph

- **In the graph:** sequencing of attempt/validate/retry/fallback, and the
  decision edge. This is *control flow*.
- **Outside the graph:** provider SDK calls (wrapped by `Provider`), error
  classification, backoff/jitter math, schema validation, and all
  application/business logic. These stay as pure functions/objects so they are
  unit-testable without the graph.
- **Business logic never enters the graph.** The graph returns an
  `OrchestrationResult`; the consumer decides what it means.

---

## 9. Structured output (Phase 0 Step 9)

### Flow

```
LLM ─► structured response ─► schema validation ─► success
                                              └──► recovery / fallback
```

### Provider-agnostic port

```python
class SchemaAdapter(Protocol):
    def request_hint(self, schema) -> ProviderHints: ...
    def validate(self, raw, schema) -> object: ...   # raises StructuredOutputValidationError
    def repair_instruction(self, error) -> str | None: ...
```

- The library defines the port; **Pydantic is one adapter, not the only one.**
- Consumers bring **their own** schemas (`request.response_schema` is generic:
  a model class, a `TypeAdapter`, or any adapter-supported schema).
- The library never dictates a schema, never forces Pydantic onto callers, and
  never embeds domain models.
- `request_hint` lets providers opt into native JSON/structured modes when
  available; when unavailable, the library may parse from text instead.
- On validation failure, `repair_instruction` can drive a bounded re-prompt
  (retry) before falling back.

Pydantic fits as the **default** adapter (`structured/pydantic_adapter.py`) and
as the config layer's validator — but it is a dependency of an *adapter*, not of
the schema contract.

---

## 10. Observability (Phase 0 Step 10)

A single `Observer` protocol with structured events; the default is a no-op so
the library stays silent unless asked. No network telemetry is enabled by
default.

Events (each carries `request_id`):

- `RequestStart` / `RequestEnd`
- `AttemptStart` / `AttemptEnd` — `provider`, `model`, `attempt`, `latency_ms`,
  `success`, `error_category`
- `RetryEvent` — attempt number, backoff applied, triggering category
- `FallbackEvent` — from-candidate → to-candidate, triggering category/validation
- `ValidationEvent` — schema id, result, repair attempt
- `Usage` — input/output/total tokens and `estimated_cost` when available

**What to observe and why:** request ID (correlate a whole run), provider/model
(know who answered), attempt number (retry behavior), latency (perf + timeouts),
success/failure and error classification (reliability), fallback/retry events
(why the run took the path it did), validation result (structured-output health),
token usage and estimated cost (spend visibility). All events are
privacy-conscious: prompts/responses are **not** included unless the consumer
explicitly opts into a verbose sink.

Sinks: `NullObserver` (default), `LoggingObserver`, `InMemoryObserver` (tests).
A future OpenTelemetry adapter implements the same protocol — no core changes.

---

## 11. Configuration (Phase 0 Step 11)

### Where each concern lives

| Concern | Home | Example |
| --- | --- | --- |
| Secrets | **Environment variables only** | `GEMINI_API_KEY`, `GROQ_API_KEY` |
| Runtime/environment switches | Environment variables | `LLM_ORCH_ENV`, `LLM_ORCH_LOG_LEVEL` |
| Provider config | `ProviderConfig` objects (from env/app) | base URL, default timeout, credential env var *name* |
| Model registry | `ModelConfig` list (from app config) | model ids, capabilities, pricing |
| Retry/fallback/timeout/routing | Policy objects (from app config) | `RetryPolicy`, `FallbackPolicy`, `RoutingPolicy` |
| Policy *logic* | Code | backoff math, classification |
| Business choices | **Consuming application** | prompts, which models, schemas |

- Loading uses `pydantic-settings` (planned) with env-var validation and clear
  `ConfigurationError` on anything missing/malformed.
- Secrets are **never** hardcoded, never logged, and never stored in config
  objects — only the env var *name* is referenced.
- Configuration is validated once at startup; misconfiguration fails fast.

---

## 12. Security (Phase 0 Step 12)

- **No API keys in source.** Credentials are read from the environment at
  startup; `.env` is for local dev only.
- **`.env` handling.** `.env` is git-ignored. A committed `.env.example`
  documents required variable **names** with empty/placeholder values.
- **Logging hygiene.** A redaction step strips secrets and (by default) prompt
  and response bodies from logs and error messages.
- **Sensitive prompt/response handling.** Body logging is opt-in and off by
  default; `LLMResponse.raw` is excluded from default serialization.
- **Error message sanitization.** `LLMError.safe_message` never includes secrets,
  auth headers, or raw provider payloads; the original `cause` is available only
  to code, not to logs.
- **Dependency security.** Pin dependencies, prefer maintained libraries, and
  run `pip-audit` in CI. Keep the dependency surface small.
- **Least privilege.** Credentials are scoped per provider and loaded lazily.

---

## 13. Testing strategy (Phase 0 Step 13)

| # | Suite | Scope | Offline? |
| --- | --- | --- | --- |
| 1 | Unit | contracts, config, backoff math, classification | Yes |
| 2 | Provider | each adapter against mocked HTTP/SDK + a shared contract test | Yes |
| 3 | Routing | ordering, capability filtering, empty candidate sets | Yes |
| 4 | Retry | budgets, backoff/jitter, `retry_on` overrides | Yes |
| 5 | Fallback | model-level and provider-level, cross-provider rules | Yes |
| 6 | Error classification | provider exception → category mapping table | Yes |
| 7 | Structured output | parse, validate, repair, failure paths | Yes |
| 8 | LangGraph | node/edge behavior, termination, state transitions | Yes |
| 9 | Integration | orchestrator + fake providers + real policies | Yes |
| 10 | Failure simulation | injected timeouts/5xx/rate limits/validation failures | Yes |
| 11 | End-to-end | real Gemini + Groq | **No — live** |

Key techniques:

- A **`FakeProvider`** with programmable successes/failures powers most suites —
  fast, deterministic, and quota-free.
- **Injected clock and sleep/jitter RNG** make backoff tests exact (no real
  waits).
- **Shared provider contract test:** every provider must pass the same behavioral
  suite, so a new provider is "done" when it passes.
- **Live tests** are marked (e.g. `@pytest.mark.live`) and skipped unless the
  relevant API key is present, so normal development consumes **no** API quota.
- Target: the entire offline suite runs in CI with no network access.

---

## 14. V1 scope (Phase 0 Step 14)

### In scope

- Clean provider abstraction + registry.
- Gemini adapter, Groq adapter.
- Model selection (explicit/static routing).
- Timeout handling.
- Retry with exponential backoff + jitter.
- Model-level and provider-level fallback.
- Normalized error classification.
- Structured output + schema validation (Pydantic adapter).
- LangGraph-based orchestration.
- Configuration management (env + config objects).
- Basic observability (Observer protocol + logging/in-memory sinks).
- Strong offline test suite + live opt-in tests.

### Explicitly out of scope for V1

- Multi-agent / autonomous agents.
- Local model hosting.
- Databases / vector stores / RAG.
- Frontend.
- Unrelated business features.
- Excessive enterprise abstractions (plugin marketplaces, DI containers).
- Streaming, tool calling, multimodal, embeddings/batch (postponed unless a
  second consumer justifies them).
- Adaptive cost/latency/reliability routing (designed for, not built).
- Telemetry exporters (OTel/Prometheus).
- Persistence of run history.

---

## 15. Recommended directory structure (Phase 0 Step 15)

### Target (evolves over phases)

```
llm-orchestrator/
├── src/llm_orchestrator/
│   ├── contracts/        # requests, responses, configs, Provider protocol
│   ├── errors/           # taxonomy + classification
│   ├── providers/        # base + gemini + groq + registry
│   ├── models/           # model registry + capabilities/pricing
│   ├── routing/          # routing policies
│   ├── reliability/      # retry + fallback + backoff
│   ├── structured/       # schema port + pydantic adapter
│   ├── orchestration/    # run loop + langgraph graph
│   ├── config/           # settings from env
│   └── observability/    # observer + events + sinks
├── tests/                # unit, provider, integration, failure, e2e
├── examples/             # runnable usage samples (later phases)
├── docs/                 # architecture.md, roadmap.md
├── .env.example
├── .gitignore
├── pyproject.toml
├── README.md
└── LICENSE
```

### Created in Phase 0 (this phase)

```
llm-orchestrator/
├── .gitignore
├── README.md
└── docs/
    ├── architecture.md
    └── roadmap.md
```

**Deferred to later phases:** `src/` tree, `tests/`, `examples/`,
`pyproject.toml`, `.env.example`, `LICENSE`.

Rationale: Phase 0 is architecture only. Creating `src/` or `pyproject.toml`
now would commit to packaging/dependency choices before the human has approved
them (see §18). `LICENSE` requires a human licensing decision and is intentionally
not created.

---

## 18. Concerns requiring human approval

1. **Python version.** Local interpreter is 3.14.0, which may lack wheels for
   LangGraph/provider SDKs. Decide whether to target 3.11/3.12/3.13 for V1 or
   verify 3.14 support before Phase 1. Proposed floor: `requires-python >= 3.11`.
2. **License.** No `LICENSE` file created — licensing is a human decision.
3. **Dependency choices** (deferred to the phase that needs each one). Phase 1
   keeps runtime dependencies **empty**; `pydantic` v2 + `pydantic-settings` are
   added with the contracts/config work (Phase 2), LangGraph remains an
   **optional extra** added when orchestration lands, official provider SDKs
   (`google-genai`, `groq`) are wrapped behind adapters in their provider
   phases, and retry uses a custom in-library implementation (for explicit,
   testable backoff/jitter) rather than a third-party retry library.
4. **Packaging tool.** Environment has no Poetry/uv; proposal is a standard
   `pyproject.toml` with a PEP 621 layout and `pip install -e .` for dev.
