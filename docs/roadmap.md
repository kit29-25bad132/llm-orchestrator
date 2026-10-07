# llm-orchestrator — Development Roadmap

This roadmap leads from an empty repository to a production-hardened, portable
library validated on a second consumer. Each phase lists its objective,
components, tasks, tests, completion criteria, what must not change, and its
deliverable.

**Global rules for every phase**

- The library must never contain application/business logic (especially no
  ReviewIQ logic).
- The architecture must remain provider-agnostic; no phase may hard-wire Gemini
  or Groq into the core.
- Prefer offline tests; live provider calls are opt-in and quota-gated.
- Secrets only via environment variables.
- No Git mutations by the agent — the human reviews the diff and commits.

Legend: **O** objective · **C** components/files · **T** tasks · **X** tests ·
**D** done when · **!** must not change · **Deliverable**.

---

## Phase 0 — Architecture & Foundation  ← *this phase*

- **O:** Verify the empty-repo assumption; establish boundaries, architecture,
  abstractions, and the phase plan. No implementation.
- **C:** `README.md`, `.gitignore`, `docs/architecture.md`, `docs/roadmap.md`.
- **T:** Audit repo; document IS/IS NOT, dependency direction, contracts, error
  taxonomy, retry-vs-fallback, LangGraph role, structured output, observability,
  configuration, security, testing, V1 scope, structure.
- **X:** Read-only validation: no secrets, provider-agnostic, internally
  consistent, no code introduced.
- **D:** Docs exist, review passes, no source files created.
- **!:** No `src/`, no provider code, no dependencies installed, no LICENSE.
- **Deliverable:** Reviewed architecture + roadmap; human-approved decisions in
  §18 of `architecture.md`.

---

## Phase 1 — Package Foundation

- **O:** Create an installable, linted, typed, testable package skeleton.
- **C:** `pyproject.toml` (PEP 621), `src/llm_orchestrator/__init__.py`,
  `.env.example`, CI config, dev tooling (ruff, mypy, pytest).
- **T:** Decide Python floor (3.11+) and verify against the local 3.14
  interpreter; keep runtime dependencies **empty** (Pydantic/pydantic-settings
  and LangGraph are added in the phases that consume them); set up
  `pip install -e ".[dev]"`; add a foundation test.
- **X:** Package imports; `pytest` runs a placeholder-free foundation test;
  ruff + mypy pass.
- **D:** `pip install -e ".[dev]"` works; CI is green; no behavior yet.
- **!:** No provider, retry, fallback, or graph code. No secrets committed.
- **Deliverable:** Installable skeleton with tooling and CI.

---

## Phase 2 — Contracts

- **O:** Implement the public vocabulary (design in `architecture.md` §4).
- **C:** `contracts/` (messages, `LLMRequest`, `LLMResponse`, `Usage`,
  `ModelConfig`, `ProviderConfig`, policy contracts, `OrchestrationResult`,
  `CandidateRef`, `ErrorInfo`); `errors/` (`ErrorCategory` + `LLMError`
  hierarchy). `config/` (env-backed settings) moved to the phase that first
  reads the environment, since the contract layer must not touch env vars; the
  `Provider` protocol belongs to Phase 3.
- **T:** Define frozen Pydantic data models with scoped validation; implement the
  error hierarchy with per-category default `retryable`/`fallback_eligible`
  flags and message sanitization.
- **X:** Unit tests for validation, defaults, boundary cases, serialization
  round-trips, error flags, provider-independence invariants; all offline.
- **D:** Contracts install, import cleanly, are fully typed, and match the
  documented design; `ruff`, `ruff format`, `mypy`, and `pytest` all pass.
- **!:** No concrete provider, no retry/fallback execution, no LangGraph.
- **Deliverable:** Frozen contracts + error taxonomy (no config loader yet).

---

## Phase 3 — Provider Abstraction

- **O:** Define how providers plug in, independent of any vendor.
- **C:** `providers/base.py` (`LLMProvider` protocol), `providers/registry.py`
  (`ProviderRegistry`), and a test-only `FakeProvider` under `tests/`.
- **T:** Implement the protocol and registry (reject duplicates, raise on unknown
  ids); implement `FakeProvider` with scriptable success/failure.
- **X:** Protocol conformance, invocation, registry (duplicate/unknown/isolation),
  error propagation, and AST-based provider-independence invariants.
- **D:** A provider can be registered and invoked offline; no SDK, env, or
  concrete provider is introduced.
- **!:** No vendor SDK imports in core; providers perform exactly one attempt.
- **Deliverable:** Provider seam + registry + fake provider + contract suite.

---

## Phase 4 — Gemini Provider

- **O:** Add the first real adapter without touching the core.
- **C:** `providers/gemini.py` (the only module allowed to import the Gemini
  SDK).
- **T:** Translate request/messages/generation parameters, normalize responses and
  usage, and map Gemini errors to the taxonomy; inject the credential explicitly;
  keep the SDK isolated to this module.
- **X:** Offline adapter tests against a fake SDK client (translation,
  normalization, error mapping, secret safety, no-retry/no-fallback) plus an
  opt-in `live` integration test.
- **D:** Gemini passes the offline suite; core contracts, abstraction, and
  registry are unchanged; `docs/providers.md` documents it.
- **!:** No routing/retry/fallback inside the provider; no environment reads; no
  hidden model substitution; no core edits.
- **Deliverable:** Working Gemini adapter + provider documentation.

---

## Phase 5 — Groq Provider

- **O:** Prove the abstraction with a second, structurally different provider.
- **C:** `providers/groq.py` (+ error mapping).
- **T:** Same shape as Phase 4; confirm no core changes are needed.
- **X:** Contract test (offline); mapping tests; opt-in live test.
- **D:** Groq passes the same contract suite; core still unchanged.
- **!:** No provider-specific branching added to the core.
- **Deliverable:** Working Groq adapter; abstraction validated by two providers.

---

## Phase 6 — Retry

- **O:** Implement per-candidate retry with exponential backoff + jitter.
- **C:** `reliability/retry.py`, `reliability/backoff.py`.
- **T:** Implement budgets, `initial_backoff_s`, `multiplier`, `max_backoff_s`,
  jitter strategies; honor the taxonomy's default retryability; inject clock and
  sleep/RNG for determinism; respect `Retry-After` on rate limits.
- **X:** Budget tests; exact backoff/jitter tests (no real sleeps); override
  tests (`retry_on`); non-retryable categories never retried.
- **D:** Retry is deterministic under test and honors policy + taxonomy.
- **!:** Retry must never change candidates (that is fallback).
- **Deliverable:** Tested retry engine (pure, injectable).

---

## Phase 7 — Error Classification

- **O:** Mature the provider-exception → category mapping and its guarantees.
- **C:** `errors/classify.py` (+ per-provider mapping helpers).
- **T:** Consolidate mapping tables; ensure unmapped → `UNKNOWN`; preserve
  `cause`; guarantee `safe_message` redacts secrets and payloads.
- **X:** Table-driven tests per provider category; redaction tests; unknown-error
  behavior tests.
- **D:** Every documented category has a tested mapping; no secret can leak
  through `safe_message`.
- **!:** Classification stays pure; it must not perform I/O or decide policy.
- **Deliverable:** Complete, tested classification layer.

---

## Phase 8 — Fallback

- **O:** Move execution to another candidate when policy requires.
- **C:** `reliability/fallback.py`.
- **T:** Implement `FallbackPolicy` evaluation: exhaust retries per candidate,
  advance through routing order, enforce `max_fallbacks`, honor `cross_provider`,
  and support `include_validation_failures`.
- **X:** Model-level and provider-level fallback tests; exhausted-candidate
  tests; cross-provider rule tests; failure simulation.
- **D:** Fallback advances candidates correctly and terminates predictably.
- **!:** Fallback must not reorder/select candidates (routing owns order).
- **Deliverable:** Tested fallback engine.

---

## Phase 9 — LangGraph Orchestration

> Status: the single-candidate slice has landed early as Phase 8
> (`orchestration/` with `START → execute_model → END`; see
> `docs/orchestration.md`). What remains here is the multi-node control flow:
> candidate planning, validation, `decide` edges, and fallback sequencing once
> those phases exist.

- **O:** Express the run loop as a LangGraph graph (design in §8).
- **C:** `orchestration/graph.py`, `orchestration/state.py`,
  `orchestration/runner.py`.
- **T:** Implement nodes (`plan_candidates`, `invoke_candidate`,
  `validate_output`, `decide`, `finalize_*`) and conditional edges; assemble the
  retry/fallback/success/fail control flow; keep policies injectable.
- **X:** Node/edge tests; state-transition tests; termination tests (success and
  exhaustion); integration with fake providers; compile-graph smoke test.
- **D:** A full run produces an `OrchestrationResult` with correct retry/fallback
  traces; core semantics remain engine-agnostic.
- **!:** No business logic in the graph; no provider SDK calls in nodes (delegate
  to `Provider`); LangGraph stays swappable.
- **Deliverable:** LangGraph-based orchestration engine.

---

## Phase 10 — Structured Output

> Status: the post-hoc validation slice has landed early as Phase 9
> (`structured/` with `validate_structured_output`; see
> `docs/structured-output.md`). What remains here is provider-native
> structured modes (`request_hint`), bounded repair re-prompts, and wiring
> validation into the graph once fallback/routing exist.

- **O:** Provider-agnostic structured generation + schema validation.
- **C:** `structured/base.py` (schema port), `structured/pydantic_adapter.py`.
- **T:** Implement `request_hint`, `validate`, `repair_instruction`; wire
  validation into the graph's `validate_output` node; bounded repair re-prompts;
  map failures to `StructuredOutputValidationError`.
- **X:** Parse/validate tests; schema-mismatch tests; repair-path tests; fallback
  on validation failure; adapter-contract tests.
- **D:** Consumers get validated objects; failures recover or fall back per
  policy.
- **!:** No domain schema in the library; Pydantic is an adapter, not the
  contract.
- **Deliverable:** Structured-output pipeline + Pydantic adapter.

---

## Phase 11 — Routing

> Status: the static/explicit slice has landed as Phase 10 (`routing/` with
> `StaticRoutingPolicy`, `ModelRouter`, and deterministic capability/cost/
> priority selection; see `docs/routing.md`). What remains here is the
> `RoutingContext` seam for future adaptive signals (health/latency/cost)
> and orchestrator wiring once fallback exists.

- **O:** Select and order candidates.
- **C:** `routing/static.py` (+ `RoutingPolicy` implementations).
- **T:** Implement explicit/static ordering and capability filtering; define the
  `RoutingContext` seam for future signals (health/latency/cost).
- **X:** Ordering tests; capability-filter tests; empty/degenerate candidate sets;
  custom-policy tests.
- **D:** Routing deterministically orders candidates and the orchestrator uses it.
- **!:** Routing must not perform retries/fallbacks or call providers.
- **Deliverable:** Routing policies + context seam.

---

## Phase 12 — Observability

- **O:** Make every run explainable via structured events.
- **C:** `observability/observer.py`, `observability/events.py`,
  `observability/sinks.py` (null, logging, in-memory).
- **T:** Emit Request/Attempt/Retry/Fallback/Validation/Usage events; wire into
  the graph runner; redact prompt/response bodies by default.
- **X:** Event-emission tests; in-memory sink assertions; redaction tests.
- **D:** A run yields a complete event trail; default sink is a no-op.
- **!:** No network telemetry by default; no prompts/responses logged unless
  opted in.
- **Deliverable:** Observer protocol + events + sinks.

---

## Phase 13 — Testing Hardening

- **O:** Complete the offline suite and the live/e2e layer (design in §13).
- **C:** `tests/` full matrix + shared fixtures + failure-injection utilities;
  CI configuration.
- **T:** Finish unit, provider, routing, retry, fallback, classification,
  structured, LangGraph, integration, and failure-simulation suites; add live
  e2e tests gated by env; enforce `pytest.mark.live` skipping.
- **X:** The entire offline suite; CI runs with no network and no API quota.
- **D:** Offline suite is comprehensive and green; live tests run only on demand.
- **!:** Do not weaken assertions or skip offline tests to make CI pass.
- **Deliverable:** Strong, layered test suite.

---

## Phase 14 — Packaging & Release

- **O:** Make the library installable and maturable for consumers.
- **C:** `pyproject.toml` metadata, versioning, changelog, docs polish,
  extras (`[langgraph]`, `[dev]`, provider extras).
- **T:** Finalize public API surface and `__all__`; build sdist/wheel; verify a
  clean-venv install; document quickstart in `README.md`.
- **X:** Build + install test in a fresh environment; import of the public API.
- **D:** A new project can `pip install` the library and use it.
- **!:** No API churn without a version bump and a note.
- **Deliverable:** Packaged, documented release candidate.

---

## Phase 15 — ReviewIQ Integration

- **O:** Validate the library against its first real consumer.
- **C:** Changes **in the ReviewIQ repo**, not here.
- **T:** Consume `llm-orchestrator` from ReviewIQ: define ReviewIQ's schemas,
  model choices, and policies in the app; keep all ReviewIQ logic out of the
  library.
- **X:** ReviewIQ-side tests; confirm the library needed **no** ReviewIQ-specific
  changes.
- **D:** ReviewIQ works end-to-end using the public API only.
- **!:** No ReviewIQ imports, names, or domain logic may enter this repository.
- **Deliverable:** A working real consumer.

---

## Phase 16 — Second-Project Portability Test

- **O:** Prove reusability on an unrelated project.
- **C:** A small throwaway consumer (e.g. under `examples/` or a scratch repo).
- **T:** Integrate the library with a different domain and a different provider
  preference; note every friction point.
- **X:** The second consumer runs; friction list reviewed.
- **D:** The second consumer integrates with no core changes; gaps are recorded.
- **!:** No changes that benefit only one consumer's domain.
- **Deliverable:** Portability evidence + prioritized gap list.

---

## Phase 17 — Production Hardening

- **O:** Address reliability, security, and operational gaps surfaced by Phases
  15–16 and the gap list.
- **C:** Targeted changes across `reliability/`, `errors/`, `config/`, docs,
  dependency pins.
- **T:** Tune backoff/timeouts; revisit error mappings; run `pip-audit`; finalize
  redaction and secret handling; add any justified constraints; document
  operational guidance.
- **X:** Full offline regression; failure-simulation stress; dependency audit.
- **D:** Hardening items closed or explicitly deferred with rationale.
- **!:** No reduction of test coverage to gain speed.
- **Deliverable:** Hardened library + operational docs.

---

## Phase 18 — Final E2E Validation

- **O:** Confirm the library is production-ready and portable.
- **C:** Full system.
- **T:** Run offline + live e2e suites; verify both Gemini and Groq; verify
  provider-agnosticism (add/remove a provider without core edits); freeze V1.
- **X:** Complete offline suite; opt-in live e2e; portability check.
- **D:** All suites green; V1 scope met; no blocking concerns.
- **!:** No new features in this phase — validation only.
- **Deliverable:** V1 sign-off; tag performed manually by the human.

---

## Capability → phase map

| Capability | Phase |
| --- | --- |
| Provider abstraction | 3 |
| Gemini | 4 |
| Groq | 5 |
| Retry | 6 |
| Error classification | 2 (taxonomy), 7 (mapping) |
| Fallback | 8 |
| LangGraph orchestration | 9 |
| Structured output | 10 |
| Routing / model selection | 11 |
| Observability | 12 |
| Configuration | 2 |
| Testing | continuous; hardened in 13 |
| Packaging | 1, 14 |
| ReviewIQ integration | 15 |
| Portability test | 16 |
| Production hardening | 17 |
| Cost/latency/reliability-aware routing | **deferred** (seam only, Phase 11) |
