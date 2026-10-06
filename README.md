# llm-orchestrator

A reusable, **provider-agnostic** LLM orchestration library — provider and model
abstraction, routing, retries with backoff/jitter, model- and provider-level
fallback, normalized error classification, structured output with schema
validation, LangGraph-based orchestration, configuration, and observability.

It is **infrastructure**, not an application: consuming projects (such as
ReviewIQ) bring their own prompts, schemas, and business logic.

> **Status: Phase 0 — architecture and foundation only.**
> No implementation exists yet. There is no installable package, no provider
> code, and no dependencies have been installed. This repository currently
> contains documentation that defines the design and the development plan.

## Why

Applications that call LLMs repeatedly re-solve the same problems: swapping
providers, handling rate limits and timeouts, retrying safely, falling back to
another model, validating structured output, and observing what happened. This
library solves those problems once, behind a clean seam, so any project can reuse
it without inheriting another project's domain logic.

## Design principles

- **Provider-agnostic core.** The orchestrator depends on a `Provider` protocol,
  never on Gemini or Groq directly. Adding a provider is a new adapter plus a
  registry entry — no core changes.
- **Dumb providers, smart policies.** Providers translate requests/responses and
  map errors; they never decide retries, fallbacks, or routing.
- **Retry ≠ fallback.** Retry re-attempts the same candidate; fallback moves to a
  different candidate. They are separate policies that compose.
- **Not every error is retryable.** A normalized error taxonomy drives
  retry/fallback/fail-fast decisions instead of blind retrying.
- **LangGraph is an engine, not the architecture.** The run semantics are
  engine-agnostic; LangGraph is the V1 control-flow implementation.
- **No abstraction without a second consumer.** Interfaces appear only where a
  second implementation is genuinely expected.
- **Secrets stay in the environment.** Never hardcoded, never logged.

## Planned capabilities (V1)

Provider abstraction · Gemini and Groq adapters · model selection · timeout
handling · retry with exponential backoff + jitter · model/provider fallback ·
normalized error classification · structured output + schema validation ·
LangGraph orchestration · configuration management · basic observability · a
strong offline test suite.

## Documentation

- [`docs/architecture.md`](docs/architecture.md) — boundaries, core abstractions,
  provider/retry/fallback design, error taxonomy, LangGraph role, structured
  output, observability, configuration, security, testing, V1 scope.
- [`docs/roadmap.md`](docs/roadmap.md) — the phase-by-phase plan from foundation
  to production readiness.

## Roadmap at a glance

```
Empty repo → Architecture (Phase 0)
           → Package foundation → Contracts → Provider abstraction
           → Gemini → Groq → Retry → Error classification → Fallback
           → LangGraph orchestration → Structured output → Routing
           → Observability → Testing → Packaging
           → ReviewIQ integration → Second-project portability test
           → Production hardening → Final E2E validation
```

## Getting started

Not yet runnable. The package foundation (packaging, tooling, and CI) arrives in
Phase 1; see [`docs/roadmap.md`](docs/roadmap.md).

## License

Not yet chosen. A `LICENSE` file will be added once the licensing decision is
made (see `docs/architecture.md` §18).
