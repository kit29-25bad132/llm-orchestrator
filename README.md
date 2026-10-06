# llm-orchestrator

A reusable, **provider-agnostic** LLM orchestration library — provider and model
abstraction, routing, retries with backoff/jitter, model- and provider-level
fallback, normalized error classification, structured output with schema
validation, LangGraph-based orchestration, configuration, and observability.

It is **infrastructure**, not an application: consuming projects (such as
ReviewIQ) bring their own prompts, schemas, and business logic.

> **Status: Phase 3 — provider abstraction.**
> The package installs and exposes a provider-neutral **contract layer**
> (`llm_orchestrator.contracts`), a normalized **error taxonomy**
> (`llm_orchestrator.errors`), and the **provider abstraction + registry**
> (`llm_orchestrator.providers`). **No concrete providers, routing, retry,
> fallback, structured generation, or orchestration exist yet** — those consume
> these boundaries in later phases (see [`docs/roadmap.md`](docs/roadmap.md)).

## Why

Applications that call LLMs repeatedly re-solve the same problems: swapping
providers, handling rate limits and timeouts, retrying safely, falling back to
another model, validating structured output, and observing what happened. This
library will solve those problems once, behind a clean seam, so any project can
reuse it without inheriting another project's domain logic.

## Architecture purpose

The library separates concerns along a strict dependency direction:

```
Application (brings its own business logic / schemas)
    ↓
Orchestrator
    ↓
Policies (routing / reliability / validation)
    ↓
Provider abstraction
    ↓
Concrete providers (added later)
```

Design principles:

- **Provider-agnostic core.** The orchestrator will depend on a `Provider`
  protocol, never on a specific vendor.
- **Dumb providers, smart policies.** Providers translate; policies decide
  retries, fallbacks, and routing.
- **Retry ≠ fallback.** Retry re-attempts the same candidate; fallback moves to
  a different one.
- **Not every error is retryable.** A normalized taxonomy drives decisions
  instead of blind retrying.
- **No abstraction without a second consumer.**
- **Secrets stay in the environment** — never hardcoded, never logged.

See [`docs/architecture.md`](docs/architecture.md) for the full design.

## Core contracts (Phase 2)

The contract layer is the stable boundary between applications and the
orchestration infrastructure. It is provider-neutral, application-neutral, and
its only runtime dependency is Pydantic.

```python
from llm_orchestrator.contracts import LLMRequest, Message, MessageRole

request = LLMRequest(messages=[Message(role=MessageRole.USER, content="Hello")])
```

Contract groups:

- **Conversation:** `Message`, `MessageRole`
- **Request:** `LLMRequest` (expresses intent only — never names a provider/model)
- **Response:** `LLMResponse`, `Usage`, `FinishReason`
- **Provider/model descriptors:** `ModelConfig`, `ModelCapabilities`,
  `ModelPricing`, `ProviderConfig`, `CandidateRef`
- **Policies:** `RetryPolicy`, `FallbackPolicy`, `RoutingPolicy` (protocol),
  `RoutingContext`
- **Outcome:** `OrchestrationResult`, `AttemptRecord`, `ErrorInfo`

Errors live in `llm_orchestrator.errors`: a normalized `LLMError` hierarchy with
one class per `ErrorCategory`, default retry/fallback semantics, and message
sanitization so credentials cannot leak into logs.

```python
from llm_orchestrator.errors import ErrorCategory, LLMError, RateLimitError
```

Pydantic is the single runtime dependency: it provides validation, immutability,
and predictable serialization for the contracts.

## Provider abstraction (Phase 3)

Providers plug into a tiny, SDK-free boundary. A provider receives a normalized
`LLMRequest` plus the target `ModelConfig` and returns a normalized `LLMResponse`
— vendor SDK types never cross the boundary. The target model is passed
explicitly because `LLMRequest` carries intent only and never names a provider.

```python
from llm_orchestrator.providers import LLMProvider, ProviderRegistry


class MyProvider:  # satisfies LLMProvider structurally
    provider_id = "my-vendor"

    async def generate(self, request, model, *, timeout_s=None):
        ...  # exactly one attempt; raise llm_orchestrator.errors types on failure

    async def aclose(self) -> None:
        ...


registry = ProviderRegistry()
registry.register(MyProvider())
registry.get("my-vendor")
```

- `LLMProvider` is a `Protocol` (structural typing), so providers do not inherit
  from the library and test doubles stay trivial.
- `ProviderRegistry` maps provider id → instance. Duplicate ids are rejected
  unless `replace=True`; unknown ids raise `ProviderNotRegisteredError`.
- Capabilities stay **model-level** (`ModelCapabilities` on `ModelConfig`), so
  there is a single source of truth; the provider interface is not a metadata
  dump.
- Concrete providers (Gemini, Groq, …) arrive in Phases 4–5; the registry never
  special-cases a vendor.

## Project structure

```
llm-orchestrator/
├── src/
│   └── llm_orchestrator/
│       ├── __init__.py         # package identity only (version)
│       ├── contracts/          # provider-neutral contracts
│       │   ├── __init__.py     # curated public exports
│       │   ├── messages.py     # Message, MessageRole
│       │   ├── requests.py     # LLMRequest
│       │   ├── responses.py    # LLMResponse, Usage, FinishReason
│       │   ├── models.py       # ModelConfig, ProviderConfig, capabilities
│       │   ├── policies.py     # Retry/Fallback/Routing policy contracts
│       │   └── results.py      # OrchestrationResult, AttemptRecord, ErrorInfo
│       ├── errors/             # normalized error taxonomy
│       │   ├── __init__.py
│       │   ├── categories.py   # ErrorCategory + default semantics
│       │   ├── base.py         # LLMError + message sanitization
│       │   └── exceptions.py   # one error class per category
│       └── providers/          # provider abstraction
│           ├── __init__.py     # curated public exports
│           ├── base.py         # LLMProvider protocol
│           └── registry.py     # ProviderRegistry
├── tests/
│   └── unit/
│       ├── test_package.py     # foundation tests
│       ├── contracts/          # contract, serialization, invariant tests
│       └── providers/          # abstraction, registry, invariant tests
├── docs/
│   ├── architecture.md
│   └── roadmap.md
├── .github/
│   └── workflows/
│       └── ci.yml          # install + ruff + mypy + pytest
├── .env.example
├── .gitignore
├── pyproject.toml
└── README.md
```

The package uses the modern `src/` layout so tests run against the installed
package rather than an accidentally-imported source directory.

## Development setup

Requires **Python 3.11 or newer** (see [Python version policy](#python-version-policy)).

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

macOS / Linux:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

The virtual environment (`.venv/`), caches, build artifacts, and local `.env`
files are git-ignored. Never commit the environment or real credentials.

## Running tests

```bash
pytest
```

Optional quality checks (also run in CI):

```bash
ruff check .
ruff format --check .
mypy
```

Tests are grouped by kind under `tests/`. Provider/API-calling tests will be
marked `live` and skipped by default so normal development consumes no API
quota.

## Python version policy

- **Supported:** `>=3.11` (enforced via `requires-python`, so installation fails
  clearly on an older interpreter).
- **Floor rationale:** 3.11 provides the modern typing features the design
  relies on while keeping compatibility broad.
- **Development interpreter:** validated on Python 3.14.0 locally, with CI
  covering 3.11–3.14.

## Current phase

**Phase 3 — Provider Abstraction.** The `LLMProvider` protocol and a
`ProviderRegistry`, plus a test-only fake provider proving the boundary works
entirely offline. No concrete provider, retry, fallback, or orchestration
behavior is implemented yet.

## Roadmap at a glance

```
Empty repo → Architecture (Phase 0) → Package foundation (Phase 1)
           → Contracts (Phase 2) → Provider abstraction (Phase 3, current)
           → Gemini → Groq → Retry
           → Error classification → Fallback → LangGraph orchestration
           → Structured output → Routing → Observability → Testing
           → Packaging → ReviewIQ integration → Portability test
           → Production hardening → Final E2E validation
```

High-level future capabilities: provider/model abstraction, routing, retries
with backoff and jitter, model- and provider-level fallback, normalized error
classification, structured output with validation, LangGraph orchestration,
configuration, and observability. Details in
[`docs/roadmap.md`](docs/roadmap.md).

## Development / Git workflow

This repository is developed phase by phase. The agent does not perform Git
mutations — the human developer reviews `git status` and `git diff`, then
commits each completed phase manually.

## License

Not yet chosen. A `LICENSE` file will be added once the licensing decision is
made (see [`docs/architecture.md`](docs/architecture.md) §18).
