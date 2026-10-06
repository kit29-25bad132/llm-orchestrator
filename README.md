# llm-orchestrator

A reusable, **provider-agnostic** LLM orchestration library — provider and model
abstraction, routing, retries with backoff/jitter, model- and provider-level
fallback, normalized error classification, structured output with schema
validation, LangGraph-based orchestration, configuration, and observability.

It is **infrastructure**, not an application: consuming projects (such as
ReviewIQ) bring their own prompts, schemas, and business logic.

> **Status: Phase 5 — Groq provider.**
> The package installs and exposes a provider-neutral **contract layer**
> (`llm_orchestrator.contracts`), a normalized **error taxonomy**
> (`llm_orchestrator.errors`), the **provider abstraction + registry**
> (`llm_orchestrator.providers`), and two concrete adapters (**Gemini** in
> `llm_orchestrator.providers.gemini`, **Groq** in
> `llm_orchestrator.providers.groq`). **No routing, retry, fallback,
> structured generation, or orchestration exists yet** — those consume these
> boundaries in later phases (see [`docs/roadmap.md`](docs/roadmap.md)).

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
orchestration infrastructure. It is provider-neutral and application-neutral,
and it has no provider SDK dependency.

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

Runtime dependencies are deliberately minimal: **Pydantic** (contract
validation, immutability, and predictable serialization), **`google-genai`**
(the official Gemini SDK, imported only inside the Gemini adapter), and
**`groq`** (the official Groq SDK, imported only inside the Groq adapter).

## Providers (Phase 3 abstraction, Phases 4–5 adapters)

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
- Gemini is implemented in `llm_orchestrator.providers.gemini` and Groq in
  `llm_orchestrator.providers.groq`. The registry never special-cases a vendor.

### Gemini (Phase 4)

The first concrete adapter. It is the only module allowed to import the Google
GenAI SDK, so no SDK type crosses the boundary:

```python
from llm_orchestrator.contracts import LLMRequest, Message, MessageRole, ModelConfig
from llm_orchestrator.providers.gemini import GeminiProvider

provider = GeminiProvider(api_key=os.environ["GEMINI_API_KEY"])
request = LLMRequest(messages=[Message(role=MessageRole.USER, content="Hi")])
model = ModelConfig(provider="gemini", model="<gemini-model-id>")

response = await provider.generate(request, model)  # one attempt, normalized LLMResponse
await provider.aclose()
```

Configuration is injected explicitly (the provider never reads the environment),
the SDK client is created lazily and reused, Gemini errors are normalized into
the library's error hierarchy, and import of the library never touches the SDK
or the network. See [`docs/providers.md`](docs/providers.md).

### Groq (Phase 5)

The second concrete adapter, proving the abstraction needs no core changes for
a structurally different vendor. It is the only module allowed to import the
Groq SDK:

```python
from llm_orchestrator.contracts import LLMRequest, Message, MessageRole, ModelConfig
from llm_orchestrator.providers.groq import GroqProvider

provider = GroqProvider(api_key=os.environ["GROQ_API_KEY"])
request = LLMRequest(messages=[Message(role=MessageRole.USER, content="Hi")])
model = ModelConfig(provider="groq", model="<groq-model-id>")

response = await provider.generate(request, model)  # one attempt, normalized LLMResponse
await provider.aclose()
```

Groq speaks the OpenAI-compatible chat shape, so messages map one-to-one with
ordering preserved; errors are normalized into the same hierarchy, and the
SDK's own retries are disabled (`max_retries=0`) so the single-attempt contract
holds. See [`docs/providers.md`](docs/providers.md).

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
│       └── providers/          # provider abstraction + adapters
│           ├── __init__.py     # curated public exports
│           ├── base.py         # LLMProvider protocol
│           ├── registry.py     # ProviderRegistry
│           ├── gemini.py       # GeminiProvider (Phase 4)
│           └── groq.py         # GroqProvider (Phase 5)
├── tests/
│   ├── unit/
│   │   ├── test_package.py     # foundation tests
│   │   ├── contracts/          # contract, serialization, invariant tests
│   │   └── providers/          # abstraction, registry, Gemini/Groq adapter tests
│   └── integration/
│       ├── test_gemini_live.py # opt-in live test (marked `live`)
│       └── test_groq_live.py   # opt-in live test (marked `live`)
├── docs/
│   ├── architecture.md
│   ├── providers.md            # provider + Gemini/Groq adapter documentation
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

Tests are grouped by kind under `tests/`. The default suite is fully offline.
Provider API-calling tests are marked `live` and skipped unless explicitly
enabled, so normal development consumes no API quota:

```bash
RUN_GEMINI_INTEGRATION=1 GEMINI_API_KEY=... GEMINI_TEST_MODEL=... pytest -m live
RUN_GROQ_INTEGRATION=1 GROQ_API_KEY=... GROQ_TEST_MODEL=... pytest -m live
```

## Python version policy

- **Supported:** `>=3.11` (enforced via `requires-python`, so installation fails
  clearly on an older interpreter).
- **Floor rationale:** 3.11 provides the modern typing features the design
  relies on while keeping compatibility broad.
- **Development interpreter:** validated on Python 3.14.0 locally, with CI
  covering 3.11–3.14.

## Current phase

**Phase 5 — Groq Provider.** The second concrete adapter (`GroqProvider`)
implements the same Phase 3 protocol, proving the abstraction needs no core
changes for a structurally different vendor. Retry, fallback, routing, and
orchestration are still unimplemented.

## Roadmap at a glance

```
Empty repo → Architecture (Phase 0) → Package foundation (Phase 1)
           → Contracts (Phase 2) → Provider abstraction (Phase 3)
            → Gemini (Phase 4) → Groq (Phase 5, current) → Retry
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
