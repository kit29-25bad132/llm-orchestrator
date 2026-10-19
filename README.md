# llm-orchestrator

A reusable, **provider-agnostic** LLM orchestration library — provider and model
abstraction, routing, retries with backoff/jitter, model- and provider-level
fallback, normalized error classification, structured output with schema
validation, LangGraph-based orchestration, and configuration. Observability is
deferred (see [`docs/roadmap.md`](docs/roadmap.md)).

It is **infrastructure**, not an application: consuming projects (such as
ReviewIQ) bring their own prompts, schemas, and business logic.

> **Status: Phase 14 — Packaging & release readiness.**
> The package installs and exposes a provider-neutral **contract layer**
> (`llm_orchestrator.contracts`), a normalized **error taxonomy**
> (`llm_orchestrator.errors`), the **provider abstraction + registry**
> (`llm_orchestrator.providers`), two concrete adapters (**Gemini** in
> `llm_orchestrator.providers.gemini`, **Groq** in
> `llm_orchestrator.providers.groq`), a provider-agnostic **retry engine**
> plus candidate-level **fallback execution** (`llm_orchestrator.reliability`),
> a deterministic **LangGraph orchestration** boundary
> (`llm_orchestrator.orchestration`),
> **Pydantic-native structured-output validation** (`llm_orchestrator.structured`),
> deterministic **routing & model selection** (`llm_orchestrator.routing`),
> and a secure, typed **configuration system** (`llm_orchestrator.config`).
> The default install is core-only (**Pydantic**); provider SDKs and LangGraph
> are optional extras — see [Installation](#installation).
> **Observability is deferred** — runs already preserve the attempt history it
> will consume (see [`docs/roadmap.md`](docs/roadmap.md)).

## Why

Applications that call LLMs repeatedly re-solve the same problems: swapping
providers, handling rate limits and timeouts, retrying safely, falling back to
another model, validating structured output, and observing what happened. This
library will solve those problems once, behind a clean seam, so any project can
reuse it without inheriting another project's domain logic.

## Installation

Requires **Python 3.11 or newer**. The package is not published to an index
yet, so install it from a source checkout or from a built wheel. The default
install is **core-only**: contracts, errors, provider abstraction, retry,
fallback, routing, structured output, and configuration — needing only
**Pydantic**. Provider SDKs and LangGraph arrive through extras, so consumers
install only what they use.

From a source checkout:

```bash
python -m pip install .                 # core only
python -m pip install ".[gemini]"       # + Gemini adapter (google-genai)
python -m pip install ".[groq]"         # + Groq adapter (groq)
python -m pip install ".[langgraph]"    # + LangGraph orchestration
python -m pip install ".[all]"          # every optional runtime capability
```

From a built wheel (`python -m build` produces `dist/`):

```bash
python -m pip install dist/llm_orchestrator-0.1.0-py3-none-any.whl
```

Extras are additive and never required by the core:

| Extra | Enables | Installs |
| --- | --- | --- |
| *(none)* | contracts, errors, provider abstraction, retry/fallback, routing, structured output, configuration | `pydantic` |
| `gemini` | `llm_orchestrator.providers.gemini.GeminiProvider`, Gemini entries in `build_registry` | `google-genai`, `httpx` |
| `groq` | `llm_orchestrator.providers.groq.GroqProvider`, Groq entries in `build_registry` | `groq`, `httpx` |
| `langgraph` | `llm_orchestrator.orchestration` (`Orchestrator`) | `langgraph` |
| `all` | all of the above | union of the three |
| `dev` | test/lint/type/build toolchain for a checkout (`pytest`, `ruff`, `mypy`, `build`) | tools only |

Using an optional feature without its extra raises `ConfigurationError` with
the exact `pip install "llm-orchestrator[...]"` command — never an opaque
`ImportError`. For development from a checkout, see
[Development setup](#development-setup).

## Quickstart

**Core (no extras, no credentials, no network).** Express intent, pick a
candidate, and wrap any provider in a retry policy:

```python
from llm_orchestrator.contracts import (
    LLMRequest,
    Message,
    MessageRole,
    ModelConfig,
    RetryPolicy,
)
from llm_orchestrator.reliability import RetryEngine
from llm_orchestrator.routing import ModelRouter, StaticRoutingPolicy

# 1. Provider-neutral intent.
request = LLMRequest(messages=[Message(role=MessageRole.USER, content="Hello")])

# 2. Deterministic initial-candidate selection.
candidates = [
    ModelConfig(provider="gemini", model="<gemini-model-id>"),
    ModelConfig(provider="groq", model="<groq-model-id>"),
]
router = ModelRouter(StaticRoutingPolicy(preferred_provider="gemini"))
selected = router.select(request, candidates)  # -> ModelConfig

# 3. Bounded retries around any object satisfying the LLMProvider protocol.
engine = RetryEngine(RetryPolicy(max_attempts=3, initial_backoff_s=0.5))
```

`selected` flows straight into `engine.generate(provider, request, selected)`
— `provider` can be anything structural (see
[Providers](#providers-phase-3-abstraction-phases-45-adapters)) or a concrete
adapter from an extra.

**Calling a provider (extra required).** This needs
`pip install "llm-orchestrator[gemini]"` (or `[groq]`) and a credential in the
environment — never hard-coded:

```python
import os

from llm_orchestrator.providers.gemini import GeminiProvider  # extra: [gemini]

provider = GeminiProvider(api_key=os.environ["GEMINI_API_KEY"])  # GEMINI_API_KEY=<your key>
response = await provider.generate(request, selected)  # one live API call
await provider.aclose()
```

Construction and `import llm_orchestrator.providers.gemini` never touch the
network or read the environment; only `generate` performs a real call. The
LangGraph workflow additionally needs `pip install "llm-orchestrator[langgraph]"`
(`from llm_orchestrator.orchestration import Orchestrator`) — see
[`docs/orchestration.md`](docs/orchestration.md), and
[`docs/configuration.md`](docs/configuration.md) for settings-driven
`build_registry` setups.

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

See [`docs/errors.md`](docs/errors.md) for the full classification model,
the category matrix, and the per-provider mapping tables.

Runtime dependencies are deliberately minimal: the core install depends only
on **Pydantic** (contract validation, immutability, and predictable
serialization). The official provider SDKs (**`google-genai`**, **`groq`**)
and **LangGraph** are optional extras imported only inside their own adapter /
orchestration modules — see [Installation](#installation).

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

The first concrete adapter (requires the `gemini` extra). It is the only
module allowed to import the Google GenAI SDK, so no SDK type crosses the
boundary:

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

The second concrete adapter (requires the `groq` extra), proving the
abstraction needs no core changes for a structurally different vendor. It is
the only module allowed to import the
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

## Configuration (Phase 12)

`OrchestratorSettings` (`llm_orchestrator.config`) is the single validated
object describing providers, model candidates, retry/fallback/routing
policies, and an optional library-level default timeout. It reuses the
existing contract types, holds no secret values, and reads nothing from the
environment on construction:

```python
import os

from llm_orchestrator.config import (
    OrchestratorSettings,
    build_registry,
    load_settings,
)
from llm_orchestrator.contracts import ModelConfig, ProviderConfig

settings = load_settings(
    explicit=OrchestratorSettings(
        providers=(ProviderConfig(name="gemini", credential_env_var="GEMINI_API_KEY"),),
        models=(ModelConfig(provider="gemini", model="<gemini-model-id>"),),
    ),
    env=os.environ,
)
registry = build_registry(settings, env=os.environ)
```

- Explicit values always win; the environment only fills
  `default_timeout_s` (via `LLM_ORCH_DEFAULT_TIMEOUT_S`) when explicit
  settings leave it unset. Only `GEMINI_API_KEY`, `GROQ_API_KEY`, and
  `LLM_ORCH_DEFAULT_TIMEOUT_S` belong to the library surface, and no `.env`
  file is ever loaded automatically.
- Credentials are resolved inside `build_registry` from each provider's own
  `credential_env_var`, passed straight to the provider constructor, and
  never stored — missing values raise `ConfigurationError` naming the
  variable, never the value. `import llm_orchestrator.config` needs no keys
  and loads no provider SDKs. A provider configured without its extra installed
  raises `ConfigurationError` naming the `pip install "llm-orchestrator[...]"`
  command; a Gemini-only setup never imports the Groq adapter (and vice
  versa).

See [`docs/configuration.md`](docs/configuration.md).

## Reliability — retry (Phase 6)

`RetryEngine` (`llm_orchestrator.reliability`) wraps any `LLMProvider` with
bounded retries of the **same** provider, model, and request. It is fully
provider-agnostic: it decides from normalized `LLMError` retryability only.

```python
from llm_orchestrator.contracts import RetryPolicy
from llm_orchestrator.reliability import RetryEngine

engine = RetryEngine(RetryPolicy(max_attempts=3, initial_backoff_s=0.5))
response = await engine.generate(provider, request, model)  # same candidate, retried
```

- `max_attempts` counts total provider attempts (`1` = one attempt, no retry).
- Only retryable errors are retried (rate limits, timeouts, network and
  provider outages); auth, invalid-request, and unknown failures propagate at
  once, as do unexpected exceptions and cancellation.
- Waits follow exponential backoff capped at `max_backoff_s`, with the
  policy's jitter strategy and honoring `RateLimitError.retry_after_s`.
- Exhaustion re-raises the original normalized error (with `attempt` stamped),
  never a replacement. Moving to another candidate after that terminal failure
  is fallback's job (Phase 11, below), never the retry engine's.

See [`docs/reliability.md`](docs/reliability.md).

## Fallback execution (Phase 11)

`FallbackExecutor` (`llm_orchestrator.reliability`) moves execution to another
candidate only after `RetryEngine` reaches a terminal failure for the current
one. It is a reusable service around the retry engine — the LangGraph workflow
stays single-candidate:

```python
from llm_orchestrator.reliability import FallbackExecutor

executor = FallbackExecutor(registry, retry_engine, fallback_policy)
result = await executor.run(request, candidates, initial=selected)
```

- Routing chooses the initial candidate; retry re-attempts the same candidate;
  fallback advances to the next one on a fallback-eligible normalized error.
- Order is deterministic (initial first, duplicates collapsed, routing order
  preserved) and bounded: `max_fallbacks=0` means the initial candidate only,
  `max_fallbacks=2` allows at most `A → B → C`. (`max_attempts` counts attempts
  on one candidate; `max_fallbacks` counts candidate transitions.)
- Success stops immediately; all-candidate failure returns the last terminal
  error with the complete per-call attempt history. Cancellation and
  programming errors propagate untouched and never trigger a transition.

See [`docs/fallback.md`](docs/fallback.md).

## Routing & model selection (Phase 10)

`ModelRouter` (`llm_orchestrator.routing`) answers one question — **which
model should we attempt first?** — by applying a declarative
`StaticRoutingPolicy` to an explicit `ModelConfig` candidate set. It returns
exactly one initial candidate and never calls providers, retries, or falls
back:

```python
from llm_orchestrator.contracts import ModelConfig
from llm_orchestrator.routing import ModelRouter, StaticRoutingPolicy

candidates = [
    ModelConfig(provider="provider-a", model="model-x"),
    ModelConfig(provider="provider-b", model="model-y"),
]

router = ModelRouter(StaticRoutingPolicy(preferred_provider="provider-a"))
selected = router.select(request, candidates)  # a ModelConfig
```

- Eligibility filtering first (`enabled`, allowlists, explicit-model
  requirement, required capabilities — a request carrying
  `response_schema` requires `structured_output` — minimum context), then
  deterministic ranking (explicit match, provider/model preference,
  estimated cost when asked with explicit token estimates, `quality_tier`
  priority, lexicographic tiebreak).
- An explicit `required_model` is never silently substituted; a missing or
  ineligible requirement raises `ConfigurationError` (no retry, no
  fallback) instead of picking something else.
- Cost comparison uses only configured `ModelPricing` plus the policy's own
  token estimates — never invented counts, never post-generation usage.
- `router.explain(...)` returns a `RoutingDecision` (selected, eligible,
  policy, `SelectionReason`) for debugging and tests. The selected
  candidate flows into the existing path unchanged:
  `router.select(...)` → `RetryEngine.generate(...)` → provider.

See [`docs/routing.md`](docs/routing.md).

## Orchestration (Phase 8)

`Orchestrator` (`llm_orchestrator.orchestration`, requires the `langgraph`
extra) runs one model candidate
through a deterministic LangGraph workflow (`START → execute_model → END`).
The graph owns workflow state and transitions; providers, retries, and error
semantics stay in their existing layers.

```python
from llm_orchestrator.orchestration import Orchestrator

orchestrator = Orchestrator(registry, retry_engine)  # graph compiled once
result = await orchestrator.run(request, model)  # OrchestrationResult
```

- The `execute_model` node resolves the provider from the registry and makes
  exactly one `RetryEngine.generate` call — no reimplemented retries.
- Success yields `OrchestrationResult` with the `LLMResponse`; normalized
  `LLMError` yields one with a sanitized `ErrorInfo`. Bugs and cancellation
  propagate untouched.
- One candidate only: no routing, validation, or multi-agent behavior.
  Multi-candidate fallback lives in `FallbackExecutor` (Phase 11), a reusable
  service around the retry engine — the graph itself is unchanged.

See [`docs/orchestration.md`](docs/orchestration.md).

## Structured output (Phase 9)

`validate_structured_output` (`llm_orchestrator.structured`) validates
normalized `LLMResponse` text against a caller-owned Pydantic model and
returns the typed instance — pure local work, no model calls, no repair:

```python
from pydantic import BaseModel

from llm_orchestrator.structured import validate_structured_output


class ShipmentLabel(BaseModel):
    recipient: str
    zip_code: str


label = validate_structured_output(response, ShipmentLabel)
```

- Accepts plain JSON or one complete ` ```json ` fenced block; prose,
  truncated payloads, and wrong shapes fail explicitly.
- Failures raise the normalized `StructuredOutputValidationError` with
  provider/model/request context and safe diagnostics — never raw payloads
  or secrets.
- Extra fields follow the caller's own model configuration; provider
  adapters still reject `response_schema` (native structured modes are
  future work).

See [`docs/structured-output.md`](docs/structured-output.md).

## Project structure

```
llm-orchestrator/
├── src/
│   └── llm_orchestrator/
│       ├── __init__.py         # package identity only (version)
│       ├── py.typed            # PEP 561 marker (ships in the wheel)
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
│       └── reliability/        # retry engine + fallback executor (Phases 6, 11)
│           ├── __init__.py     # curated public exports
│           ├── retry.py        # RetryEngine
│           ├── backoff.py      # backoff + jitter math
│           └── fallback.py     # FallbackExecutor
│       └── routing/            # routing & model selection (Phase 10)
│           ├── __init__.py     # curated public exports
│           ├── static.py       # StaticRoutingPolicy + estimate_cost
│           └── router.py       # ModelRouter + RoutingDecision
│       └── orchestration/      # LangGraph workflow (Phase 8)
│           ├── __init__.py     # curated public exports (Orchestrator)
│           ├── state.py        # OrchestrationState + initial_state
│           ├── nodes.py        # execute_model node
│           └── graph.py        # build_orchestration_graph + Orchestrator
│       └── structured/         # structured-output validation (Phase 9)
│           ├── __init__.py     # curated public exports
│           └── pydantic_adapter.py  # validate_structured_output
│       └── config/             # configuration system (Phase 12)
│           ├── __init__.py     # curated public exports
│           ├── settings.py     # OrchestratorSettings
│           └── loader.py       # load_settings + build_registry
├── tests/
│   ├── support/                  # shared test-only helpers (fakes, builders)
│   ├── unit/
│   │   ├── test_package.py     # foundation tests (incl. py.typed marker)
│   │   ├── test_optional_extras.py  # missing-extra + lazy-import behavior
│   │   ├── contracts/          # contract, serialization, invariant tests
│   │   ├── providers/          # abstraction, registry, Gemini/Groq adapter tests
│   │   ├── reliability/        # retry + fallback + failure-simulation tests
│   │   ├── routing/            # static routing + router + invariant tests
│   │   ├── orchestration/      # state, nodes, graph + invariant tests
│   │   ├── structured/         # validation behavior + invariant tests
│   │   └── config/             # settings, loader, registry + invariant tests
│   └── integration/
│       ├── test_offline_end_to_end.py  # offline composed path (config →
│       │                              # registry → routing → retry/fallback)
│       ├── test_gemini_live.py # opt-in live test (marked `live`)
│       └── test_groq_live.py   # opt-in live test (marked `live`)
├── docs/
│   ├── architecture.md
│   ├── configuration.md      # configuration system documentation
│   ├── providers.md            # provider + Gemini/Groq adapter documentation
│   ├── reliability.md          # retry engine documentation
│   ├── fallback.md             # fallback execution documentation
│   ├── routing.md              # routing & model selection documentation
│   ├── errors.md               # normalized error classification model
│   ├── orchestration.md        # LangGraph workflow documentation
│   ├── structured-output.md    # structured-output validation documentation
│   ├── testing.md              # test inventory, boundaries, quality gates
│   └── roadmap.md
├── scripts/
│   ├── validate_dist.py        # wheel/sdist contents + metadata checks
│   └── smoke_install.py        # post-install import/extra smoke checks
├── .github/
│   └── workflows/
│       └── ci.yml          # test gates + packaging job (3.11–3.14)
├── .env.example
├── .gitignore
├── CHANGELOG.md
├── pyproject.toml
└── README.md
```

The package uses the modern `src/` layout so tests run against the installed
package rather than an accidentally-imported source directory.

## Development setup

Requires **Python 3.11 or newer** (see [Python version policy](#python-version-policy)).
The commands below install the dev toolchain **plus all optional extras**
(`[dev,all]`), which is what the full test suite needs; `[dev]` alone installs
tools only.

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[dev,all]"
```

macOS / Linux:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev,all]"
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

Packaging checks (also run in the CI `package` job) — build and inspect the
distributions, then install the wheel in a fresh environment:

```bash
python -m build
python scripts/validate_dist.py dist
python -m venv /tmp/smoke-venv
/tmp/smoke-venv/bin/python -m pip install dist/llm_orchestrator-0.1.0-py3-none-any.whl
cd /tmp && /tmp/smoke-venv/bin/python /path/to/repo/scripts/smoke_install.py
```

See [`docs/testing.md`](docs/testing.md) for the full gate list.

Tests are grouped by kind under `tests/`. The default suite is fully offline.
Shared test-only helpers live in `tests/support/` (scripted providers,
recording sleeps, builders, scripted SDK clients) and are imported explicitly
as `support.fakes` — see [`docs/testing.md`](docs/testing.md) for the full
test inventory, the offline end-to-end suite, and the unit/integration/live
boundaries.
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

**Phase 14 — Packaging & release readiness.** The library now ships a
core-only default install (Pydantic alone) with optional `gemini`, `groq`,
`langgraph`, `all`, and `dev` extras declared in `pyproject.toml`; provider
SDKs and LangGraph are imported lazily and missing extras fail with an
actionable `ConfigurationError`. The package carries its PEP 561 `py.typed`
marker, builds wheel and sdist via `python -m build`, and is validated by
content/metadata checks, clean-environment install smokes, and a dedicated CI
`package` job across Python 3.11–3.14. Earlier phases are unchanged: routing
chooses the initial candidate, retry re-attempts it, fallback moves to another
one, and the LangGraph workflow is untouched. Observability remains deferred.
The MIT license ships in the repository as [`LICENSE`](LICENSE).

Release history: [`CHANGELOG.md`](CHANGELOG.md).

## Roadmap at a glance

```
Empty repo → Architecture (Phase 0) → Package foundation (Phase 1)
            → Contracts (Phase 2) → Provider abstraction (Phase 3)
              → Gemini (Phase 4) → Groq (Phase 5) → Retry (Phase 6)
            → Error classification (Phase 7) → LangGraph orchestration (Phase 8)
             → Structured output & validation (Phase 9)
              → Routing & model selection (Phase 10)
              → Fallback execution (Phase 11)
              → Configuration system (Phase 12)
              → Testing hardening (Phase 13)
              → Packaging (Phase 14, current)
           → ReviewIQ integration → Portability test
           → Production hardening → Final E2E validation
Observability: deferred (attempt history already preserved)
```

High-level capabilities: provider/model abstraction, routing, retries
with backoff and jitter, model- and provider-level fallback, normalized error
classification, structured output with validation, LangGraph orchestration,
and configuration; observability remains deferred. Details in
[`docs/roadmap.md`](docs/roadmap.md).

## Development / Git workflow

This repository is developed phase by phase. The agent does not perform Git
mutations — the human developer reviews `git status` and `git diff`, then
commits each completed phase manually.

## License

**MIT** — see [`LICENSE`](LICENSE) for the full text
(`Copyright (c) 2026 RISHI V.`). The license is also declared in
`pyproject.toml` as a PEP 639 SPDX expression and ships in both the wheel and
the sdist.
