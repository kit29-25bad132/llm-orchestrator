# llm-orchestrator

A reusable, **provider-agnostic** LLM orchestration library — provider and model
abstraction, routing, retries with backoff/jitter, model- and provider-level
fallback, normalized error classification, structured output with schema
validation, LangGraph-based orchestration, configuration, and observability.

It is **infrastructure**, not an application: consuming projects (such as
ReviewIQ) bring their own prompts, schemas, and business logic.

> **Status: Phase 1 — package foundation only.**
> The project is an installable, tested, lint-checked Python package that
> exposes package identity and nothing else. **No providers, routing, retry,
> fallback, structured output, or orchestration exist yet** — those arrive in
> later phases (see [`docs/roadmap.md`](docs/roadmap.md)).

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

## Project structure

```
llm-orchestrator/
├── src/
│   └── llm_orchestrator/
│       └── __init__.py     # package identity only (version)
├── tests/
│   └── unit/
│       └── test_package.py # foundation tests
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

**Phase 1 — Package Foundation.** Installable package skeleton, packaging
metadata, dev tooling, foundation tests, and environment template. No LLM
behavior is implemented.

## Roadmap at a glance

```
Empty repo → Architecture (Phase 0) → Package foundation (Phase 1, current)
           → Contracts → Provider abstraction → Gemini → Groq → Retry
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
