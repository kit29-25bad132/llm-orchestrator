# Testing

Status: **Phase 13.** The offline suite is the primary quality gate: it is
deterministic, requires no credentials, performs no network calls, and never
waits on real time. Live provider tests exist but are opt-in and skipped by
default, so normal development consumes no API quota.

## Layout

```
tests/
├── support/                          # shared test-only helpers (Phase 13)
│   ├── __init__.py                   # re-exports the helpers
│   └── fakes.py                      # scripted providers, recording sleep,
│                                     # builders, scripted SDK clients
├── unit/
│   ├── test_package.py               # foundation tests
│   ├── contracts/                    # validation, serialization, invariants,
│   │                                 # plus matrix-completion gaps
│   ├── providers/                    # protocol, registry, Gemini/Groq
│   │                                 # adapters, error/consistency matrices
│   ├── reliability/                  # retry, fallback, failure simulation
│   ├── routing/                      # static policy, router, invariants
│   ├── orchestration/                # state, nodes, graph, invariants
│   ├── structured/                   # validation behavior, invariants
│   └── config/                       # settings, loader, registry, invariants
└── integration/
    ├── test_offline_end_to_end.py    # offline composed path (Phase 13)
    ├── test_gemini_live.py           # opt-in live test (marked `live`)
    └── test_groq_live.py             # opt-in live test (marked `live`)
```

`pyproject.toml` sets `pythonpath = ["tests"]` so helpers are imported
explicitly as `support.fakes` — there is no top-level `conftest.py` with
hidden fixtures. Existing suites keep their local doubles; new tests reuse
`support.fakes` instead of defining another scripted provider.

## Shared helpers (`tests/support/fakes.py`)

- `ScriptedProvider(provider_id, outcomes)` — a fake `LLMProvider` replaying
  scripted `LLMResponse`/`BaseException` outcomes per call, recording
  `request`, `model`, and `timeout_s`. When the script drains, the final
  outcome repeats so over-budget loops still terminate.
- `RecordingSleep` — an injectable async sleep recording delays instead of
  waiting; pass `sleep=recorder.sleep` to `RetryEngine`.
- Builders — `make_request`, `make_model`, `make_response`, `make_registry`
  (fresh independent registry), and `run_sync` (drives one coroutine).
- `FakeGeminiSdkClient` / `FakeGroqSdkClient` — scripted stand-ins exposing
  only the SDK seam each real adapter calls (`aio.models.generate_content` /
  `chat.completions.create`). Outcomes are real SDK response/error types, so
  adapter normalization is genuinely exercised while staying offline.
- Assertion helper — `attempt_sequence` projects an `OrchestrationResult`'s
  attempt history for order-sensitive assertions.

Helpers are stateful per instance: construct fresh ones per test (never
module-level) so tests stay isolated and order-independent.

## Offline end-to-end (`tests/integration/test_offline_end_to_end.py`)

Not marked `live`: it runs in the default suite. It composes the existing
public APIs — `OrchestratorSettings` → `load_settings` (injected env mapping)
→ `build_registry` (injected fake SDK clients) → `ModelRouter` →
`RetryEngine` + `FallbackExecutor` → `OrchestrationResult` — plus a
single-candidate `Orchestrator` run over the same registry. The main test
fails the preferred provider with scripted HTTP 503s (exhausting its retry
budget) and recovers through the alternative provider, asserting selection,
invocation counts, attempt order, retry/fallback counts, timeout propagation
to both SDK seams, request/model/provider identity, secret redaction, and the
config-layer timeout precedence. It never mocks the orchestrator itself.

## Boundaries

| Suite | Marker | Network | Credentials | When it runs |
| --- | --- | --- | --- | --- |
| `tests/unit/**` | none | never | never | default (`pytest`) |
| `test_offline_end_to_end.py` | none | never | never (fake key shapes) | default (`pytest`) |
| `test_*_live.py` | `live` | yes (real APIs) | yes (env vars) | only with `pytest -m live` + env |

Live tests skip unless explicitly enabled:

```bash
RUN_GEMINI_INTEGRATION=1 GEMINI_API_KEY=... GEMINI_TEST_MODEL=... pytest -m live
RUN_GROQ_INTEGRATION=1 GROQ_API_KEY=... GROQ_TEST_MODEL=... pytest -m live
```

## Quality gates

From the repository root:

```bash
python -m compileall -q src
ruff check src tests
ruff format --check src tests
mypy src
pytest -q
git diff --check
```

CI (`.github/workflows/ci.yml`) runs install plus `ruff check .`,
`ruff format --check .`, `mypy`, and `pytest` across Python 3.11–3.14 with no
secrets configured, so the offline suite must stay green without credentials
or network access. All backoff/wait coverage uses injected sleeps and seeded
or disabled jitter — no real sleeps, no wall-clock assertions.
