# Testing

Status: **Phase 14.** The offline suite (hardened in Phase 13) is the primary
quality gate: it is
deterministic, requires no credentials, performs no network calls, and never
waits on real time. Live provider tests exist but are opt-in and skipped by
default, so normal development consumes no API quota. Packaging gates (build,
artifact validation, clean-install smokes) complement it and are documented
below.

## Layout

```
tests/
├── support/                          # shared test-only helpers (Phase 13)
│   ├── __init__.py                   # re-exports the helpers
│   └── fakes.py                      # scripted providers, recording sleep,
│                                     # builders, scripted SDK clients
├── unit/
│   ├── test_package.py               # foundation tests + py.typed marker
│   ├── test_optional_extras.py       # missing-extra errors, lazy imports
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

## Packaging gates (Phase 14)

Offline artifact checks, run from the repository root (also executed by the CI
`package` job on Python 3.11–3.14):

```bash
python -m build                                        # wheel + sdist into dist/
python scripts/validate_dist.py dist                   # contents + metadata contract
python -m pip wheel dist/*.tar.gz --no-deps -w dist/from-sdist   # sdist -> wheel
python scripts/validate_dist.py dist/from-sdist --wheel-only
```

Clean-environment install smoke — the wheel is installed into a **fresh
temporary virtual environment** (not the repository `.venv`), then the script
runs from a working directory outside the repository so imports can only come
from the installed artifact:

```bash
python -m venv /tmp/core-venv
/tmp/core-venv/bin/python -m pip install dist/llm_orchestrator-0.1.0-py3-none-any.whl
cd /tmp && /tmp/core-venv/bin/python /path/to/repo/scripts/smoke_install.py
# with every optional feature:
/tmp/core-venv/bin/python -m pip install "llm-orchestrator[all] @ file:///path/to/dist/llm_orchestrator-0.1.0-py3-none-any.whl"
cd /tmp && /tmp/core-venv/bin/python /path/to/repo/scripts/smoke_install.py \
  --expect-extras gemini groq langgraph
```

What each gate proves:

- `validate_dist.py` — the wheel contains exactly the `src/llm_orchestrator`
  files (including `py.typed`), no tests/docs/env/cache material, and metadata
  matching `pyproject.toml` (name, version, `requires-python`, `Typing ::
  Typed`, core/extra dependency sets, and the MIT license expression/file,
  with `LICENSE` shipped at the sdist root); the sdist matches the declared
  sdist policy and can rebuild the wheel.
- `smoke_install.py` — the installed distribution (resolved from
  `site-packages`, never the source checkout) imports all core subpackages,
  exposes the frozen public API and version, ships `py.typed`, reports each
  optional feature as either constructible (extra installed) or failing with
  the actionable `ConfigurationError` naming its extra (extra absent), and
  confirms no optional dependency leaks into a core-only environment.
- `tests/unit/test_optional_extras.py` — the same missing-extra and lazy-import
  guarantees inside the offline suite, simulated by hiding import roots in a
  child interpreter (no network, no credentials).

A consumer typing smoke (mypy against an installed environment) is run as part
of release validation: with a wheel installed in an environment, `mypy --strict`
on a one-file consumer must resolve package types via the `py.typed` marker.
