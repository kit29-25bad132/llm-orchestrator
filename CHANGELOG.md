# Changelog

All notable changes to `llm-orchestrator` are documented here. Versions follow
a simple `0.x` scheme while the API matures; per the roadmap, public API churn
requires a version bump and a note here. Dates are intentionally omitted until
a version is actually released — nothing below has been published to a package
index.

## 0.1.0 — Unreleased

### Added

- **Contracts** (`llm_orchestrator.contracts`): provider-neutral messages,
  requests, responses, model/provider descriptors, policies, and outcome types
  (`LLMRequest`, `LLMResponse`, `ModelConfig`, `RetryPolicy`, `FallbackPolicy`,
  `OrchestrationResult` outcome contracts), frozen and serializable.
- **Error taxonomy** (`llm_orchestrator.errors`): one `LLMError` subclass per
  `ErrorCategory` with normalized retry/fallback semantics and credential
  redaction in messages.
- **Provider abstraction + adapters** (`llm_orchestrator.providers`): the
  structural `LLMProvider` protocol, `ProviderRegistry`, and the concrete
  Gemini (`providers.gemini`) and Groq (`providers.groq`) adapters with lazy
  SDK-client creation and full error normalization.
- **Reliability** (`llm_orchestrator.reliability`): `RetryEngine` with
  exponential backoff, jitter strategies, and `retry_after` handling;
  `FallbackExecutor` for deterministic, bounded candidate transitions.
- **Routing** (`llm_orchestrator.routing`): `ModelRouter` with
  `StaticRoutingPolicy` — eligibility filtering, deterministic ranking, and
  explainable `RoutingDecision` output.
- **Structured output** (`llm_orchestrator.structured`):
  `validate_structured_output` for Pydantic-backed validation of provider text.
- **LangGraph orchestration** (`llm_orchestrator.orchestration`): a
  deterministic single-candidate `Orchestrator` workflow
  (`START → execute_model → END`).
- **Configuration** (`llm_orchestrator.config`): `OrchestratorSettings`,
  `load_settings`, and `build_registry` with explicit-first precedence and
  environment-resolved, never-stored credentials.
- **Test suite**: layered offline unit/integration tests (696 passing on the
  Phase 13 baseline), shared test doubles under `tests/support/`, opt-in
  `live`-marked provider tests, and failure-simulation coverage.
- **Documentation**: per-layer docs under `docs/`, architecture decisions, and
  the phase roadmap.

### Changed (packaging, Phase 14)

- Dependency policy split: the default install now requires **only Pydantic**;
  `google-genai`, `groq`, and `langgraph` moved to the optional `gemini`,
  `groq`, and `langgraph` extras, with `all` (every optional runtime
  capability) and `dev` (pytest, ruff, mypy, build) convenience extras.
- Optional features fail with an actionable `ConfigurationError` naming the
  exact `pip install "llm-orchestrator[...]"` command when their extra is
  missing; provider adapters and LangGraph are imported lazily.
- The package ships a PEP 561 `py.typed` marker, matching its
  `Typing :: Typed` classifier.
- Wheel and sdist builds (`python -m build`) are validated for contents and
  metadata (`scripts/validate_dist.py`), by clean-environment install smokes
  (`scripts/smoke_install.py`), and by a dedicated CI `package` job across
  Python 3.11–3.14.

### Compatibility

- Python: `>=3.11` (CI-verified matrix 3.11–3.14; the local development
  interpreter is 3.14).
- Public API: unchanged since Phase 13 — top-level `llm_orchestrator` exposes
  identity only (`__version__`); all layers keep their established import
  paths, names, and semantics.
- License: MIT, with `Copyright (c) 2026 RISHI V.` — see
  [`LICENSE`](LICENSE).
