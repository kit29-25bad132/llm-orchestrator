# Configuration

Status: **Phase 12.** `llm_orchestrator.config` is the single validated entry
point for application setup. It reuses the Phase 2 contracts
(`ProviderConfig`, `ModelConfig`, `RetryPolicy`, `FallbackPolicy`) and the
Phase 10 routing policy (`StaticRoutingPolicy`) verbatim — no policy shape is
duplicated.

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

## What configuration is

- **One validated object.** `OrchestratorSettings` holds providers, model
  candidates, retry/fallback/routing policies, and an optional
  library-level `default_timeout_s`. It is frozen and rejects unknown fields,
  like every other contract. `OrchestratorSettings()` is valid and means
  "nothing configured": `build_registry` yields an empty registry, and
  candidate selection over the empty model set fails with
  `ConfigurationError` at use time.
- **Explicit first.** Direct construction reads nothing from the environment
  and is the deterministic path tests use.
- **A thin environment overlay.** `load_settings(explicit, *, env)` fills
  exactly one field from the environment — `default_timeout_s` from
  `LLM_ORCH_DEFAULT_TIMEOUT_S` — and only when the explicit settings leave it
  as `None`. `None` means "unset" everywhere: an omitted timeout and an
  explicit `None` are equivalent (Pydantic preserves no distinction), so
  both receive the environment value. There is intentionally no way to force
  "no timeout" against an environment default — omit the variable instead.
  Every field set explicitly is preserved ("explicit wins");
  providers, models, and policies are never merged from the environment.
  Pass `env` explicitly for deterministic tests; when omitted,
  `os.environ` is read inside the function only. No `.env` file is ever
  loaded automatically.
- **A registry factory.** `build_registry(settings, *, env,
  client_factory)` constructs each enabled provider, resolves its
  credential via its own `credential_env_var`, and registers it. Disabled
  providers are skipped; unknown names raise `ConfigurationError`.
  Construction makes no model calls and opens no connections (adapters
  create SDK clients lazily). `client_factory` (a mapping or a callable
  taking the provider name) injects test doubles so registry tests stay
  offline. Every call returns a fresh, independent registry — there is no
  global state.

## Environment surface

Only three variables belong to the library (see `.env.example`):

| Variable | Purpose |
| --- | --- |
| `GEMINI_API_KEY` | Credential for the `gemini` provider, when configured |
| `GROQ_API_KEY` | Credential for the `groq` provider, when configured |
| `LLM_ORCH_DEFAULT_TIMEOUT_S` | Optional library-level default timeout (seconds) |

`GEMINI_TEST_MODEL`, `GROQ_TEST_MODEL`, and `RUN_*_INTEGRATION` remain
test-only and are not part of the runtime settings model.

## Timeout precedence

`resolve_timeout_s(settings, *, provider_name, request, timeout_s)`
implements the single rule, preserving existing provider behavior:

1. Per-call `timeout_s`
2. `LLMRequest.timeout_s`
3. Provider `ProviderConfig.default_timeout_s`
4. Settings `default_timeout_s`
5. `None` (the SDK's existing default)

Pass the result into the existing `timeout_s` seams
(`RetryEngine.generate`, `FallbackExecutor.run`, `Orchestrator.run`, and the
provider `generate` methods).

### Resolution is not enforcement

`resolve_timeout_s` is a pure helper: nothing in the library calls it
automatically, and neither `RetryEngine`, `FallbackExecutor`, nor the
orchestrator reads `OrchestratorSettings`. Declaring
`LLM_ORCH_DEFAULT_TIMEOUT_S` (or `default_timeout_s`) only *supplies* a value
at level 4 of the precedence chain — it becomes an enforced per-attempt
timeout only when your code resolves it and passes it into a `timeout_s` seam:

```python
import asyncio
import os

from llm_orchestrator.config import (
    OrchestratorSettings,
    build_registry,
    load_settings,
    resolve_timeout_s,
)
from llm_orchestrator.contracts import (
    LLMRequest,
    Message,
    MessageRole,
    ModelConfig,
    ProviderConfig,
)
from llm_orchestrator.reliability import FallbackExecutor, RetryEngine

settings = load_settings(
    explicit=OrchestratorSettings(
        providers=(ProviderConfig(name="groq", credential_env_var="GROQ_API_KEY"),),
        models=(ModelConfig(provider="groq", model="<groq-model-id>"),),
        default_timeout_s=30.0,
    ),
    env=os.environ,
)
registry = build_registry(settings, env=os.environ)

request = LLMRequest(messages=(Message(role=MessageRole.USER, content="Hello"),))
effective_timeout_s = resolve_timeout_s(settings, provider_name="groq", request=request)

executor = FallbackExecutor(registry, RetryEngine())
result = asyncio.run(executor.run(request, settings.models, timeout_s=effective_timeout_s))
```

Here the resolved value (level 4, `30.0`) reaches the provider only because it
is passed explicitly; the same value goes equally to
`Orchestrator.run(request, model, timeout_s=effective_timeout_s)` or a direct
provider `generate`. Call `resolve_timeout_s` with a per-call `timeout_s=` to
win level 1, or omit it to keep walking the chain.

### Invalid values are rejected at every seam

Each `timeout_s` seam (`RetryEngine.generate`, `FallbackExecutor.run`,
`Orchestrator.run`, and both provider `generate` methods) validates its
argument before any attempt, SDK call, or backoff: `None` means "no explicit
timeout"; any other value must be a positive, finite number of seconds.
Zero, negative, NaN, and infinite values raise a value-free
`ConfigurationError` — `timeout_s must be a positive, finite number of
seconds.` — instead of ever reaching a provider. Contract-held timeouts
(`LLMRequest.timeout_s`, `ProviderConfig.default_timeout_s`, settings
`default_timeout_s`) are validated when those objects are constructed.

## Secrets and boundaries

- Settings hold **no secret values** — only variable *names* via
  `credential_env_var`. Values are resolved inside `build_registry`,
  passed straight to the provider constructor, and never stored.
- Missing or blank credentials raise `ConfigurationError` naming the
  provider and the expected variable, never the value, e.g.
  `Missing credential for provider 'gemini'; expected environment variable
  GEMINI_API_KEY.`
- `import llm_orchestrator.config` reads no environment, requires no keys,
  and loads no provider SDKs (adapters are imported lazily inside
  `build_registry`, one branch per configured provider — configuring only
  Gemini never imports the Groq adapter, and vice versa). A configured
  provider whose optional extra is not installed raises `ConfigurationError`
  naming the exact `pip install "llm-orchestrator[...]"` command.
- `config/` implements no retry, fallback, routing, or LangGraph logic and
  keeps no singletons. Supported providers are exactly `gemini` and `groq`;
  anything else fails loudly rather than pretending third-party providers
  are supported.
