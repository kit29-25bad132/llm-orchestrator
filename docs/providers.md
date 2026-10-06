# Providers

A provider is a concrete implementation of the `LLMProvider` protocol
(`llm_orchestrator.providers.base`). Providers translate between the
provider-neutral contracts and a vendor SDK, and they never perform retries,
fallbacks, or routing — those belong to the orchestration and reliability
layers.

```
LLMRequest + ModelConfig
        ↓
   LLMProvider.generate(...)      ← exactly one attempt
        ↓
   LLMResponse
```

The abstraction and registry (`llm_orchestrator.providers`) are deliberately
SDK-free. Concrete adapters live in their own modules and are imported
explicitly.

| Provider | Module | Status |
| --- | --- | --- |
| Gemini | `llm_orchestrator.providers.gemini` | Phase 4 (implemented) |
| Groq | `llm_orchestrator.providers.groq` | Phase 5 (planned) |

---

## Gemini

`GeminiProvider` is the first concrete adapter. It is the **only** module in the
library allowed to import the Google GenAI SDK, so no SDK type escapes the
provider boundary.

### Dependency

- Official package: **`google-genai`** (the current unified Google GenAI SDK;
  not the deprecated `google-generativeai`).
- Declared runtime dependency: `google-genai>=2.0` (validated against 2.28.0).
- It is the only provider SDK added in Phase 4; LangGraph, Groq, OpenAI, and
  others remain out until their phases.

### Import safety

`import llm_orchestrator` and `import llm_orchestrator.providers` do **not**
import `google.genai`, require credentials, or make network calls. Import the
adapter explicitly:

```python
from llm_orchestrator.providers.gemini import GeminiProvider
```

### Configuration

Configuration enters explicitly; the provider never reads the environment.

```python
provider = GeminiProvider(api_key=os.environ["GEMINI_API_KEY"])
```

- `api_key` is required unless a pre-built `client` is injected. Missing or
  blank keys raise `ConfigurationError` at construction (fail fast).
- The key is held privately and never appears in `repr`, errors, response
  metadata, or serialization.
- The future configuration layer (Phase 12) is responsible for reading
  `GEMINI_API_KEY` and passing it in; the provider itself stays
  environment-independent.
- `client=...` lets tests (and advanced callers) inject an SDK client.

### Model selection

The target model comes from the `ModelConfig` passed to `generate`, not from the
provider. There is no default model constant and no model registry here:

```python
from llm_orchestrator.contracts import ModelConfig

model = ModelConfig(provider="gemini", model="<gemini-model-id>")
```

If `model.provider` is not `"gemini"`, the adapter raises `InvalidRequestError`
instead of silently generating with the wrong vendor's model id. No model
substitution or fallback ever happens inside the provider.

### Client lifecycle

One SDK client per provider instance, created **lazily** on first use and reused.
`await provider.aclose()` releases it. Client creation happens neither at import
time nor in the constructor. The adapter does not claim thread safety for
concurrent use of a single client.

### Request translation

| Generic (`LLMRequest`) | Gemini |
| --- | --- |
| `Message(role=SYSTEM)` | `config.system_instruction` (multiple joined with `\n\n`) |
| `Message(role=USER)` | `Content(role="user", ...)` |
| `Message(role=ASSISTANT)` | `Content(role="model", ...)` |
| `Message(role=TOOL)` | **rejected** (`InvalidRequestError`) — tool calling is deferred |
| consecutive same-role messages | merged into one `Content` (preserves user/model alternation) |
| `temperature` | `config.temperature` |
| `max_tokens` | `config.max_output_tokens` |
| `stop` | `config.stop_sequences` |
| `timeout_s` (or `request.timeout_s`) | `config.http_options.timeout` (milliseconds) |
| `response_schema` | **rejected** (`InvalidRequestError`) — structured output is Phase 9 |

Only semantically supported parameters are forwarded; nothing is invented, and
unsupported semantics fail explicitly rather than being silently ignored.

> Timeout note: the effective timeout is applied per request through the SDK's
> HTTP options. When a caller injects a client with custom HTTP options, a
> per-request timeout may replace them.

### Response normalization

A Gemini response is converted into the Phase 2 `LLMResponse`:

- `content` — text parts of the first candidate concatenated; `None` when there
  is no text (never a fabricated empty string).
- `provider` / `model` — `"gemini"` and the requested model id.
- `finish_reason` — normalized (see below).
- `usage` — `prompt_token_count` → `input_tokens`, `candidates_token_count` →
  `output_tokens`, `total_token_count` → `total_tokens`; `None` when absent.
  Values are never estimated.
- `metadata` — non-secret provenance such as `response_id` and `model_version`.
- `latency_ms` — measured around the single attempt.

Edge cases handled explicitly:

- **Blocked prompt** (no candidates, `prompt_feedback.block_reason` set) →
  `finish_reason=CONTENT_FILTER`, `content=None`, `block_reason` in metadata.
- **No candidates and no block reason** → `MalformedResponseError`.
- **No text parts** → `content=None` with the real finish reason.
- **Multiple candidates** → the first is used (no sampling/selection logic).

Finish reason mapping:

| Gemini | Normalized |
| --- | --- |
| `STOP` | `STOP` |
| `MAX_TOKENS` | `LENGTH` |
| `SAFETY`, `RECITATION`, `BLOCKLIST`, `PROHIBITED_CONTENT`, `SPII`, image-safety reasons, `NO_IMAGE` | `CONTENT_FILTER` |
| `MALFORMED_FUNCTION_CALL`, `UNEXPECTED_TOOL_CALL`, `TOO_MANY_TOOL_CALLS` | `ERROR` |
| anything else / unspecified | `UNKNOWN` |

### Error normalization

Every exception raised by the SDK is mapped onto the Phase 2 error hierarchy;
no SDK exception and no SDK object crosses the boundary. Mapping uses the SDK's
structured error fields (`code`, `status`, `message`), not fragile message
parsing. Unclassifiable failures become `UnknownProviderError` with the original
exception chained (`raise ... from exc`) and preserved as `cause`.

| Source | Normalized error | Retryable by default |
| --- | --- | --- |
| HTTP 400 | `InvalidRequestError` | no |
| HTTP 401 | `AuthenticationError` | no |
| HTTP 403 | `AuthorizationError` | no |
| HTTP 404 | `ModelUnavailableError` | no (fallback-eligible) |
| HTTP 429 | `RateLimitError` (+ `retry_after_s`) | **yes** |
| other 4xx | `InvalidRequestError` | no |
| HTTP 5xx | `ProviderUnavailableError` | **yes** |
| SDK/HTTP timeout | `LLMTimeoutError` | **yes** |
| transport/connection error | `NetworkError` | **yes** |
| anything else | `UnknownProviderError` | no (fallback-eligible) |

Only classification happens here. There is no retry loop, no backoff, no sleep,
and no fallback — those are Phases 6–8.

### Capabilities

The adapter does not advertise capabilities; capability facts live on
`ModelConfig.capabilities` (single source of truth). `structured_output` stays
`False` until Phase 9 actually implements that pathway, so the provider never
claims something it cannot do.

---

## Testing a provider

Two layers, by design:

### Offline unit tests (default suite)

`tests/unit/providers/test_gemini_provider.py` exercises the real adapter —
translation, normalization, error mapping — against a fake SDK **client**. No
network, no API key. The real `google.genai` types are used for the response and
error inputs so the tests cover the actual translation/normalization code rather
than a re-implementation.

```bash
pytest -q          # entirely offline
```

### Opt-in live integration test

`tests/integration/test_gemini_live.py` performs one small real round trip. It is
marked `live` and skipped unless explicitly enabled, so it never runs by default:

```bash
RUN_GEMINI_INTEGRATION=1 \
GEMINI_API_KEY=... \
GEMINI_TEST_MODEL=<a valid gemini model id> \
pytest -m live
```

The live test never prints or hard-codes credentials, and the model id is
supplied by the environment rather than pinned in the repository.

---

## Intentionally deferred

- Retry, backoff, and timeout engines (Phases 6–7) — Phase 4 only classifies.
- Fallback and routing, including model substitution (Phases 8, 11).
- Structured output and validation (Phase 9).
- Tool/function calling and multimodal input.
- Groq and other providers (Phases 5+).
- The configuration framework and environment loading (Phase 12).
- Observability/telemetry emission (Phase 12).
