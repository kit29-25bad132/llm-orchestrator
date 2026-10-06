# Errors (Phase 7: classification)

One normalized error language spans every provider and every reliability
layer. Vendor SDK specifics are classified at the provider boundary; everything
above the boundary speaks only `LLMError`.

```
Gemini SDK error ──→ Gemini adapter ──→ LLMError ──→ RetryEngine ──→ caller
Groq SDK error   ──→ Groq adapter   ──→ LLMError ──→ (future fallback/routing/graph)
```

Classification describes the failure (category, retryability, fallback
eligibility, context). It never decides policy: the retry engine decides
whether to retry, and the future fallback engine will decide whether to change
candidate.

---

## The normalized model

`LLMError` (`llm_orchestrator.errors`) carries:

| Field | Meaning |
| --- | --- |
| `category` | Provider-independent `ErrorCategory` (the single source of truth) |
| `retryable` | May the *same* candidate be attempted again (defaults from category, overridable per instance) |
| `fallback_eligible` | May execution move to a *different* candidate (independent of `retryable`) |
| `provider` | `"gemini"` / `"groq"` — identity only, never a subclass per vendor |
| `model` | Model id as supplied by the caller, where known |
| `request_id` / `attempt` | Correlation; `attempt` is stamped by the retry engine |
| `status_code` | Provider HTTP status where the SDK exposed one |
| `retry_after_s` | Server-requested wait (`RateLimitError` only); normalized to a finite, non-negative float or `None` |
| `safe_message` | Sanitized message — `str(error)` and logging must use this, never raw payloads |
| `cause` | Original SDK exception (plus `raise ... from` chaining at the boundary) |
| `metadata` | Small non-secret extras |

`retryable` and `fallback_eligible` are **independent**: a rate limit is both,
an invalid request is neither, an unknown failure is not retried but may fall
back. `RetryPolicy.retry_on` can override the taxonomy per call.

---

## Category matrix

| Category | Class | Retryable | Fallback-eligible | Notes |
| --- | --- | --- | --- | --- |
| `AUTHENTICATION` | `AuthenticationError` | no | yes | Bad/missing credential; another provider's credential may work |
| `AUTHORIZATION` | `AuthorizationError` | no | yes | Valid credential, not permitted |
| `INVALID_REQUEST` | `InvalidRequestError` | no | no | Deterministic for this request; retrying cannot help |
| `RATE_LIMIT` | `RateLimitError` | **yes** | yes | Honors normalized `retry_after_s` |
| `TIMEOUT` | `LLMTimeoutError` | **yes** | yes | Operation exceeded its time budget (incl. HTTP 408) |
| `NETWORK` | `NetworkError` | **yes** | yes | Transport/connectivity failure |
| `MODEL_UNAVAILABLE` | `ModelUnavailableError` | no | yes | Missing/deprecated/overloaded model — try *another* model, not again |
| `PROVIDER_UNAVAILABLE` | `ProviderUnavailableError` | **yes** | yes | Provider-side 5xx/outage |
| `MALFORMED_RESPONSE` | `MalformedResponseError` | **yes** | yes | Unparseable success payload; may be transient |
| `STRUCTURED_OUTPUT_VALIDATION` | `StructuredOutputValidationError` | **yes** | yes | Reserved for the Phase 10 validation path |
| `CONFIGURATION` | `ConfigurationError` | no | no | Programmer/operator error — fail fast |
| `UNKNOWN` | `UnknownProviderError` | no | yes | Conservative: never blindly retried |

`ProviderNotRegisteredError` subclasses `ConfigurationError` (lookup miss =
never retried, never fallen back).

---

## Provider mappings

Both adapters classify by SDK exception type first, HTTP status second, and
never by message text. Equivalent failures converge on identical normalized
semantics (proven by `test_cross_provider_consistency.py`).

### Gemini (`providers/gemini.py`)

| SDK failure | Normalized | Category | Retryable |
| --- | --- | --- | --- |
| HTTP 400 / other 4xx | `InvalidRequestError` | `INVALID_REQUEST` | no |
| HTTP 401 | `AuthenticationError` | `AUTHENTICATION` | no |
| HTTP 403 | `AuthorizationError` | `AUTHORIZATION` | no |
| HTTP 404 | `ModelUnavailableError` | `MODEL_UNAVAILABLE` | no |
| HTTP 408 | `LLMTimeoutError` | `TIMEOUT` | **yes** |
| HTTP 429 (+ `Retry-After`) | `RateLimitError` | `RATE_LIMIT` | **yes** |
| HTTP 5xx | `ProviderUnavailableError` | `PROVIDER_UNAVAILABLE` | **yes** |
| `httpx` timeout | `LLMTimeoutError` | `TIMEOUT` | **yes** |
| `httpx` transport error | `NetworkError` | `NETWORK` | **yes** |
| anything else | `UnknownProviderError` | `UNKNOWN` | no |

### Groq (`providers/groq.py`)

| SDK failure | Normalized | Category | Retryable |
| --- | --- | --- | --- |
| `BadRequestError` (400), `UnprocessableEntityError` (422), `ConflictError` (409), other 4xx | `InvalidRequestError` | `INVALID_REQUEST` | no |
| `AuthenticationError` (401) | `AuthenticationError` | `AUTHENTICATION` | no |
| `PermissionDeniedError` (403) | `AuthorizationError` | `AUTHORIZATION` | no |
| `NotFoundError` (404) | `ModelUnavailableError` | `MODEL_UNAVAILABLE` | no |
| HTTP 408 | `LLMTimeoutError` | `TIMEOUT` | **yes** |
| `RateLimitError` (429, + `Retry-After`) | `RateLimitError` | `RATE_LIMIT` | **yes** |
| `InternalServerError` / other 5xx | `ProviderUnavailableError` | `PROVIDER_UNAVAILABLE` | **yes** |
| `APITimeoutError` / `httpx` timeout | `LLMTimeoutError` | `TIMEOUT` | **yes** |
| `APIConnectionError` / `httpx` transport error | `NetworkError` | `NETWORK` | **yes** |
| anything else | `UnknownProviderError` | `UNKNOWN` | no |

---

## Safety rules

- **Secrets.** `safe_message` redacts `Bearer` tokens, `key=...`-style
  assignments, and bare provider-shaped keys; `str(error)` returns the
  sanitized form. `ErrorInfo.from_error` snapshots use `safe_message`, so
  serialized results stay clean. Raw SDK objects never enter serializable
  models — `cause` stays on the live exception only.
- **`retry_after_s`.** Normalized at construction: only finite,
  non-negative numbers survive (as `float`); everything else becomes `None`.
  Consumers additionally guard before sleeping.
- **Unknown errors.** `UNKNOWN` is never retryable by default; unmapped
  failures keep the original exception as `cause` for diagnostics.
- **Decisions are category-driven.** Retry/fallback/routing logic keys on
  `error.category` / `isinstance`, never on message strings.

---

## Intentionally deferred

- Fallback execution (Phase 8) — eligibility flags exist, no engine yet.
- Routing (Phase 11), LangGraph orchestration, structured-output validation
  (Phase 10), observability event emission, configuration loading.
