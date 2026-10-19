# Structured output (Phase 9)

`validate_structured_output` (`llm_orchestrator.structured`) converts the
normalized text of an `LLMResponse` into a caller-owned Pydantic model. It is
a pure post-orchestration function — not a graph node, not a provider feature,
and not a repair loop.

```python
from pydantic import BaseModel

from llm_orchestrator.structured import validate_structured_output


class ShipmentLabel(BaseModel):
    recipient: str
    zip_code: str
    fragile: bool = False


label = validate_structured_output(response, ShipmentLabel)
# type(label) is ShipmentLabel — never a raw dict
```

The consuming application owns its schema. The library ships no domain
models, accepts any `BaseModel` subclass, and returns the validated instance
typed as that subclass.

---

## What structured output means here

Provider text is untrusted input. Validation answers one question: *does this
text parse as JSON and satisfy the caller's schema?* Anything else — prose,
truncated payloads, wrong shapes — is an explicit, normalized failure, never
a silent reinterpretation.

## Supported input format

The normalized `LLMResponse.content` must hold either:

### A. Pure JSON

```json
{"recipient": "Ada", "zip_code": "12345"}
```

Surrounding whitespace is ignored.

### B. One complete fenced block

````text
```json
{"recipient": "Ada", "zip_code": "12345"}
```
````

The fence must be complete (opening + closing), with an empty or `json`
language tag. Anything else is rejected.

## What is intentionally NOT supported

- Prose containing a JSON fragment (fences are only recognized at the
  document level, never hunted inside text).
- Non-`json` fence tags, partial fences, trailing commentary.
- JSON repair, schema inference, default-filling beyond the schema's own
  defaults, or any re-prompt / additional model call.
- Provider-native structured modes: adapters still reject `response_schema`
  (`InvalidRequestError`); validation here is strictly post-hoc text
  validation. Native modes and the broader schema-adapter port
  (`architecture.md` §9) remain future work.

## Validation behavior

1. **Extract** — missing/blank content or a malformed fence fails here.
2. **Parse** — standard-library `json.loads`; failures report the JSON
   error position, never the payload.
3. **Validate** — `schema.model_validate`; failures summarize each field
   problem as `dotted.path: message` (first few, then a count), using only
   locations and messages — never input values.

Extra fields follow the caller's own model configuration (Pydantic's
default ignores them; `extra="forbid"` rejects them). A non-`BaseModel`
schema is a programmer bug and raises `TypeError` immediately rather than
masquerading as an LLM failure.

## Failure semantics

`validate_structured_output` (`llm_orchestrator.structured`) is a pure
post-orchestration function — it runs after the provider call (and any retry
or fallback planning) has already completed. It parses the normalized
`LLMResponse.content` as JSON (plain or `` ```json ``-fenced) and validates
it against a caller‑owned Pydantic model.

When validation fails, the function raises `StructuredOutputValidationError`
(`STRUCTURED_OUTPUT_VALIDATION` category). This error is **not** automatically
caught by the retry or fallback machinery, because it is raised after the
executor has already returned. The consumer must handle the exception
explicitly; error‑category flags such as ``retryable`` and
``fallback_eligible`` describe the error's taxonomy defaults but do not
trigger another provider attempt or a retry of the generation step.

Neither the documented post‑execution validation flow nor the library's
retry/fallback logic provides automatic validation‑error repair, retry, or
fallback. If a caller wishes to retry the generation or switch provider on a
validation failure, that must be implemented in application code.

The error preserves `provider`, `model`, and `request_id` from the response,
chains the original `json`/`Pydantic` exception as `cause`, and records
`metadata` with the `stage` (`extract`/`parse`/`validate`), the schema name,
and a sanitized ≤500‑character content preview. Messages and metadata never
carry secrets or full payloads.

## Relationship to ReviewIQ

ReviewIQ (or any consumer) defines its own Pydantic models and calls
`validate_structured_output` on the responses it receives. No ReviewIQ
concept appears anywhere in this package — see the invariant tests.
