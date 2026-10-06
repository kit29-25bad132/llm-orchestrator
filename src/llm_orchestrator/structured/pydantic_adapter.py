"""Pydantic-backed structured-output validation.

Converts the normalized text of an :class:`LLMResponse` into a caller-owned
Pydantic model: extract a JSON document (plain or `````json``-fenced),
parse it with the standard library, and validate it with the supplied
schema. Every failure becomes a normalized
:class:`StructuredOutputValidationError` — never a raw ``json`` or Pydantic
exception, and never provider-specific behavior.

Deliberately narrow: no JSON repair, no schema inference, no re-prompting,
no additional model calls. The caller owns its schema; this module only
checks untrusted text against it.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from typing import Any, TypeVar

from pydantic import BaseModel, ValidationError

from ..contracts import LLMResponse
from ..errors import StructuredOutputValidationError, sanitize_message

T = TypeVar("T", bound=BaseModel)

#: Maximum characters of (sanitized) model output embedded in error metadata.
#: Bounds exception size; the full text is untrusted and never needed verbatim.
_MAX_PREVIEW_CHARS = 500

#: Maximum Pydantic field errors summarized in one message. Bounds message
#: size while still pointing at the first problems.
_MAX_REPORTED_ERRORS = 5

#: A complete fenced block: opening fence with an optional language tag,
#: a body, and a closing fence. Anything else starting with a fence is
#: malformed rather than JSON with decoration.
_FENCED_JSON_RE = re.compile(
    r"^```(?P<tag>[^`\r\n]*)\r?\n(?P<body>.*)\r?\n```[ \t]*$",
    re.DOTALL,
)


class _ExtractionError(ValueError):
    """Internal signal: response text holds no extractable JSON document."""


def _extract_json_text(content: str) -> str:
    """Return the JSON document embedded in normalized response text.

    Accepts a bare document (surrounding whitespace ignored) or one complete
    `````json`` (or untagged) fenced block. Anything else — prose, partial
    fences, unsupported tags — is rejected so callers never receive a
    silently reinterpreted payload.
    """
    text = content.strip()
    if not text:
        raise _ExtractionError("response has no content to validate")
    if not text.startswith("```"):
        return text
    match = _FENCED_JSON_RE.match(text)
    if match is None:
        raise _ExtractionError("response fence is malformed")
    if match.group("tag").strip().casefold() not in ("", "json"):
        raise _ExtractionError("only plain or ```json fenced output is supported")
    return match.group("body").strip()


def _format_item(item: Mapping[str, Any]) -> str:
    """Format one Pydantic failure as ``dotted.path: message``."""
    location = ".".join(str(part) for part in item.get("loc", ())) or "<root>"
    return f"{location}: {item.get('msg', 'invalid value')}"


def _summarize_validation_errors(error: ValidationError) -> str:
    """Summarize Pydantic failures as ``path: message`` pairs, newest-safe.

    Only locations and messages are kept — never input values, which come
    from untrusted model output and may carry secrets.
    """
    shown = [_format_item(item) for item in error.errors(include_url=False)]
    parts = shown[:_MAX_REPORTED_ERRORS]
    remaining = error.error_count() - len(parts)
    if remaining > 0:
        parts.append(f"... and {remaining} more")
    return "; ".join(parts)


def _preview(content: str) -> str:
    """Sanitized, truncated model output for error metadata (never the secret)."""
    return sanitize_message(content)[:_MAX_PREVIEW_CHARS]


def _failure(
    response: LLMResponse,
    schema_name: str,
    *,
    stage: str,
    detail: str,
    content: str,
    cause: BaseException,
) -> StructuredOutputValidationError:
    """Build the normalized error for one failed validation stage."""
    return StructuredOutputValidationError(
        f"structured output failed for {schema_name} at {stage}: {detail}",
        provider=response.provider,
        model=response.model,
        request_id=response.request_id,
        cause=cause,
        metadata={
            "stage": stage,
            "schema": schema_name,
            "content_preview": _preview(content),
        },
    )


def validate_structured_output(response: LLMResponse, schema: type[T]) -> T:
    """Validate an ``LLMResponse`` against a caller-owned Pydantic model.

    Parses the normalized ``content`` as JSON (plain or `````json``-fenced)
    and returns ``schema.model_validate(...)`` typed as ``T``. Any failure —
    missing content, malformed fence, invalid JSON, schema mismatch — raises
    :class:`StructuredOutputValidationError` carrying provider/model/request
    context, a safe message, and the original exception as ``cause``.

    Pure local CPU work: no network, no model calls, no retries, no repair.
    """
    if not (isinstance(schema, type) and issubclass(schema, BaseModel)):
        raise TypeError(
            f"schema must be a Pydantic BaseModel subclass, got {type(schema).__name__}"
        )
    schema_name = schema.__name__
    content = response.content or ""
    try:
        document = _extract_json_text(content)
    except _ExtractionError as exc:
        raise _failure(
            response,
            schema_name,
            stage="extract",
            detail=str(exc),
            content=content,
            cause=exc,
        ) from exc
    try:
        parsed: Any = json.loads(document)
    except json.JSONDecodeError as exc:
        raise _failure(
            response,
            schema_name,
            stage="parse",
            detail=f"invalid JSON: {exc.msg} (line {exc.lineno}, column {exc.colno})",
            content=content,
            cause=exc,
        ) from exc
    try:
        return schema.model_validate(parsed)
    except ValidationError as exc:
        raise _failure(
            response,
            schema_name,
            stage="validate",
            detail=_summarize_validation_errors(exc),
            content=content,
            cause=exc,
        ) from exc
