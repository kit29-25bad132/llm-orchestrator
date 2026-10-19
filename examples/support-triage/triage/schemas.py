"""Structured classification schema for support tickets.

This schema is what the consumer hands to the library (via the request's
post-hoc validation step) and what ``validate_structured_output`` enforces
against the model's reply.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

TicketCategory = Literal[
    "billing",
    "account_access",
    "bug_report",
    "feature_request",
    "how_to",
    "other",
]

TicketUrgency = Literal["low", "normal", "high", "critical"]


class TicketClassification(BaseModel):
    """The structured verdict the orchestrator must produce for one ticket."""

    category: TicketCategory
    urgency: TicketUrgency
    summary: str = Field(min_length=1, max_length=400)
    suggested_reply: str = Field(min_length=1)
    confidence: float = Field(ge=0.0, le=1.0)
