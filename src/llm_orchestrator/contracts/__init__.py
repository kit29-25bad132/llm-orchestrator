"""Provider-neutral contract layer.

This is the stable vocabulary shared by applications and the orchestration
infrastructure. It depends on nothing provider-, framework-, or
application-specific:

    from llm_orchestrator.contracts import LLMRequest, LLMResponse, Message
"""

from __future__ import annotations

from .messages import Message, MessageRole
from .models import (
    CandidateRef,
    ModelCapabilities,
    ModelConfig,
    ModelPricing,
    ProviderConfig,
)
from .policies import (
    FallbackPolicy,
    JitterStrategy,
    RetryPolicy,
    RoutingContext,
    RoutingPolicy,
)
from .requests import LLMRequest
from .responses import FinishReason, LLMResponse, Usage
from .results import AttemptRecord, ErrorInfo, OrchestrationResult

__all__ = [
    "AttemptRecord",
    "CandidateRef",
    "ErrorInfo",
    "FallbackPolicy",
    "FinishReason",
    "JitterStrategy",
    "LLMRequest",
    "LLMResponse",
    "Message",
    "MessageRole",
    "ModelCapabilities",
    "ModelConfig",
    "ModelPricing",
    "OrchestrationResult",
    "ProviderConfig",
    "RetryPolicy",
    "RoutingContext",
    "RoutingPolicy",
    "Usage",
]
