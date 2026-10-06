"""Unit tests for orchestration state: construction and preservation."""

from __future__ import annotations

import typing

from llm_orchestrator.contracts import LLMRequest, Message, MessageRole, ModelConfig
from llm_orchestrator.orchestration import OrchestrationState, initial_state


def _request() -> LLMRequest:
    return LLMRequest(messages=[Message(role=MessageRole.USER, content="hi")])


def _model() -> ModelConfig:
    return ModelConfig(provider="fake", model="m")


def test_initial_state_carries_request_model_and_timeout() -> None:
    request = _request()
    model = _model()
    state = initial_state(request, model, timeout_s=2.5)
    assert state["request"] is request
    assert state["model"] is model
    assert state["timeout_s"] == 2.5


def test_initial_state_timeout_defaults_to_none() -> None:
    state = initial_state(_request(), _model())
    assert state["timeout_s"] is None


def test_initial_state_has_no_outcome_fields() -> None:
    state = initial_state(_request(), _model())
    assert "response" not in state
    assert "error" not in state
    assert "attempts" not in state
    assert "result" not in state


def test_state_is_a_typed_dict() -> None:
    hints = typing.get_type_hints(OrchestrationState)
    assert hints["request"].__name__ == "LLMRequest"
    assert hints["model"].__name__ == "ModelConfig"
    assert hints["result"].__name__ == "OrchestrationResult"
