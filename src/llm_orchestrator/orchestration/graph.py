"""LangGraph orchestration for single-candidate runs.

Topology (deliberately linear — fallback branching arrives in a later phase)::

    START → execute_model → END

The graph is compiled once and invoked repeatedly: compilation binds the
workflow, while every ``run`` supplies fresh per-invocation state, so no
request data can leak between runs. Execution is async throughout
(``ainvoke``); cancellation propagates instead of becoming a result.
"""

from __future__ import annotations

from asyncio import CancelledError
from functools import partial
from typing import Any

try:
    from langgraph.errors import NodeCancelledError
    from langgraph.graph import END, START, StateGraph
    from langgraph.graph.state import CompiledStateGraph
except ImportError as exc:
    # LangGraph is an optional extra; never fail with an opaque ImportError.
    from ..errors import ConfigurationError

    raise ConfigurationError(
        "llm_orchestrator.orchestration requires the optional 'langgraph' package. "
        'Install it with: pip install "llm-orchestrator[langgraph]"'
    ) from exc

from ..contracts import LLMRequest, ModelConfig, OrchestrationResult
from ..errors import require_timeout_s
from ..providers import ProviderRegistry
from ..reliability import RetryEngine
from .nodes import execute_model
from .state import OrchestrationState, initial_state

#: The single workflow node in Phase 8. Later phases add siblings here
#: (fallback, validation) and conditional edges between them.
EXECUTE_MODEL_NODE = "execute_model"


def build_orchestration_graph(
    registry: ProviderRegistry,
    retry_engine: RetryEngine,
) -> CompiledStateGraph[OrchestrationState, Any, Any, Any]:
    """Compile the orchestration workflow bound to ``registry``/``retry_engine``.

    The compiled graph holds workflow structure plus references to the
    injected infrastructure — no per-request data. Invoke it repeatedly with
    fresh input state.
    """
    builder = StateGraph(OrchestrationState)
    builder.add_node(
        EXECUTE_MODEL_NODE,
        partial(execute_model, registry=registry, retry_engine=retry_engine),
    )
    builder.add_edge(START, EXECUTE_MODEL_NODE)
    builder.add_edge(EXECUTE_MODEL_NODE, END)
    return builder.compile()


class Orchestrator:
    """Runs single-candidate LLM requests through the LangGraph workflow.

    The orchestrator owns no providers and implements no retry logic: it
    resolves candidates through the injected ``ProviderRegistry`` and retries
    through the injected ``RetryEngine``. Timeouts pass straight through to
    the retry engine and provider.
    """

    def __init__(
        self,
        registry: ProviderRegistry,
        retry_engine: RetryEngine | None = None,
    ) -> None:
        self._registry = registry
        self._retry_engine = retry_engine if retry_engine is not None else RetryEngine()
        self._graph = build_orchestration_graph(registry, self._retry_engine)

    @property
    def registry(self) -> ProviderRegistry:
        """The provider registry this orchestrator resolves candidates from."""
        return self._registry

    @property
    def retry_engine(self) -> RetryEngine:
        """The retry engine this orchestrator delegates retries to."""
        return self._retry_engine

    async def run(
        self,
        request: LLMRequest,
        model: ModelConfig,
        *,
        timeout_s: float | None = None,
    ) -> OrchestrationResult:
        """Execute one candidate and return its normalized result.

        Success yields ``OrchestrationResult`` with the ``LLMResponse``;
        expected provider failures yield one with the normalized ``ErrorInfo``.
        Unexpected exceptions propagate untouched, as does cancellation: when
        the node itself is cancelled, LangGraph surfaces ``NodeCancelledError``
        wrapping the ``CancelledError``; that wrapper is unwrapped here so
        callers always observe standard cancellation semantics.
        """
        require_timeout_s(timeout_s)
        try:
            final = await self._graph.ainvoke(initial_state(request, model, timeout_s=timeout_s))
        except NodeCancelledError as exc:
            cause = exc.__cause__
            if isinstance(cause, CancelledError):
                raise cause from exc
            raise
        result = final.get("result")
        if not isinstance(result, OrchestrationResult):
            raise RuntimeError("orchestration graph did not produce a result")
        return result
