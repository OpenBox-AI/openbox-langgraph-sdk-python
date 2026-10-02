"""HTTP-hook approval resumes inside the tool without replaying earlier work.

Uses real base instrumentation and a local HTTP server; Core is simulated.
"""

from __future__ import annotations

from typing import Annotated, Any, TypedDict

import pytest
import requests
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.tools import tool
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

pytest.importorskip("openbox_core")

from openbox_core.conformance.fake_core import FakeCore
from openbox_core.conformance.instrumentation import (
    LocalCountingServer,
    installed_conformance_runtime,
)
from openbox_core.context import ContextStore
from openbox_core.contracts.context import ActivityContext

from openbox_langgraph.client import GovernanceClient
from openbox_langgraph.core_adapter import LangGraphFrameworkAdapter
from openbox_langgraph.core_runtime import get_trace_registry
from openbox_langgraph.langgraph_handler import (
    OpenBoxLangGraphHandler,
    OpenBoxLangGraphHandlerOptions,
)
from openbox_langgraph.tool_activity_binding import bind_tools_activity_scope
from openbox_langgraph.types import GovernanceVerdictResponse, Verdict


class _AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


class _AllowEverythingClient(GovernanceClient):
    """Always-ALLOW for the LangGraph SDK's OWN lifecycle events — isolates
    this test to the hook-level REQUIRE_APPROVAL wait, not the
    unrelated WorkflowStarted/LLMStarted pre-screen governance."""

    def __init__(self) -> None:
        super().__init__(api_url="https://core.openbox.ai", api_key="obx_test_abc")

    async def evaluate_event(self, event: Any) -> GovernanceVerdictResponse | None:
        return GovernanceVerdictResponse(verdict=Verdict.ALLOW)


@pytest.fixture
def server():
    srv = LocalCountingServer()
    yield srv
    srv.stop()


def _build_tool_call_graph(url: str, effects: list[str]) -> Any:
    """A tool side effect before the HTTP request must happen only once."""

    @tool
    def http_tool(query: str) -> str:
        """Fetch a result for the given query."""
        effects.append("before-http")
        response = requests.get(url, timeout=5)
        effects.append("after-http")
        return f"status={response.status_code}"

    async def call_model(state: _AgentState) -> dict[str, Any]:
        # First call in THIS invocation (just the starting HumanMessage) ->
        # emit the tool call. After the tool round-trip adds a ToolMessage,
        # the next call finishes with a plain text reply.
        if len(state["messages"]) <= 1:
            return {
                "messages": [
                    AIMessage(
                        content="",
                        tool_calls=[{"name": "http_tool", "args": {"query": "hi"}, "id": "call-1"}],
                    )
                ]
            }
        return {"messages": [AIMessage(content="done")]}

    def should_continue(state: _AgentState) -> str:
        last = state["messages"][-1]
        if getattr(last, "tool_calls", None):
            return "tools"
        return END

    graph = StateGraph(_AgentState)
    graph.add_node("agent", call_model)
    graph.add_node("tools", ToolNode([http_tool]))
    graph.add_edge(START, "agent")
    graph.add_conditional_edges("agent", should_continue, {"tools": "tools", END: END})
    graph.add_edge("tools", "agent")
    return graph.compile()


def _build_handler(graph: Any, runtime: Any) -> OpenBoxLangGraphHandler:
    """Injected client keeps `__init__` from building its own core runtime —
    `_core_runtime` is then pointed at the EXACT runtime
    `installed_conformance_runtime` armed.

    Because the injected-client path skips `__init__`'s tool binding, we bind
    the graph's ToolNode here explicitly: the sync `http_tool` runs via
    `run_in_executor`, which does NOT carry an OTel parent context, so the base
    exact-trace tier alone misses (a fresh root-span trace_id) — the ContextVar
    tier bound at the ToolNode seam is what resolves the tool's HTTP hook to its
    activity. `store.registry` is published so `reset_after_approval` can sweep
    the turn's abort marks (matching what `create_core_runtime` wires up).
    """
    handler = OpenBoxLangGraphHandler(
        graph=graph, options=OpenBoxLangGraphHandlerOptions(client=_AllowEverythingClient())
    )
    handler._core_runtime = runtime  # type: ignore[attr-defined]
    runtime.context_store.registry = get_trace_registry(runtime)
    bind_tools_activity_scope(
        graph,
        core_runtime=runtime,
        config=handler._config,  # type: ignore[attr-defined]
        resolve_tool_type=lambda name: None,
    )
    return handler


async def test_approved_hook_resumes_without_replaying_tool(server) -> None:
    fake_core = FakeCore({"verdict": "require_approval", "approval_id": "app-1"})
    store = ContextStore()
    adapter = LangGraphFrameworkAdapter(context_store=store)
    effects: list[str] = []
    graph = _build_tool_call_graph(server.url, effects)

    with installed_conformance_runtime(fake_core, adapter, store) as runtime:
        handler = _build_handler(graph, runtime)
        before = server.hits
        result = await handler.ainvoke(
            {"messages": [HumanMessage(content="please fetch it")]},
            config={"configurable": {"thread_id": "approval-thread"}},
        )

    assert len(fake_core.approval_requests) == 1
    assert len(fake_core.started_payloads) == 1
    assert (
        fake_core.approval_requests[0]["activity_id"]
        == (fake_core.started_payloads[0]["activity_id"])
    )
    assert effects == ["before-http", "after-http"]
    assert server.hits == before + 1
    tool_message = next(m for m in result["messages"] if m.type == "tool")
    assert tool_message.content == "status=200"
    assert result["messages"][-1].content == "done"


def test_reset_after_approval_clears_the_turns_abort_marks(server) -> None:
    """The legacy reset helper remains available to direct adapter callers."""
    fake_core = FakeCore()
    store = ContextStore()
    adapter = LangGraphFrameworkAdapter(context_store=store)

    with installed_conformance_runtime(fake_core, adapter, store) as runtime:
        store.registry = get_trace_registry(runtime)
        workflow_id = "reset-workflow"
        ctx = ActivityContext(
            workflow_id=workflow_id,
            run_id="reset-run",
            workflow_type="ResetWorkflow",
            task_queue="langgraph",
            activity_id="reset-activity",
            activity_type="http_tool",
        )
        # Registering via the published registry records the activity key the
        # workflow-scoped sweep later clears.
        store.registry.register(trace_id=12345, ctx=ctx)
        store.mark_activity_aborted(workflow_id, "reset-activity")
        assert store.is_activity_aborted(workflow_id, "reset-activity")

        adapter.reset_after_approval(workflow_id)

        assert not store.is_activity_aborted(workflow_id, "reset-activity"), (
            "reset_after_approval must clear the turn's abort marks so an "
            "approved retry runs governed"
        )
