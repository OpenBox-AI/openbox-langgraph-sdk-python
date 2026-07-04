"""The post-approval reset for the core-instrumentation path: a tool's HTTP
hook returns REQUIRE_APPROVAL, `ainvoke`'s outer loop polls, approval resolves,
`_reset_after_approval` clears the turn's abort marks, and the retry runs
GOVERNED (its HTTP call reaches the server).

The tool's HTTP hook resolves its activity via the ContextVar tier bound at the
ToolNode seam (see `tool_activity_binding`) — there is no single-active/
last-registered fallback. Two tests here: the end-to-end approved-retry-runs
(`test_approved_retry_runs_governed_not_short_circuited`), and a focused unit
check that `reset_after_approval` actually clears a workflow's abort marks
(`test_reset_after_approval_clears_the_turns_abort_marks`).

Uses REAL base instrumentation (`openbox_core.conformance`) against a REAL
local HTTP server so "GOVERNED" is observed as an actual request reaching the
server — not a re-implementation of the hook runtime's own decision logic. The
handler's `_core_runtime` is set to the EXACT runtime object
`installed_conformance_runtime` installs (not a second, separate one) so the
real, globally-published `HookRuntime` the instrumented `requests` library
calls into shares the same adapter/store the test's assertions inspect.
"""

from __future__ import annotations

from typing import Annotated, Any, TypedDict
from unittest.mock import AsyncMock, patch

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
    this test to the hook-level REQUIRE_APPROVAL retry path, not the
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


def _build_tool_call_graph(url: str) -> Any:
    """LLM emits one tool call to `http_tool` (a REAL `@tool`-decorated
    function run through a REAL `ToolNode`), which makes a REAL HTTP request
    to `url` — the operation base instrumentation actually governs. Only
    `on_tool_start`/`on_chat_model_start` boundaries trigger
    `_process_event`'s dual-write registration (confirmed empirically — a
    plain graph chain node with no tool/LLM call inside it never registers
    ANY activity context, so its HTTP calls run completely ungoverned
    regardless of instrumentation).

    Decides "call the tool" vs "finish" by MESSAGE COUNT within the CURRENT
    invocation's state, not a response-list cursor or a module-level
    counter: `ainvoke`'s retry calls `self._graph.ainvoke(input, ...)`
    DIRECTLY with the ORIGINAL input, so LangGraph state resets to just the
    starting `HumanMessage` on the retry (same caveat
    `test_hook_approval_retry_baseline.py`'s module docstring documents) — a
    response-list-cursor fake model would have its SECOND canned response
    consumed by the retry's FIRST model call instead of a second tool
    invocation, silently skipping the tool round-trip this module's
    assertions need to observe. Message-count-based decisions are correct
    on EITHER path because each is a fresh invocation with its own message
    history.
    """

    @tool
    def http_tool(query: str) -> str:
        """Fetch a result for the given query."""
        response = requests.get(url, timeout=5)
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
                        tool_calls=[
                            {"name": "http_tool", "args": {"query": "hi"}, "id": "call-1"}
                        ],
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


async def test_approved_retry_runs_governed_not_short_circuited(server) -> None:
    """The core assertion: after `_reset_after_approval`, the retry's HTTP
    call actually REACHES the server — the abort mark from the blocked first
    pass must not silently short-circuit the approved retry with a fabricated
    'already aborted' block that never even asks Core again."""
    fake_core = FakeCore({"verdict": "require_approval", "approval_id": "app-1"})
    # Plain ContextStore, zero fallback: the sync tool's HTTP hook resolves
    # via the ContextVar tier bound at the ToolNode seam (see _build_handler),
    # not a single-active/last-registered guess. `mock_poll.assert_awaited_once`
    # below proves the first pass actually hit REQUIRE_APPROVAL.
    store = ContextStore()
    adapter = LangGraphFrameworkAdapter(context_store=store)
    graph = _build_tool_call_graph(server.url)

    with installed_conformance_runtime(fake_core, adapter, store) as runtime:
        handler = _build_handler(graph, runtime)
        with patch(
            "openbox_langgraph.langgraph_handler.poll_until_decision",
            new=AsyncMock(return_value=None),
        ) as mock_poll:
            before = server.hits
            result = await handler.ainvoke(
                {"messages": [HumanMessage(content="please fetch it")]},
                config={"configurable": {"thread_id": "approval-retry-thread"}},
            )

    mock_poll.assert_awaited_once()
    # The retry's tool call reached the REAL server — proves GOVERNED
    # execution, not a silent short-circuit on the stale abort mark. This is
    # the OBSERVABLE proof the reset worked: a still-aborted retry would have
    # raised out of `ainvoke` a second time instead of returning a result.
    assert server.hits == before + 1
    tool_message = next(m for m in result["messages"] if m.type == "tool")
    assert tool_message.content == "status=200"
    assert result["messages"][-1].content == "done"  # the agent's final reply


def test_reset_after_approval_clears_the_turns_abort_marks(server) -> None:
    """The reset mechanism itself: `reset_after_approval(workflow_id)` clears
    every abort mark registered under that workflow this turn.

    With zero fallback, a retry mints its OWN fresh activity id and re-evaluates
    from scratch (no stale-mark short-circuit resolved by guessing), so the old
    end-to-end 'without reset the retry stays blocked' contrast no longer holds
    — that blocking was the single-active fallback the plan removed. This is the
    focused replacement: register an activity, mark it aborted, reset, assert it
    is cleared — proving the workflow-scoped sweep `_reset_after_approval` calls
    still does its job on a plain ContextStore + published registry."""
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
