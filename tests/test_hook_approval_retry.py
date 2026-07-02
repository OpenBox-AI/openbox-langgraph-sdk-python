"""The post-approval reset: REQUIRE_APPROVAL raises, `ainvoke`'s outer
loop polls, approval resolves, `_reset_after_approval` clears the abort
mark BEFORE the retry, and the retry runs GOVERNED (not short-circuited by
the stale abort flag the blocked first pass left behind).

Contrast with `tests/test_hook_approval_retry_baseline.py`, which PINS the
legacy-only bug: nothing on that path ever called `clear_activity_abort`, so
a real hook-governed retry (unlike that baseline's simulated in-graph raise)
would stay blocked forever. This module proves the FIX for the opt-in
core-instrumentation path specifically — the flag-off legacy path's bug
remains exactly as pinned, untouched.

Uses REAL base instrumentation (`openbox_core.conformance`) against a REAL
local HTTP server so "GOVERNED" and "blocked" are observed as actual
requests reaching (or not reaching) the server — not a re-implementation of
the hook runtime's own decision logic. The handler's `_core_runtime` is set
to the EXACT runtime object `installed_conformance_runtime` installs (not a
second, separate one) so the real, globally-published `HookRuntime` the
instrumented `requests` library calls into shares the same adapter/store the
test's assertions inspect.
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

from openbox_langgraph.client import GovernanceClient
from openbox_langgraph.core_adapter import LangGraphFrameworkAdapter
from openbox_langgraph.fallback_context_store import FallbackContextStore
from openbox_langgraph.langgraph_handler import (
    OpenBoxLangGraphHandler,
    OpenBoxLangGraphHandlerOptions,
)
from openbox_langgraph.span_processor import WorkflowSpanProcessor
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
    """Injected client keeps `__init__` from installing GLOBAL legacy OTel
    hooks (network + process-wide side effects) — `_core_runtime` is then
    pointed at the EXACT runtime `installed_conformance_runtime` armed, same
    technique `test_hook_approval_retry_baseline.py` uses for its own
    standalone `WorkflowSpanProcessor`.

    `_span_processor` MUST be set too, even though this test never asserts
    on it directly: `_process_event`'s tool-span creation (and the base
    dual-write registration nested inside it) is gated on
    `self._span_processor is not None` — production always builds both
    together from the SAME global-config check, so a handler with a real
    `_core_runtime` but a `None` `_span_processor` (an invalid combination
    that never occurs outside tests) silently skips ALL trace registration,
    including the base one, and every hook resolves "no bound context" —
    found empirically debugging this exact test.
    """
    handler = OpenBoxLangGraphHandler(
        graph=graph, options=OpenBoxLangGraphHandlerOptions(client=_AllowEverythingClient())
    )
    handler._core_runtime = runtime  # type: ignore[attr-defined]
    handler._span_processor = WorkflowSpanProcessor()  # type: ignore[attr-defined]
    return handler


async def test_approved_retry_runs_governed_not_short_circuited(server) -> None:
    """The core assertion: after `_reset_after_approval`, the retry's HTTP
    call actually REACHES the server — the abort mark from the blocked first
    pass must not silently short-circuit the approved retry with a fabricated
    'already aborted' block that never even asks Core again."""
    fake_core = FakeCore({"verdict": "require_approval", "approval_id": "app-1"})
    # FallbackContextStore, NOT a plain ContextStore: LangGraph's ToolNode
    # runs a sync tool via `run_in_executor` — the base SDK's exact-trace
    # tier misses (fresh, unrelated trace_id on that thread), so the tool's
    # HTTP call only resolves context through the fallback registry's
    # single-active/last-registered tiers. Confirmed empirically: a plain
    # ContextStore here makes every hook resolve "no bound context" and the
    # whole REQUIRE_APPROVAL flow never triggers.
    store = FallbackContextStore()
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


async def test_without_reset_the_retry_would_stay_blocked(server) -> None:
    """Direct contrast proving the reset is what fixes it: an adapter whose
    `reset_after_approval` is a no-op (standing in for the pre-fix behavior,
    without touching the real fix) leaves the retry blocked before it ever
    reaches the server — same shape of failure
    `test_hook_approval_retry_baseline.py` pins for the legacy path."""

    class _NoResetAdapter(LangGraphFrameworkAdapter):
        def reset_after_approval(self, workflow_id: str | None) -> None:
            return None

    fake_core = FakeCore({"verdict": "require_approval", "approval_id": "app-2"})
    store = FallbackContextStore()  # see the other test's comment for why
    adapter = _NoResetAdapter(context_store=store)
    graph = _build_tool_call_graph(server.url)

    with installed_conformance_runtime(fake_core, adapter, store) as runtime:
        handler = _build_handler(graph, runtime)
        with patch(
            "openbox_langgraph.langgraph_handler.poll_until_decision",
            new=AsyncMock(return_value=None),
        ):
            before = server.hits
            with pytest.raises(Exception):  # noqa: B017 — any stop-shaped governance error
                await handler.ainvoke(
                    {"messages": [HumanMessage(content="please fetch it")]},
                    config={"configurable": {"thread_id": "approval-no-reset-thread"}},
                )
    # Without the reset, the retry's tool call never reaches the server —
    # blocked by the stale abort mark from the first pass.
    assert server.hits == before
