"""The ActivityContext is bound around ACTUAL tool execution.

Drives a REAL compiled LangGraph graph through the REAL
``OpenBoxLangGraphHandler`` (non-injected, ``use_core_instrumentation=True``)
and asserts that WHILE a tool body runs, the runtime's private
``ContextStore`` resolves that tool's ``ActivityContext`` via its ContextVar
tier — the exact tier ``openbox_core.hooks.events.resolve_context`` consults
FIRST. So any HTTP/file/db hook the tool triggers resolves to that tool.

A probe tool reads ``store.current_activity_context()`` from inside its own
body (that is precisely what a base hook sees) and also performs a real
``Path.read_text`` to confirm nothing errors on the instrumented path.

Binding is done by wrapping the graph's tools (``tool_activity_scope``), NOT by
a callback: LangChain isolates callback context from the tool body, so a
callback bind never reaches the hooks.
"""

from __future__ import annotations

import http.server
import json
import threading
from typing import Annotated, Any, TypedDict

import pytest

pytest.importorskip("openbox_core")

from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.tools import tool
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

from openbox_langgraph.langgraph_handler import create_openbox_graph_handler


class _FakeGovernanceApi:
    """Loopback server answering evaluate/auth-validate with ALLOW."""

    def __init__(self) -> None:
        class Handler(http.server.BaseHTTPRequestHandler):
            def _json(self, body: dict[str, Any]) -> None:
                payload = json.dumps(body).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)

            def do_GET(self) -> None:
                self._json({"ok": True})

            def do_POST(self) -> None:
                length = int(self.headers.get("Content-Length", 0))
                if length:
                    self.rfile.read(length)
                self._json({"verdict": "allow"})

            def log_message(self, *args: Any) -> None:
                pass

        self._server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()
        self.url = f"http://127.0.0.1:{self._server.server_port}"

    def stop(self) -> None:
        self._server.shutdown()
        self._server.server_close()


@pytest.fixture
def governance_api():
    srv = _FakeGovernanceApi()
    yield srv
    srv.stop()


class _AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


def _build_probe_graph(store_holder: dict[str, Any], captured: list[Any], probe_path):
    """react-style graph whose single tool records the bound ActivityContext."""

    @tool
    def probe_tool(text: str) -> str:
        """Record the ActivityContext resolved from the runtime store mid-execution."""
        store = store_holder["store"]
        captured.append(store.current_activity_context())
        # A real instrumented file op inside the tool body must not error and
        # runs under the same bound context.
        probe_path.read_text()
        return "ok"

    model = FakeMessagesListChatModel(
        responses=[
            AIMessage(
                content="",
                tool_calls=[{"name": "probe_tool", "args": {"text": "hi"}, "id": "call-1"}],
            ),
            AIMessage(content="done"),
        ]
    )

    async def call_model(state: _AgentState) -> dict[str, Any]:
        return {"messages": [await model.ainvoke(state["messages"])]}

    def should_continue(state: _AgentState) -> str:
        return "tools" if getattr(state["messages"][-1], "tool_calls", None) else END

    graph = StateGraph(_AgentState)
    graph.add_node("agent", call_model)
    graph.add_node("tools", ToolNode([probe_tool]))
    graph.add_edge(START, "agent")
    graph.add_conditional_edges("agent", should_continue, {"tools": "tools", END: END})
    graph.add_edge("tools", "agent")
    return graph.compile()


@pytest.mark.asyncio
async def test_tool_body_resolves_its_own_activity_context(governance_api, tmp_path) -> None:
    store_holder: dict[str, Any] = {}
    captured: list[Any] = []
    probe_path = tmp_path / "probe.txt"
    probe_path.write_text("payload")

    graph = _build_probe_graph(store_holder, captured, probe_path)
    handler = create_openbox_graph_handler(
        graph=graph,
        api_url=governance_api.url,
        api_key="obx_test_activity_ctx",
        validate=False,
        use_core_instrumentation=True,
    )
    store_holder["store"] = handler._core_runtime.context_store  # type: ignore[union-attr]
    try:
        await handler.ainvoke(
            {"messages": [HumanMessage(content="go")]},
            config={"configurable": {"thread_id": "t-activity-ctx"}},
        )

        # The tool ran, and mid-body the store resolved THIS tool's context via
        # the ContextVar tier — exactly what a base HTTP/file hook resolves.
        assert captured, "probe_tool never executed"
        ctx = captured[0]
        assert ctx is not None, "no ActivityContext bound during tool execution"
        assert ctx.activity_type == "probe_tool"
        assert ctx.metadata.get("tool_name") == "probe_tool"
        assert ctx.workflow_id.startswith("t-activity-ctx")

        # After the turn, the binding is reset — no context leaks past execution.
        assert handler._core_runtime.context_store.current_activity_context() is None  # type: ignore[union-attr]
    finally:
        handler._core_runtime.close()  # type: ignore[union-attr]


@pytest.mark.asyncio
async def test_tools_are_wrapped_idempotently(governance_api, tmp_path) -> None:
    """Building a second handler over the same graph must not double-wrap."""
    store_holder: dict[str, Any] = {}
    captured: list[Any] = []
    probe_path = tmp_path / "probe.txt"
    probe_path.write_text("x")
    graph = _build_probe_graph(store_holder, captured, probe_path)

    tool_node = graph.nodes["tools"].bound  # type: ignore[attr-defined]
    probe = tool_node.tools_by_name["probe_tool"]

    h1 = create_openbox_graph_handler(
        graph=graph, api_url=governance_api.url, api_key="obx_test_idem1", validate=False,
        use_core_instrumentation=True,
    )
    wrapped_once = probe.func
    h2 = create_openbox_graph_handler(
        graph=graph, api_url=governance_api.url, api_key="obx_test_idem2", validate=False,
        use_core_instrumentation=True,
    )
    try:
        assert getattr(probe, "_openbox_activity_scoped", False) is True
        # Second wrap is a no-op — the func object is unchanged.
        assert probe.func is wrapped_once
    finally:
        h1._core_runtime.close()  # type: ignore[union-attr]
        h2._core_runtime.close()  # type: ignore[union-attr]


def test_no_turn_bound_outside_a_governed_turn() -> None:
    """Calling a wrapped tool with no turn metadata in config must not raise and
    must not fabricate a context (direct tool use outside a governed turn)."""
    from openbox_core.context import ContextStore

    from openbox_langgraph.config import GovernanceConfig
    from openbox_langgraph.tool_activity_scope import wrap_graph_tools

    captured: list[Any] = []
    store = ContextStore()

    @tool
    def bare_tool(text: str) -> str:
        """Probe that records the ambient context."""
        captured.append(store.current_activity_context())
        return "ok"

    class _RT:
        context_store = store

    graph = StateGraph(_AgentState)
    graph.add_node("tools", ToolNode([bare_tool]))
    graph.add_edge(START, "tools")
    graph.add_edge("tools", END)
    compiled = graph.compile()
    wrap_graph_tools(
        compiled, core_runtime=_RT(), config=GovernanceConfig(), resolve_tool_type=lambda n: None
    )

    node_tool = compiled.nodes["tools"].bound.tools_by_name["bare_tool"]
    assert node_tool.func("hi") == "ok"  # no turn metadata → passthrough, no raise
    assert captured == [None]


@pytest.mark.asyncio
async def test_concurrent_turns_do_not_cross_contaminate() -> None:
    """Two turns driven CONCURRENTLY each resolve their OWN workflow_id inside
    the tool body. The per-turn ids ride the tool's own RunnableConfig (set per
    invocation by BaseTool.arun), NOT a shared module ContextVar — so there is
    no last-writer-wins cross-contamination. Uses two graphs sharing ONE store
    (the hard case: the store's ContextVar bind must be per-execution-context).
    """
    import asyncio

    from openbox_core.context import ContextStore

    from openbox_langgraph.config import GovernanceConfig
    from openbox_langgraph.tool_activity_scope import turn_metadata, wrap_graph_tools

    store = ContextStore()
    seen: dict[str, str | None] = {}

    class _RT:
        context_store = store

    def build(tag: str) -> Any:
        @tool
        def probe_tool(text: str) -> str:
            """Record the workflow_id resolved from the shared store mid-body."""
            ctx = store.current_activity_context()
            seen[tag] = ctx.workflow_id if ctx is not None else None
            return "ok"

        model = FakeMessagesListChatModel(
            responses=[
                AIMessage(
                    content="",
                    tool_calls=[{"name": "probe_tool", "args": {"text": tag}, "id": "c1"}],
                ),
                AIMessage(content="done"),
            ]
        )

        async def call_model(state: _AgentState) -> dict[str, Any]:
            return {"messages": [await model.ainvoke(state["messages"])]}

        def cont(state: _AgentState) -> str:
            return "tools" if getattr(state["messages"][-1], "tool_calls", None) else END

        g = StateGraph(_AgentState)
        g.add_node("agent", call_model)
        g.add_node("tools", ToolNode([probe_tool]))
        g.add_edge(START, "agent")
        g.add_conditional_edges("agent", cont, {"tools": "tools", END: END})
        g.add_edge("tools", "agent")
        compiled = g.compile()
        wrap_graph_tools(
            compiled, core_runtime=_RT(), config=GovernanceConfig(),
            resolve_tool_type=lambda n: None,
        )
        return compiled

    async def drive(tag: str, workflow_id: str) -> None:
        graph = build(tag)
        cfg = {"metadata": turn_metadata(workflow_id, f"run-{tag}")}
        async for _ in graph.astream_events(
            {"messages": [HumanMessage(content="go")]}, config=cfg, version="v2"
        ):
            pass

    await asyncio.gather(drive("A", "wf-A"), drive("B", "wf-B"))

    assert seen == {"A": "wf-A", "B": "wf-B"}, seen
