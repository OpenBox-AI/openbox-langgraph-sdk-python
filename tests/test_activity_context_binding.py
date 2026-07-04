"""The base-SDK ``ActivityContext`` is bound around ACTUAL tool execution at
the LangGraph ``ToolNode`` request seam, with an id that matches the tool's own
lifecycle ``on_tool_start`` run id exactly.

The binder (``tool_activity_binding``) mints a canonical id, WRITES it into the
tool's ``config["run_id"]`` so ``on_tool_start`` (hence ``ToolStarted.activity_id``
in the handler) carries it, and runs the tool inside
``openbox_core.context.activity_scope(ctx, store=...)``. ``resolve_context``
checks that ContextVar tier FIRST, so any HTTP/file/db hook fired inside the
tool resolves to THIS tool's exact activity — no minted-but-unmatched id, no
single-active/last-registered guessing.

These tests drive REAL compiled LangGraph graphs. Where they need to compare
the bound context against the lifecycle event id, they capture BOTH the
``on_tool_start``/``on_tool_end`` run ids (from ``astream_events``) and the
``ActivityContext`` the tool body resolves mid-execution — the exact identity a
base hook would resolve.
"""

from __future__ import annotations

import asyncio
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
from openbox_core.context import ContextStore

from openbox_langgraph.config import GovernanceConfig
from openbox_langgraph.errors import OpenBoxConfigError
from openbox_langgraph.langgraph_handler import create_openbox_graph_handler
from openbox_langgraph.tool_activity_binding import bind_tools_activity_scope, turn_metadata


class _AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


class _RT:
    """Minimal stand-in for the parts of an OpenBoxRuntime the binder reads."""

    def __init__(self, store: ContextStore) -> None:
        self.context_store = store


# ─────────────────────────────────────────────────────────────
# Direct-graph harness: capture lifecycle run ids + bound contexts
# ─────────────────────────────────────────────────────────────


def _build_bound_graph(
    tools: list[Any],
    tool_calls: list[dict[str, Any]],
    *,
    store: ContextStore,
    config: GovernanceConfig | None = None,
    handle_tool_errors: bool = True,
) -> Any:
    """A model→tools→model graph whose ToolNode is bound by the OpenBox binder.

    The model emits ``tool_calls`` on the first pass, then finishes. Binding is
    applied to the compiled graph exactly as the handler applies it.
    """
    tool_node = ToolNode(tools, handle_tool_errors=handle_tool_errors)

    def model(state: _AgentState) -> dict[str, Any]:
        msgs = state["messages"]
        if msgs and any(getattr(m, "type", None) == "tool" for m in msgs):
            return {"messages": [AIMessage(content="done")]}
        return {"messages": [AIMessage(content="", tool_calls=tool_calls)]}

    def route(state: _AgentState) -> str:
        return "tools" if getattr(state["messages"][-1], "tool_calls", None) else END

    graph = StateGraph(_AgentState)
    graph.add_node("model", model)
    graph.add_node("tools", tool_node)
    graph.add_edge(START, "model")
    graph.add_conditional_edges("model", route, {"tools": "tools", END: END})
    graph.add_edge("tools", "model")
    compiled = graph.compile()
    bind_tools_activity_scope(
        compiled,
        core_runtime=_RT(store),
        config=config or GovernanceConfig(),
        resolve_tool_type=lambda name: None,
    )
    return compiled


async def _drive(graph: Any, *, workflow_id: str | None = None, run_id: str | None = None) -> dict:
    """Run the graph, returning on_tool_start/on_tool_end run ids by tool name."""
    cfg: dict[str, Any] = {}
    if workflow_id is not None and run_id is not None:
        cfg["metadata"] = turn_metadata(workflow_id, run_id)
    starts: dict[str, str] = {}
    ends: dict[str, str] = {}
    async for ev in graph.astream_events({"messages": []}, config=cfg, version="v2"):
        if ev["event"] == "on_tool_start":
            starts[ev["name"]] = str(ev["run_id"])
        elif ev["event"] == "on_tool_end":
            ends[ev["name"]] = str(ev["run_id"])
    return {"starts": starts, "ends": ends}


# ─────────────────────────────────────────────────────────────
# 1 + 2: exact tool span activity id (no minted-but-unmatched uuid)
# ─────────────────────────────────────────────────────────────


async def test_bound_activity_id_equals_on_tool_start_run_id() -> None:
    """The context bound around the tool body carries the SAME id the tool's
    ``on_tool_start`` event carries — so a hook resolving the bound context
    reports the exact lifecycle ``ActivityStarted.activity_id``, never a fresh
    unrelated uuid."""
    store = ContextStore()
    seen: dict[str, Any] = {}

    @tool
    def probe(query: str) -> str:
        """Record the activity id resolved mid-body."""
        ctx = store.current_activity_context()
        seen["activity_id"] = ctx.activity_id if ctx else None
        seen["activity_type"] = ctx.activity_type if ctx else None
        return "ok"

    graph = _build_bound_graph(
        [probe], [{"name": "probe", "args": {"query": "x"}, "id": "call-1"}], store=store
    )
    result = await _drive(graph, workflow_id="wf-1", run_id="run-1")

    assert seen["activity_type"] == "probe"
    # The bound id is a real id that MATCHES the lifecycle event — not an
    # unrelated minted uuid, and not None.
    assert seen["activity_id"] is not None
    assert seen["activity_id"] == result["starts"]["probe"]


# ─────────────────────────────────────────────────────────────
# 6: tool completion id parity
# ─────────────────────────────────────────────────────────────


async def test_started_and_completed_share_one_activity_id() -> None:
    """One id for the whole tool lifecycle: ``on_tool_start`` and ``on_tool_end``
    carry the same run id (which the handler maps to ActivityStarted /
    ActivityCompleted), and it equals the bound context id."""
    store = ContextStore()
    seen: dict[str, Any] = {}

    @tool
    def probe(query: str) -> str:
        """Record the bound id."""
        ctx = store.current_activity_context()
        seen["activity_id"] = ctx.activity_id if ctx else None
        return "ok"

    graph = _build_bound_graph(
        [probe], [{"name": "probe", "args": {"query": "x"}, "id": "call-1"}], store=store
    )
    result = await _drive(graph, workflow_id="wf-1", run_id="run-1")

    assert result["starts"]["probe"] == result["ends"]["probe"]
    assert seen["activity_id"] == result["starts"]["probe"]


# ─────────────────────────────────────────────────────────────
# 4: concurrent tool calls in one ToolNode each map to their own id
# ─────────────────────────────────────────────────────────────


async def test_concurrent_tool_calls_each_bind_their_own_activity() -> None:
    """Two tool calls dispatched together in ONE ToolNode: each tool body sees
    ONLY its own bound context id, matching its own ``on_tool_start`` event —
    zero cross-contamination."""
    store = ContextStore()
    seen: dict[str, str | None] = {}

    @tool
    async def alpha(query: str) -> str:
        """Record alpha's bound id."""
        await asyncio.sleep(0.02)  # force interleave with beta
        ctx = store.current_activity_context()
        seen["alpha"] = ctx.activity_id if ctx else None
        return "alpha-done"

    @tool
    async def beta(query: str) -> str:
        """Record beta's bound id."""
        await asyncio.sleep(0.02)
        ctx = store.current_activity_context()
        seen["beta"] = ctx.activity_id if ctx else None
        return "beta-done"

    graph = _build_bound_graph(
        [alpha, beta],
        [
            {"name": "alpha", "args": {"query": "a"}, "id": "call-a"},
            {"name": "beta", "args": {"query": "b"}, "id": "call-b"},
        ],
        store=store,
    )
    result = await _drive(graph, workflow_id="wf-1", run_id="run-1")

    assert seen["alpha"] == result["starts"]["alpha"]
    assert seen["beta"] == result["starts"]["beta"]
    assert seen["alpha"] != seen["beta"], "concurrent tool calls must not share an activity id"


# ─────────────────────────────────────────────────────────────
# 5: existing ToolNode wrappers compose
# ─────────────────────────────────────────────────────────────


async def test_existing_toolnode_wrappers_still_run_inside_the_bound_scope() -> None:
    """A user-provided ``wrap_tool_call``/``awrap_tool_call`` on the ToolNode
    keeps running, and the OpenBox binding wraps the actual execute call around
    it — the tool body still resolves the bound context."""
    store = ContextStore()
    order: list[str] = []
    seen: dict[str, Any] = {}

    async def user_awrap(request, execute):
        order.append("user_awrap_enter")
        try:
            return await execute(request)
        finally:
            order.append("user_awrap_exit")

    @tool
    async def probe(query: str) -> str:
        """Record the bound id from inside the user-wrapped tool."""
        ctx = store.current_activity_context()
        seen["activity_id"] = ctx.activity_id if ctx else None
        return "ok"

    tool_node = ToolNode([probe], awrap_tool_call=user_awrap)

    def model(state: _AgentState) -> dict[str, Any]:
        msgs = state["messages"]
        if msgs and any(getattr(m, "type", None) == "tool" for m in msgs):
            return {"messages": [AIMessage(content="done")]}
        tool_calls = [{"name": "probe", "args": {"query": "x"}, "id": "c1"}]
        return {"messages": [AIMessage(content="", tool_calls=tool_calls)]}

    def route(state: _AgentState) -> str:
        return "tools" if getattr(state["messages"][-1], "tool_calls", None) else END

    graph = StateGraph(_AgentState)
    graph.add_node("model", model)
    graph.add_node("tools", tool_node)
    graph.add_edge(START, "model")
    graph.add_conditional_edges("model", route, {"tools": "tools", END: END})
    graph.add_edge("tools", "model")
    compiled = graph.compile()
    bind_tools_activity_scope(
        compiled,
        core_runtime=_RT(store),
        config=GovernanceConfig(),
        resolve_tool_type=lambda n: None,
    )

    result = await _drive(compiled, workflow_id="wf-1", run_id="run-1")

    assert order == ["user_awrap_enter", "user_awrap_exit"], "user wrapper must run"
    assert seen["activity_id"] == result["starts"]["probe"], "OpenBox binding still applies"


async def test_sync_only_user_wrapper_still_runs_on_async_path() -> None:
    """LangGraph's ``_arun_one`` falls back to the SYNC wrapper when no async
    wrapper exists. Binding must preserve that: a user sync-only
    ``wrap_tool_call`` keeps running under ``astream_events``/``ainvoke``
    (openbox leaves ``_awrap_tool_call`` unset and composes the user wrapper on
    the sync seam), and the binding still applies with the exact lifecycle id."""
    store = ContextStore()
    order: list[str] = []
    seen: dict[str, Any] = {}

    @tool
    def probe(query: str) -> str:
        """Record the bound id from inside the sync tool."""
        ctx = store.current_activity_context()
        seen["activity_id"] = ctx.activity_id if ctx else None
        return "ok"

    def user_sync_wrap(request, execute):
        order.append("enter")
        result = execute(request)
        order.append("exit")
        return result

    tool_node = ToolNode([probe], wrap_tool_call=user_sync_wrap)

    def model(state: _AgentState) -> dict[str, Any]:
        msgs = state["messages"]
        if msgs and any(getattr(m, "type", None) == "tool" for m in msgs):
            return {"messages": [AIMessage(content="done")]}
        tool_calls = [{"name": "probe", "args": {"query": "x"}, "id": "c1"}]
        return {"messages": [AIMessage(content="", tool_calls=tool_calls)]}

    def route(state: _AgentState) -> str:
        return "tools" if getattr(state["messages"][-1], "tool_calls", None) else END

    graph = StateGraph(_AgentState)
    graph.add_node("model", model)
    graph.add_node("tools", tool_node)
    graph.add_edge(START, "model")
    graph.add_conditional_edges("model", route, {"tools": "tools", END: END})
    graph.add_edge("tools", "model")
    compiled = graph.compile()
    bind_tools_activity_scope(
        compiled,
        core_runtime=_RT(store),
        config=GovernanceConfig(),
        resolve_tool_type=lambda n: None,
    )

    # The async seam stays unset — LangGraph's own async→sync-wrapper fallback
    # must keep routing through openbox_sync (which composes the user wrapper).
    assert tool_node._awrap_tool_call is None

    result = await _drive(compiled, workflow_id="wf-1", run_id="run-1")

    assert order == ["enter", "exit"], "user sync wrapper must run on the async path"
    assert seen["activity_id"] == result["starts"]["probe"]


async def test_retry_wrapper_reexecutes_under_the_same_canonical_id() -> None:
    """A user wrapper may call ``execute`` MORE THAN ONCE (retries — LangGraph
    documents the multi-call contract). ``BaseTool`` consumes ``config["run_id"]``
    with ``pop()`` per run, so binding re-installs the canonical id before every
    delegate call: all attempts' ``on_tool_start`` events AND the bound scope
    share ONE id — the second attempt never drifts to a fresh LangChain id."""
    store = ContextStore()
    bound_ids: list[str | None] = []

    @tool
    async def probe(query: str) -> str:
        """Record the bound id per attempt."""
        ctx = store.current_activity_context()
        bound_ids.append(ctx.activity_id if ctx else None)
        return "ok"

    async def retry_awrap(request, execute):
        await execute(request)  # first attempt, result discarded
        return await execute(request)  # retry

    tool_node = ToolNode([probe], awrap_tool_call=retry_awrap)

    def model(state: _AgentState) -> dict[str, Any]:
        msgs = state["messages"]
        if msgs and any(getattr(m, "type", None) == "tool" for m in msgs):
            return {"messages": [AIMessage(content="done")]}
        tool_calls = [{"name": "probe", "args": {"query": "x"}, "id": "c1"}]
        return {"messages": [AIMessage(content="", tool_calls=tool_calls)]}

    def route(state: _AgentState) -> str:
        return "tools" if getattr(state["messages"][-1], "tool_calls", None) else END

    graph = StateGraph(_AgentState)
    graph.add_node("model", model)
    graph.add_node("tools", tool_node)
    graph.add_edge(START, "model")
    graph.add_conditional_edges("model", route, {"tools": "tools", END: END})
    graph.add_edge("tools", "model")
    compiled = graph.compile()
    bind_tools_activity_scope(
        compiled,
        core_runtime=_RT(store),
        config=GovernanceConfig(),
        resolve_tool_type=lambda n: None,
    )

    start_ids: list[str] = []
    cfg = {"metadata": turn_metadata("wf-1", "run-1")}
    async for ev in compiled.astream_events({"messages": []}, config=cfg, version="v2"):
        if ev["event"] == "on_tool_start":
            start_ids.append(str(ev["run_id"]))

    assert len(bound_ids) == 2, "the tool must have run twice (attempt + retry)"
    one_id = {*start_ids, *bound_ids}
    assert len(one_id) == 1 and None not in one_id, (
        f"every attempt and the bound scope must share one canonical id, got {one_id}"
    )


# ─────────────────────────────────────────────────────────────
# 7: strict vs default no-context policy
# ─────────────────────────────────────────────────────────────


async def test_strict_mode_raises_when_turn_context_cannot_be_proven() -> None:
    """Driven WITHOUT turn metadata (no governed turn), strict mode raises
    before executing the tool rather than running it unbound."""
    store = ContextStore()

    @tool
    def probe(query: str) -> str:
        """Should never run in strict mode without a turn."""
        return "ok"

    graph = _build_bound_graph(
        [probe],
        [{"name": "probe", "args": {"query": "x"}, "id": "c1"}],
        store=store,
        config=GovernanceConfig(strict_activity_context=True),
        handle_tool_errors=False,  # let the strict raise propagate, not become a ToolMessage
    )

    with pytest.raises(OpenBoxConfigError):
        await _drive(graph)  # no workflow_id/run_id → no turn metadata


async def test_default_mode_executes_unbound_when_turn_context_absent(caplog) -> None:
    """Default mode logs a warning once and runs the tool UNBOUND (its body
    resolves no ActivityContext) rather than raising or fabricating a context."""
    store = ContextStore()
    seen: dict[str, Any] = {}

    @tool
    def probe(query: str) -> str:
        """Record that no context is bound outside a governed turn."""
        seen["ctx"] = store.current_activity_context()
        return "ok"

    graph = _build_bound_graph(
        [probe], [{"name": "probe", "args": {"query": "x"}, "id": "c1"}], store=store
    )

    with caplog.at_level("WARNING", logger="openbox_langgraph.tool_activity_binding"):
        await _drive(graph)  # no turn metadata

    assert "ctx" in seen and seen["ctx"] is None, "no context should be bound"
    assert any("executing tool UNBOUND" in r.message for r in caplog.records)


# ─────────────────────────────────────────────────────────────
# 3: no fallback guessing
# ─────────────────────────────────────────────────────────────


def test_two_registered_contexts_never_resolve_an_unrelated_trace() -> None:
    """With TWO exact contexts registered and NO ContextVar bound, resolving a
    trace id matching NEITHER returns None — the store never guesses one of the
    two active activities (single-active/last-registered fallback is gone)."""
    from openbox_core.contracts.context import ActivityContext

    store = ContextStore()

    def _ctx(activity_id: str) -> ActivityContext:
        return ActivityContext(
            workflow_id="wf",
            run_id="run",
            workflow_type="W",
            task_queue="q",
            activity_id=activity_id,
            activity_type="probe",
        )

    store.register_trace(111, _ctx("act-A"))
    store.register_trace(222, _ctx("act-B"))

    # An unrelated trace id resolves NOTHING — not act-A, not act-B.
    assert store.context_for_trace(999) is None
    # And with no ContextVar bind, the primary tier is empty too.
    assert store.current_activity_context() is None


# ─────────────────────────────────────────────────────────────
# Real handler path + idempotency
# ─────────────────────────────────────────────────────────────


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


def _build_probe_graph(store_holder: dict[str, Any], captured: list[Any], probe_path):
    """react-style graph whose single tool records the bound ActivityContext."""

    @tool
    def probe_tool(text: str) -> str:
        """Record the ActivityContext resolved from the runtime store mid-execution."""
        store = store_holder["store"]
        captured.append(store.current_activity_context())
        probe_path.read_text()  # a real instrumented file op under the bound context
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
    """End to end through the REAL handler (non-injected, use_core_instrumentation):
    while the tool body runs, the runtime's private ContextStore resolves THIS
    tool's ActivityContext via the ContextVar tier — exactly what a base hook sees."""
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
async def test_toolnode_bound_idempotently(governance_api, tmp_path) -> None:
    """Building a second handler over the same graph must not re-wrap the
    ToolNode — the composed wrappers from the first handler stay in place."""
    store_holder: dict[str, Any] = {}
    captured: list[Any] = []
    probe_path = tmp_path / "probe.txt"
    probe_path.write_text("x")
    graph = _build_probe_graph(store_holder, captured, probe_path)

    tool_node = graph.nodes["tools"].bound  # type: ignore[attr-defined]

    h1 = create_openbox_graph_handler(
        graph=graph, api_url=governance_api.url, api_key="obx_test_idem1", validate=False,
        use_core_instrumentation=True,
    )
    wrapped_async_once = tool_node._awrap_tool_call
    wrapped_sync_once = tool_node._wrap_tool_call
    h2 = create_openbox_graph_handler(
        graph=graph, api_url=governance_api.url, api_key="obx_test_idem2", validate=False,
        use_core_instrumentation=True,
    )
    try:
        assert getattr(tool_node, "_openbox_activity_bound", False) is True
        # Second bind is a no-op — the composed wrappers are unchanged.
        assert tool_node._awrap_tool_call is wrapped_async_once
        assert tool_node._wrap_tool_call is wrapped_sync_once
    finally:
        h1._core_runtime.close()  # type: ignore[union-attr]
        h2._core_runtime.close()  # type: ignore[union-attr]


@pytest.mark.asyncio
async def test_concurrent_turns_do_not_cross_contaminate() -> None:
    """Two turns driven CONCURRENTLY on two graphs sharing ONE store each
    resolve their OWN workflow_id inside the tool body — the per-turn ids ride
    each tool's own RunnableConfig, not a shared module ContextVar."""
    store = ContextStore()
    seen: dict[str, str | None] = {}

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
        bind_tools_activity_scope(
            compiled, core_runtime=_RT(store), config=GovernanceConfig(),
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
