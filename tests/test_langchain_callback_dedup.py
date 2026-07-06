"""Phase 4 — dedup/ownership invariants between the pure-LangChain-Core tool
callback and the LangGraph consumer's own ToolStarted/ToolCompleted sends.

Covers: consumer skips ONLY sent-flagged tool events (C1); a prepared-but-
never-started record still falls through to consumer governance; a
subagent-gated handler installs NO callback/bridge at all (C1 blackout
regression); the sync-only-tool corner is fail-closed PRE-body via the
installed SYNC handler, not a post-body consumer catch (C2); a nested-agent
graph's inner (unbound) ToolNode sends exactly once, via the consumer, never
the record-less callback (C8); duplicate/idempotent dispatch under the
measured cross-dispatch does not double-send; concurrent tool calls resolve
distinct ids; a tool failure closes the same activity_id as a failed
completion with bridge + abort-marks cleaned; and the injected-client handler
(no core runtime) keeps the pre-phase-4 consumer-only golden ordering.
"""

from __future__ import annotations

from typing import Annotated, Any, TypedDict

import pytest
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.tools import tool
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

pytest.importorskip("openbox_core")

from openbox_core.conformance.fake_core import FakeCore
from openbox_core.conformance.instrumentation import installed_conformance_runtime
from openbox_core.context import ContextStore

from openbox_langgraph.client import GovernanceClient
from openbox_langgraph.core_adapter import LangGraphFrameworkAdapter
from openbox_langgraph.errors import GovernanceBlockedError
from openbox_langgraph.langgraph_handler import (
    OpenBoxLangGraphHandler,
    OpenBoxLangGraphHandlerOptions,
)
from openbox_langgraph.tool_activity_binding import bind_tools_activity_scope
from openbox_langgraph.types import GovernanceVerdictResponse, LangChainGovernanceEvent, Verdict


class _AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


class _AllowEverythingClient(GovernanceClient):
    """Always-ALLOW baseline client — subclassed per-test to record events."""

    def __init__(self) -> None:
        super().__init__(api_url="https://core.openbox.ai", api_key="obx_test_abc")

    async def evaluate_event(self, event: Any) -> GovernanceVerdictResponse | None:
        return GovernanceVerdictResponse(verdict=Verdict.ALLOW)


class _RecordingClient(_AllowEverythingClient):
    """Records every lifecycle event THIS SDK's consumer sends (never the
    callback's — the callback goes through the runtime's gate, a separate
    transport captured by `FakeCore`)."""

    def __init__(self) -> None:
        super().__init__()
        self.events: list[LangChainGovernanceEvent] = []

    async def evaluate_event(self, event: Any) -> GovernanceVerdictResponse | None:
        self.events.append(event)
        return await super().evaluate_event(event)


# LLM lifecycle is callback-owned too, but this module is scoped to tool
# ownership/dedup. Exclude the normalized LLM activity type from tool filters.
_LLM_ACTIVITY_TYPES = frozenset({"llm_call"})


def _lifecycle_payloads(
    fake_core: FakeCore, event_type: str, *, tool_only: bool = True
) -> list[dict[str, Any]]:
    """See test_langchain_callback_tool_ordering.py's identical helper docstring.

    `tool_only=True` (default) excludes Phase 5's LLM lifecycle payloads
    (`_LLM_ACTIVITY_TYPES`, this module's one fake model) — this module is
    scoped to TOOL callback ownership.
    """
    payloads = [p for p in fake_core.payloads if p.get("event_type") == event_type]
    if tool_only:
        payloads = [p for p in payloads if p.get("activity_type") not in _LLM_ACTIVITY_TYPES]
    return payloads


@tool
def echo_tool(text: str) -> str:
    """Echo text back."""
    return f"echo: {text}"


def _build_tool_call_graph(
    tool_fn: Any = echo_tool,
    tool_name: str = "echo_tool",
    *,
    handle_tool_errors: bool = False,
) -> Any:
    model = FakeMessagesListChatModel(
        responses=[
            AIMessage(
                content="",
                tool_calls=[{"name": tool_name, "args": {"text": "hi"}, "id": "call-1"}],
            ),
            AIMessage(content="done"),
        ]
    )

    async def call_model(state: _AgentState) -> dict[str, Any]:
        result = await model.ainvoke(state["messages"])
        return {"messages": [result]}

    def should_continue(state: _AgentState) -> str:
        last = state["messages"][-1]
        if getattr(last, "tool_calls", None):
            return "tools"
        return END

    graph = StateGraph(_AgentState)
    graph.add_node("agent", call_model)
    graph.add_node("tools", ToolNode([tool_fn], handle_tool_errors=handle_tool_errors))
    graph.add_edge(START, "agent")
    graph.add_conditional_edges("agent", should_continue, {"tools": "tools", END: END})
    graph.add_edge("tools", "agent")
    return graph.compile()


def _build_handler_with_bridge(
    graph: Any, runtime: Any, client: GovernanceClient | None = None
) -> OpenBoxLangGraphHandler:
    """C1-armed handler: injected client (skips `__init__`'s own runtime build),
    `_core_runtime` pointed at the conformance runtime, bridge + ToolNode
    binding wired explicitly (mirrors `__init__`'s C1 gating for
    `resolve_subagent_name is None`)."""
    from openbox_langchain import ActivityBridge

    handler = OpenBoxLangGraphHandler(
        graph=graph,
        options=OpenBoxLangGraphHandlerOptions(client=client or _AllowEverythingClient()),
    )
    handler._core_runtime = runtime  # type: ignore[attr-defined]
    handler._activity_bridge = ActivityBridge()  # type: ignore[attr-defined]
    bind_tools_activity_scope(
        graph,
        core_runtime=runtime,
        config=handler._config,  # type: ignore[attr-defined]
        resolve_tool_type=lambda name: None,
        bridge=handler._activity_bridge,  # type: ignore[attr-defined]
    )
    return handler


def _build_handler_subagent_gated(
    graph: Any, runtime: Any, client: GovernanceClient | None = None
) -> OpenBoxLangGraphHandler:
    """C1 blackout regression setup: a `resolve_subagent_name` resolver is
    set, so a REAL `__init__` (not the manual wiring above) must leave
    `_activity_bridge is None` and install NO callback — exercises the
    production gating code path directly, not a test-only bypass."""
    handler = OpenBoxLangGraphHandler(
        graph=graph,
        options=OpenBoxLangGraphHandlerOptions(
            client=client or _AllowEverythingClient(),
            resolve_subagent_name=lambda event: None,
        ),
    )
    handler._core_runtime = runtime  # type: ignore[attr-defined]
    return handler


async def test_consumer_skips_only_sent_tool_events() -> None:
    """C1 — with the callback installed and ALLOW verdicts throughout, the
    consumer's OWN ToolStarted/ToolCompleted sends must be skipped (the
    callback already sent+enforced both), never double-sent."""
    fake_core = FakeCore()  # ALLOW
    store = ContextStore()
    adapter = LangGraphFrameworkAdapter(context_store=store)
    graph = _build_tool_call_graph()
    client = _RecordingClient()

    with installed_conformance_runtime(fake_core, adapter, store) as runtime:
        handler = _build_handler_with_bridge(graph, runtime, client)
        await handler.ainvoke(
            {"messages": [HumanMessage(content="hi")]},
            config={"configurable": {"thread_id": "dedup-sent-flags"}},
        )

    consumer_tool_events = [
        e for e in client.events if e.event_type in ("ToolStarted", "ToolCompleted")
    ]
    assert consumer_tool_events == [], "consumer must not re-send callback-owned tool events"
    assert _lifecycle_payloads(fake_core, "ActivityStarted")
    assert _lifecycle_payloads(fake_core, "ActivityCompleted")


async def test_prepared_not_started_record_still_consumer_governed() -> None:
    """A record can be PREPARED (bridge.prepare_tool ran) without ever being
    STARTED (the callback's on_tool_start never fires — e.g. no callback
    installed for this specific invocation). Ownership is sent-flags only, so
    the consumer must still send+enforce."""
    fake_core = FakeCore()
    store = ContextStore()
    adapter = LangGraphFrameworkAdapter(context_store=store)
    graph = _build_tool_call_graph()
    client = _RecordingClient()

    with installed_conformance_runtime(fake_core, adapter, store) as runtime:
        handler = _build_handler_with_bridge(graph, runtime, client)
        # Remove the callback from the turn's config WITHOUT touching the
        # bridge — simulates "prepared, never started": the ToolNode wrapper
        # still prepares (bridge armed), but no callback ever marks it sent.
        # Phase 5 retired `_GuardrailsCallbackHandler` — the pure-LangChain-Core
        # callbacks now own BOTH tool and LLM lifecycle, so dropping every
        # installed callback is what simulates "no callback ran this turn".
        original_governed_config = handler._governed_config  # type: ignore[attr-defined]

        def _governed_config_no_callback(config: Any, **kwargs: Any) -> Any:
            cfg = original_governed_config(config, **kwargs)
            cfg["callbacks"] = []
            return cfg

        handler._governed_config = _governed_config_no_callback  # type: ignore[method-assign]

        await handler.ainvoke(
            {"messages": [HumanMessage(content="hi")]},
            config={"configurable": {"thread_id": "dedup-prepared-not-started"}},
        )

    consumer_tool_started = [e for e in client.events if e.event_type == "ToolStarted"]
    consumer_tool_completed = [e for e in client.events if e.event_type == "ToolCompleted"]
    assert consumer_tool_started, "prepared-but-never-started must fall through to the consumer"
    assert consumer_tool_completed


async def test_subagent_gated_handler_has_no_bridge_or_callback() -> None:
    """C1 blackout regression: a handler with `resolve_subagent_name` set must
    build `_activity_bridge is None` in REAL `__init__` — no callback
    installed, consumer governs every tool event exactly as pre-phase-4."""
    fake_core = FakeCore()
    store = ContextStore()
    adapter = LangGraphFrameworkAdapter(context_store=store)
    graph = _build_tool_call_graph()
    client = _RecordingClient()

    with installed_conformance_runtime(fake_core, adapter, store) as runtime:
        handler = _build_handler_subagent_gated(graph, runtime, client)
        assert handler._activity_bridge is None  # type: ignore[attr-defined]

        await handler.ainvoke(
            {"messages": [HumanMessage(content="hi")]},
            config={"configurable": {"thread_id": "dedup-subagent-gated"}},
        )

    # No callback installed -> Core's fake transport sees ZERO lifecycle
    # payloads from the callback path; the consumer is the sole sender.
    assert _lifecycle_payloads(fake_core, "ActivityStarted") == []
    assert _lifecycle_payloads(fake_core, "ActivityCompleted") == []
    consumer_tool_started = [e for e in client.events if e.event_type == "ToolStarted"]
    consumer_tool_completed = [e for e in client.events if e.event_type == "ToolCompleted"]
    assert consumer_tool_started
    assert consumer_tool_completed


async def test_sync_only_tool_corner_is_fail_closed_pre_body() -> None:
    """C2 — a sync-only tool (`@tool` on a plain function, no async impl) run
    through the async graph path (`ainvoke` -> `run_in_executor` ->
    `BaseTool.run`'s SYNC callback manager) is governed PRE-body by the
    installed SYNC handler: BLOCK stops the body from ever running, not a
    post-body consumer catch."""
    body_ran = {"value": False}

    @tool
    def sync_only_tool(text: str) -> str:
        """A sync-only tool with no async implementation."""
        body_ran["value"] = True
        return f"ran: {text}"

    # ONE leading ALLOW is the agent LLM call's ActivityCompleted (Phase 5 —
    # the SAME callback now also governs LLM lifecycle; its START is
    # pre-screen-reused with no gate call, but its COMPLETION makes a real
    # `gate.aevaluate` call that DOES consume from this FIFO queue before the
    # sync-only tool call this test targets ever reaches the gate).
    fake_core = FakeCore(
        {"verdict": "allow"},
        {"verdict": "block", "reason": "sync corner blocked"},
    )
    store = ContextStore()
    adapter = LangGraphFrameworkAdapter(context_store=store)
    graph = _build_tool_call_graph(tool_fn=sync_only_tool, tool_name="sync_only_tool")

    with installed_conformance_runtime(fake_core, adapter, store) as runtime:
        handler = _build_handler_with_bridge(graph, runtime)
        with pytest.raises(GovernanceBlockedError):
            await handler.ainvoke(
                {"messages": [HumanMessage(content="hi")]},
                config={"configurable": {"thread_id": "dedup-sync-corner"}},
            )

    assert body_ran["value"] is False, "BLOCK must stop the sync-only tool body PRE-body"


async def test_nested_agent_graph_inner_tool_sent_once() -> None:
    """C8 — a nested-agent (subgraph) ToolNode is unbound (`bind_tools_activity_scope`'s
    depth-4 walk cannot reach inside a compiled subgraph node), so the
    callback fires there with `record_less_ok=False` and NO record — it must
    NOT send. The consumer, which DOES see the inner on_tool_start/on_tool_end
    stream events (LangGraph recurses `astream_events` into subgraphs), must
    remain the sole sender — exactly once, never a double row."""

    @tool
    def inner_tool(text: str) -> str:
        """Inner subgraph tool."""
        return f"inner: {text}"

    inner_model = FakeMessagesListChatModel(
        responses=[
            AIMessage(
                content="",
                tool_calls=[{"name": "inner_tool", "args": {"text": "x"}, "id": "inner-1"}],
            ),
            AIMessage(content="inner done"),
        ]
    )

    async def inner_call_model(state: _AgentState) -> dict[str, Any]:
        result = await inner_model.ainvoke(state["messages"])
        return {"messages": [result]}

    def inner_should_continue(state: _AgentState) -> str:
        last = state["messages"][-1]
        if getattr(last, "tool_calls", None):
            return "tools"
        return END

    inner_graph = StateGraph(_AgentState)
    inner_graph.add_node("agent", inner_call_model)
    inner_graph.add_node("tools", ToolNode([inner_tool]))
    inner_graph.add_edge(START, "agent")
    inner_graph.add_conditional_edges("agent", inner_should_continue, {"tools": "tools", END: END})
    inner_graph.add_edge("tools", "agent")
    inner_compiled = inner_graph.compile()

    outer_graph = StateGraph(_AgentState)
    outer_graph.add_node("subagent", inner_compiled)
    outer_graph.add_edge(START, "subagent")
    outer_graph.add_edge("subagent", END)
    outer_compiled = outer_graph.compile()

    fake_core = FakeCore()
    store = ContextStore()
    adapter = LangGraphFrameworkAdapter(context_store=store)
    client = _RecordingClient()

    with installed_conformance_runtime(fake_core, adapter, store) as runtime:
        handler = _build_handler_with_bridge(outer_compiled, runtime, client)
        # bind_tools_activity_scope cannot reach the inner ToolNode — confirm
        # the precondition this test relies on.
        result = await handler.ainvoke(
            {"messages": [HumanMessage(content="hi")]},
            config={"configurable": {"thread_id": "dedup-nested"}},
        )

    tool_message = next(m for m in result["messages"] if m.type == "tool")
    assert tool_message.content == "inner: x"

    # Callback never sent (no bridge record, record_less_ok=False):
    assert _lifecycle_payloads(fake_core, "ActivityStarted") == []
    assert _lifecycle_payloads(fake_core, "ActivityCompleted") == []
    # Consumer sent exactly once each — no double row.
    consumer_started = [e for e in client.events if e.event_type == "ToolStarted"]
    consumer_completed = [e for e in client.events if e.event_type == "ToolCompleted"]
    assert len(consumer_started) == 1
    assert len(consumer_completed) == 1


async def test_concurrent_tool_calls_resolve_distinct_ids() -> None:
    """Two tool calls in the SAME ToolNode invocation each get their own
    canonical id/bridge record — no cross-contamination."""

    @tool
    def concurrent_tool(text: str) -> str:
        """A tool called concurrently, twice, in one turn."""
        return f"done: {text}"

    model = FakeMessagesListChatModel(
        responses=[
            AIMessage(
                content="",
                tool_calls=[
                    {"name": "concurrent_tool", "args": {"text": "a"}, "id": "call-a"},
                    {"name": "concurrent_tool", "args": {"text": "b"}, "id": "call-b"},
                ],
            ),
            AIMessage(content="done"),
        ]
    )

    async def call_model(state: _AgentState) -> dict[str, Any]:
        result = await model.ainvoke(state["messages"])
        return {"messages": [result]}

    def should_continue(state: _AgentState) -> str:
        last = state["messages"][-1]
        if getattr(last, "tool_calls", None):
            return "tools"
        return END

    graph = StateGraph(_AgentState)
    graph.add_node("agent", call_model)
    graph.add_node("tools", ToolNode([concurrent_tool]))
    graph.add_edge(START, "agent")
    graph.add_conditional_edges("agent", should_continue, {"tools": "tools", END: END})
    graph.add_edge("tools", "agent")
    compiled = graph.compile()

    fake_core = FakeCore()
    store = ContextStore()
    adapter = LangGraphFrameworkAdapter(context_store=store)

    with installed_conformance_runtime(fake_core, adapter, store) as runtime:
        handler = _build_handler_with_bridge(compiled, runtime)
        await handler.ainvoke(
            {"messages": [HumanMessage(content="hi")]},
            config={"configurable": {"thread_id": "dedup-concurrent"}},
        )

    started = _lifecycle_payloads(fake_core, "ActivityStarted")
    assert len(started) == 2
    ids = {p["activity_id"] for p in started}
    assert len(ids) == 2, "each concurrent tool call must resolve its own distinct activity id"


async def test_tool_failure_closes_same_id_with_bridge_and_abort_marks_clean() -> None:
    """A tool raising inside its body still closes with a FAILED
    ActivityCompleted on the SAME activity_id ActivityStarted opened, and
    after the turn, the bridge + `_aborted_activities` are both clean."""

    @tool
    def failing_tool(text: str) -> str:
        """A tool that always raises."""
        raise RuntimeError("tool exploded")

    graph = _build_tool_call_graph(
        tool_fn=failing_tool, tool_name="failing_tool", handle_tool_errors=True
    )
    fake_core = FakeCore()
    store = ContextStore()
    adapter = LangGraphFrameworkAdapter(context_store=store)

    with installed_conformance_runtime(fake_core, adapter, store) as runtime:
        handler = _build_handler_with_bridge(graph, runtime)
        # `handle_tool_errors=True` turns the tool exception into an error
        # ToolMessage and the turn continues (documented behavior — governance
        # still fires start+failed-completion for the SAME id; the turn
        # itself is not aborted by a plain body exception).
        result = await handler.ainvoke(
            {"messages": [HumanMessage(content="hi")]},
            config={"configurable": {"thread_id": "dedup-tool-failure"}},
        )

    started = _lifecycle_payloads(fake_core, "ActivityStarted")
    completed = _lifecycle_payloads(fake_core, "ActivityCompleted")
    assert started and completed
    assert started[0]["activity_id"] == completed[0]["activity_id"]
    assert completed[0].get("error") or completed[0].get("status") == "failed"

    bridge = handler._activity_bridge  # type: ignore[attr-defined]
    assert bridge._workflows == {}  # type: ignore[attr-defined]
    assert store._aborted_activities == set()  # type: ignore[attr-defined]
    assert result["messages"][-1].type in ("ai",)  # turn continued past the tool error


async def test_injected_client_handler_keeps_consumer_only_golden_ordering() -> None:
    """The injected-client (lifecycle-only) path builds NO core runtime at
    all, so it can never install the callback — `_activity_bridge` is None
    and every tool event is consumer-governed, matching the pre-phase-4
    golden ordering exactly (no wire-shape change for this seam)."""
    client = _RecordingClient()
    graph = _build_tool_call_graph()
    handler = OpenBoxLangGraphHandler(graph, OpenBoxLangGraphHandlerOptions(client=client))

    assert handler._core_runtime is None  # type: ignore[attr-defined]
    assert handler._activity_bridge is None  # type: ignore[attr-defined]

    await handler.ainvoke(
        {"messages": [HumanMessage(content="hi")]},
        config={"configurable": {"thread_id": "dedup-injected-client"}},
    )

    consumer_started = [e for e in client.events if e.event_type == "ToolStarted"]
    consumer_completed = [e for e in client.events if e.event_type == "ToolCompleted"]
    assert len(consumer_started) == 1
    assert len(consumer_completed) == 1


async def test_duplicate_dispatch_under_cross_dispatch_is_deduped() -> None:
    """The measured cross-dispatch (async manager also runs a sync handler,
    and vice-versa — Phase 0) must not double-send: exactly one
    ActivityStarted and one ActivityCompleted reach Core for a single tool
    call, even though both the async and sync core callbacks are installed
    and BOTH fire on every event."""
    fake_core = FakeCore()
    store = ContextStore()
    adapter = LangGraphFrameworkAdapter(context_store=store)
    graph = _build_tool_call_graph()

    with installed_conformance_runtime(fake_core, adapter, store) as runtime:
        handler = _build_handler_with_bridge(graph, runtime)
        await handler.ainvoke(
            {"messages": [HumanMessage(content="hi")]},
            config={"configurable": {"thread_id": "dedup-cross-dispatch"}},
        )

    assert len(_lifecycle_payloads(fake_core, "ActivityStarted")) == 1
    assert len(_lifecycle_payloads(fake_core, "ActivityCompleted")) == 1
