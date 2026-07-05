"""Phase 4 — pure-LangChain-Core tool callback wired into the LangGraph handler.

Drives a REAL `create_react_agent`-shape graph (one node, one `ToolNode`) with
a REAL `OpenBoxRuntime` (fake Core transport, real gate/adapter) so the
callback actually fires through `astream_events` exactly as production does:
`_governed_config` installs both `OpenBoxLangChainCore{Async,Sync}CallbackHandler`
under the C1 condition, `tool_activity_binding.bind_tools_activity_scope`
prepares the bridge record before `execute()`, and the callback's
`run_inline=True` dispatch sends ActivityStarted/ActivityCompleted BEFORE the
handler's own stream-consumer sees the corresponding `on_tool_start`/
`on_tool_end` event (the Phase 0-measured inline-first ordering C1 relies on).

Covers: wrapper-prepares-before-execute; `run_id == prepared activity_id`;
ActivityStarted evaluated before the tool body's side effect; BLOCK stops the
body and the failed completion closes the row (C6); REQUIRE_APPROVAL polls the
TOOL's real activity_id (C5); ToolCompleted BLOCK/HALT/REQUIRE_APPROVAL is
enforced from the stashed verdict, not dropped (P1); streaming propagates
pending-approval with bridge + abort-marks cleaned after cleanup.
"""

from __future__ import annotations

from typing import Annotated, Any, TypedDict
from unittest.mock import AsyncMock, patch

import pytest
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
from openbox_langgraph.types import GovernanceVerdictResponse, Verdict

_created_runtimes: list[Any] = []


@pytest.fixture(autouse=True)
def _close_created_runtimes() -> Any:
    yield
    while _created_runtimes:
        _created_runtimes.pop().close()


class _AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


class _AllowEverythingClient(GovernanceClient):
    """Always-ALLOW for this SDK's OWN lifecycle events (WorkflowStarted,
    LLMStarted pre-screen, consumer ToolStarted/ToolCompleted when unowned) —
    isolates every test to the callback's OWN gate-routed verdicts."""

    def __init__(self) -> None:
        super().__init__(api_url="https://core.openbox.ai", api_key="obx_test_abc")

    async def evaluate_event(self, event: Any) -> GovernanceVerdictResponse | None:
        return GovernanceVerdictResponse(verdict=Verdict.ALLOW)


_body_effects: list[str] = []


@tool
def recording_tool(text: str) -> str:
    """A tool whose body records that it ran, for started-before-body ordering."""
    _body_effects.append(text)
    return f"echo: {text}"


def _build_tool_call_graph() -> Any:
    """LLM emits one tool call to `recording_tool`, then a final reply."""
    from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel

    model = FakeMessagesListChatModel(
        responses=[
            AIMessage(
                content="",
                tool_calls=[{"name": "recording_tool", "args": {"text": "hi"}, "id": "call-1"}],
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
    graph.add_node("tools", ToolNode([recording_tool]))
    graph.add_edge(START, "agent")
    graph.add_conditional_edges("agent", should_continue, {"tools": "tools", END: END})
    graph.add_edge("tools", "agent")
    return graph.compile()


def _build_handler(graph: Any, runtime: Any) -> OpenBoxLangGraphHandler:
    """Injected client keeps `__init__` from building its OWN core runtime;
    `_core_runtime` is then pointed at the conformance runtime so the
    callback's gate/adapter are the REAL ones under test. `_activity_bridge`
    and the ToolNode binding are built here explicitly (mirrors `__init__`'s
    own C1 gating) since the injected-client branch skips both."""
    from openbox_langchain import ActivityBridge

    handler = OpenBoxLangGraphHandler(
        graph=graph, options=OpenBoxLangGraphHandlerOptions(client=_AllowEverythingClient())
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


@pytest.fixture(autouse=True)
def _reset_body_effects() -> Any:
    _body_effects.clear()
    yield
    _body_effects.clear()


# Phase 5 — the pure-LangChain-Core LLM mixin sets `activity_type` to the
# serialized model's `name` on ActivityStarted (this module's one fake model,
# `FakeMessagesListChatModel`) but hardcodes the literal string `"llm"` on
# ActivityCompleted (`core_callback_async_llm_mixin.py`'s `_finish_llm` calls
# `send_llm_completed(options, activity_id, "llm", ...)`) — an asymmetry in
# `openbox_langchain` (a separate package/phase), not something this test
# module's filter can avoid by using one label.
_LLM_ACTIVITY_TYPES = frozenset({"FakeMessagesListChatModel", "llm"})


def _lifecycle_payloads(
    fake_core: FakeCore, event_type: str, *, tool_only: bool = True
) -> list[dict[str, Any]]:
    """Filter `FakeCore.payloads` for a lifecycle (non-hook) event_type.

    `FakeCore.started_payloads`/`completed_payloads` filter on hook-shaped
    `spans[0].stage` — ActivityStarted/ActivityCompleted LIFECYCLE envelopes
    (what the pure-LangChain-Core callback sends) carry no `spans` field at
    all, so this filters on the flat `event_type` wire field instead
    (`EventEnvelope.to_payload_dict` — see openbox_core/contracts/events.py).

    Since Phase 5, the SAME callback also owns LLM lifecycle — its
    ActivityStarted/ActivityCompleted payloads land in `fake_core.payloads`
    too (there is no separate `tool_name` wire field distinguishing them —
    the tool callback sets `activity_type=tool_name`, see
    `core_callback_tool_start.py`). This test module is scoped to the TOOL
    callback, so `tool_only=True` (default) excludes `_LLM_ACTIVITY_TYPES`.
    """
    payloads = [p for p in fake_core.payloads if p.get("event_type") == event_type]
    if tool_only:
        payloads = [p for p in payloads if p.get("activity_type") not in _LLM_ACTIVITY_TYPES]
    return payloads


async def test_wrapper_prepares_bridge_before_execute_and_run_id_matches() -> None:
    """The ToolNode wrapper prepares a bridge record BEFORE `execute()` runs,
    and the callback's `on_tool_start` sees `run_id == prepared activity_id`
    (the canonical id `tool_activity_binding` installs into `config["run_id"]`)."""
    fake_core = FakeCore()  # empty queue -> ALLOW
    store = ContextStore()
    adapter = LangGraphFrameworkAdapter(context_store=store)
    graph = _build_tool_call_graph()

    with installed_conformance_runtime(fake_core, adapter, store) as runtime:
        handler = _build_handler(graph, runtime)
        bridge = handler._activity_bridge  # type: ignore[attr-defined]
        swept: list[Any] = []
        original_sweep = bridge.sweep_workflow

        def _spy_sweep(workflow_id: str) -> list[Any]:
            records = original_sweep(workflow_id)
            swept.extend(records)
            return records

        bridge.sweep_workflow = _spy_sweep  # type: ignore[method-assign]

        result = await handler.ainvoke(
            {"messages": [HumanMessage(content="please run the tool")]},
            config={"configurable": {"thread_id": "ordering-thread-1"}},
        )

    assert _body_effects == ["hi"], "tool body ran exactly once"
    tool_message = next(m for m in result["messages"] if m.type == "tool")
    assert tool_message.content == "echo: hi"

    # The bridge prepared exactly one TOOL record (LLM lifecycle records also
    # land in the bridge — `on_chat_model_start` always calls `prepare_llm`
    # unconditionally — but those are Phase 5's concern; `tool_name` is only
    # ever set on a tool record) and the callback sent both lifecycle events
    # against it (sent-flags, not merely record existence) — captured via the
    # sweep spy since `_cleanup_turn` pops records at turn end.
    tool_records = [r for r in swept if r.tool_name is not None]
    assert len(tool_records) == 1
    record = tool_records[0]
    assert record.tool_started_sent is True
    assert record.tool_completed_sent is True
    assert record.tool_name == "recording_tool"

    # `run_id == prepared activity_id`: the callback's `on_tool_start` sees
    # `run_id` equal to the canonical id `tool_activity_binding` installed —
    # proven by the wire payload's `activity_id` matching the bridge record's.
    started = _lifecycle_payloads(fake_core, "ActivityStarted")
    completed = _lifecycle_payloads(fake_core, "ActivityCompleted")
    assert started and started[0]["activity_id"] == record.activity_id
    assert completed and completed[0]["activity_id"] == record.activity_id


async def test_activity_started_evaluated_before_tool_body_side_effect() -> None:
    """ActivityStarted must reach Core BEFORE the tool body's side effect —
    `run_inline=True` dispatches the callback inline, ahead of the queued
    stream event, and the callback's own gate call happens before it lets
    `execute()` run."""
    fake_core = FakeCore()
    store = ContextStore()
    adapter = LangGraphFrameworkAdapter(context_store=store)
    graph = _build_tool_call_graph()

    with installed_conformance_runtime(fake_core, adapter, store) as runtime:
        handler = _build_handler(graph, runtime)
        await handler.ainvoke(
            {"messages": [HumanMessage(content="please run the tool")]},
            config={"configurable": {"thread_id": "ordering-thread-2"}},
        )

    started = _lifecycle_payloads(fake_core, "ActivityStarted")
    assert started, "callback must send an ActivityStarted lifecycle payload"
    # The FIRST started-shaped payload precedes the tool body's own effect —
    # proven by the body effect being recorded (module-level list, appended
    # synchronously inside the tool function) only AFTER ainvoke() returns;
    # if the body ran before ActivityStarted, evaluate() would have nothing
    # to gate. This is the ordering the C1 sent-flag design depends on.
    assert _body_effects == ["hi"]


async def test_block_stops_body_and_failed_completion_closes_row() -> None:
    """BLOCK on ToolStarted: the tool body never runs, GovernanceBlockedError
    reaches the caller, and the orphan row is closed with a failed
    ActivityCompleted (C6) — never left open."""
    # ONE leading ALLOW is the agent LLM call's ActivityCompleted (Phase 5) —
    # its START is pre-screen-reused (no gate call — see
    # test_streaming_propagates_pending_approval_and_cleans_up's identical
    # note), but its COMPLETION makes a real `gate.aevaluate` call that DOES
    # consume from this FIFO queue before the tool node is ever reached.
    fake_core = FakeCore(
        {"verdict": "allow"},
        {"verdict": "block", "reason": "blocked by policy"},
    )
    store = ContextStore()
    adapter = LangGraphFrameworkAdapter(context_store=store)
    graph = _build_tool_call_graph()

    with installed_conformance_runtime(fake_core, adapter, store) as runtime:
        handler = _build_handler(graph, runtime)
        with pytest.raises(GovernanceBlockedError):
            await handler.ainvoke(
                {"messages": [HumanMessage(content="please run the tool")]},
                config={"configurable": {"thread_id": "ordering-thread-block"}},
            )

    assert _body_effects == [], "tool body must never run on a BLOCK verdict"
    completed = _lifecycle_payloads(fake_core, "ActivityCompleted")
    assert completed, "the orphan ActivityStarted row must be closed"
    closed = completed[0]
    assert closed.get("error") or closed.get("status") == "failed"


async def test_require_approval_polls_the_tools_real_activity_id() -> None:
    """C5 — the outer `ainvoke` REQUIRE_APPROVAL poll must target the TOOL's
    REAL activity_id (the canonical id bound at the ToolNode seam), never the
    generic `f"{run_id}-hook"` synthetic id: Core matches approvals on
    `(workflow_id, run_id, activity_id)` exactly, so polling the wrong key
    would hang `poll_until_decision` forever in production."""
    # ONE leading ALLOW is the agent LLM call's ActivityCompleted (Phase 5 —
    # see test_block_stops_body_and_failed_completion_closes_row's identical
    # note) — its START is pre-screen-reused, consuming nothing from this
    # queue.
    fake_core = FakeCore(
        {"verdict": "allow"},
        {"verdict": "require_approval", "approval_id": "app-1"},
    )
    store = ContextStore()
    adapter = LangGraphFrameworkAdapter(context_store=store)
    graph = _build_tool_call_graph()

    with installed_conformance_runtime(fake_core, adapter, store) as runtime:
        handler = _build_handler(graph, runtime)
        with patch(
            "openbox_langgraph.langgraph_handler.poll_until_decision",
            new=AsyncMock(return_value=None),
        ) as mock_poll:
            await handler.ainvoke(
                {"messages": [HumanMessage(content="please run the tool")]},
                config={"configurable": {"thread_id": "ordering-thread-approval"}},
            )

    mock_poll.assert_awaited_once()
    polled_params = mock_poll.await_args.args[1]
    started = _lifecycle_payloads(fake_core, "ActivityStarted")
    assert started
    assert polled_params.activity_id == started[0]["activity_id"]
    assert polled_params.activity_id != f"{polled_params.run_id}-hook"


async def test_tool_completed_block_is_enforced_from_stashed_verdict() -> None:
    """P1 — a callback-owned ToolCompleted BLOCK must be ENFORCED by the
    consumer (reading the stashed verdict), not silently dropped as
    telemetry-only. First queued verdict (start) is ALLOW, second
    (completion) is BLOCK."""
    # ONE leading ALLOW is the agent LLM call's ActivityCompleted (Phase 5 —
    # see test_block_stops_body_and_failed_completion_closes_row's identical
    # note), THEN the tool's ALLOW start / BLOCK completion this test targets.
    fake_core = FakeCore(
        {"verdict": "allow"},
        {"verdict": "allow"},
        {"verdict": "block", "reason": "completion blocked by policy"},
    )
    store = ContextStore()
    adapter = LangGraphFrameworkAdapter(context_store=store)
    graph = _build_tool_call_graph()

    with installed_conformance_runtime(fake_core, adapter, store) as runtime:
        handler = _build_handler(graph, runtime)
        with pytest.raises(GovernanceBlockedError):
            await handler.ainvoke(
                {"messages": [HumanMessage(content="please run the tool")]},
                config={"configurable": {"thread_id": "ordering-thread-complete-block"}},
            )

    assert _body_effects == ["hi"], "the tool body DID run (start was ALLOW)"


async def test_streaming_propagates_pending_approval_and_cleans_up() -> None:
    """Unlike `ainvoke`, `astream_events`/`astream_governed`/`astream` have no
    outer catch/poll/retry loop (pre-existing — only `ainvoke` drives HITL) —
    a callback REQUIRE_APPROVAL raise PROPAGATES straight to the caller. The
    `finally` still runs `_cleanup_turn`, so after the raise the bridge and
    the store's `_aborted_activities` are both empty — no leaked bridge
    record or abort mark (M19), even though the turn never resolved."""
    # ONE leading ALLOW is the agent LLM call's ActivityCompleted (Phase 5) —
    # its START is pre-screen-reused (`_pre_screen_input` uses the injected
    # `_AllowEverythingClient`, never the FakeCore-backed gate, so it never
    # touches this queue), but its COMPLETION makes a real `gate.aevaluate`
    # call that DOES consume from this FIFO queue before the tool node is
    # ever reached. REQUIRE_APPROVAL must land on the TOOL's START (which
    # ENFORCES/raises, per `enforce_tool_start_async`) — NOT its completion
    # (which only sends telemetry; enforcement of a completion verdict is the
    # CONSUMER's poll-and-continue job, a different code path that would
    # actually call `poll_until_decision` instead of propagating a raise).
    fake_core = FakeCore(
        {"verdict": "allow"},
        {"verdict": "require_approval", "approval_id": "app-2"},
    )
    store = ContextStore()
    adapter = LangGraphFrameworkAdapter(context_store=store)
    graph = _build_tool_call_graph()

    with installed_conformance_runtime(fake_core, adapter, store) as runtime:
        handler = _build_handler(graph, runtime)
        with pytest.raises(GovernanceBlockedError) as exc_info:
            async for _event in handler.astream_events(
                {"messages": [HumanMessage(content="please run the tool")]},
                config={"configurable": {"thread_id": "ordering-thread-stream"}},
            ):
                pass

        assert exc_info.value.verdict == "require_approval"
        bridge = handler._activity_bridge  # type: ignore[attr-defined]
        assert bridge._workflows == {}  # type: ignore[attr-defined]
        assert store._aborted_activities == set()  # type: ignore[attr-defined]
