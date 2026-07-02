"""Cross-turn isolation of the core-runtime dual-write bindings on a SINGLE
long-lived handler instance: two sequential turns must never see each
other's registered trace context or abort marks, and an abandoned generator
from an earlier turn must not leak state into a later one.

A production handler is constructed once and reused across many user turns
(one `ainvoke`/`astream*` call per turn on the SAME `OpenBoxLangGraphHandler`,
hence the SAME private `ContextStore`/`TraceContextRegistry`) — so isolation
has to hold ACROSS calls on one instance, not just within a single call.
"""

from __future__ import annotations

from typing import Annotated, Any, TypedDict

import pytest
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

pytest.importorskip("openbox_core")

from openbox_langgraph.client import GovernanceClient
from openbox_langgraph.core_runtime import create_core_runtime, get_trace_registry
from openbox_langgraph.langgraph_handler import (
    OpenBoxLangGraphHandler,
    OpenBoxLangGraphHandlerOptions,
)
from openbox_langgraph.span_processor import WorkflowSpanProcessor
from openbox_langgraph.types import GovernanceVerdictResponse, Verdict
from tests.golden.fake_agent_graphs import build_tool_call_graph

# NOTE: the `_unconfigured_global_state` (autouse) and `_ensure_recording_tracer_provider`
# fixtures this module depends on live in `tests/conftest.py`.


class _AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


class _AllowEverythingClient(GovernanceClient):
    """Always-ALLOW stub — isolates the test to turn isolation, not verdicts."""

    def __init__(self) -> None:
        super().__init__(api_url="https://core.openbox.ai", api_key="obx_test_abc")

    async def evaluate_event(self, event: Any) -> GovernanceVerdictResponse | None:
        return GovernanceVerdictResponse(verdict=Verdict.ALLOW)


def _build_handler_with_core_runtime() -> OpenBoxLangGraphHandler:
    """Same technique as the other context-binding test modules: injected
    client avoids `initialize()`'s global OTel side effects; `_core_runtime`/
    `_span_processor` assigned directly afterward."""
    handler = OpenBoxLangGraphHandler(
        graph=None, options=OpenBoxLangGraphHandlerOptions(client=_AllowEverythingClient())
    )
    handler._core_runtime = create_core_runtime(  # type: ignore[attr-defined]
        handler._config,  # type: ignore[attr-defined]
        api_url="https://core.openbox.ai",
        api_key="obx_test_abc",
        governance_timeout=30.0,
    )
    handler._span_processor = WorkflowSpanProcessor()  # type: ignore[attr-defined]
    return handler


def _simple_graph() -> Any:
    async def agent(state: _AgentState) -> dict[str, Any]:
        return {"messages": [AIMessage(content="done")]}

    graph = StateGraph(_AgentState)
    graph.add_node("agent", agent)
    graph.add_edge(START, "agent")
    graph.add_edge("agent", END)
    return graph.compile()


@pytest.mark.asyncio
async def test_two_sequential_ainvoke_turns_use_distinct_workflow_ids() -> None:
    """Each `ainvoke` call mints a fresh, distinct `workflow_id` — the
    precondition every other isolation guarantee in this file depends on."""
    handler = _build_handler_with_core_runtime()
    handler._graph = build_tool_call_graph()  # type: ignore[attr-defined]

    seen_workflow_ids: list[str] = []
    original_cleanup = handler._cleanup_turn  # type: ignore[attr-defined]

    def _spy(workflow_id: str) -> None:
        seen_workflow_ids.append(workflow_id)
        original_cleanup(workflow_id)

    handler._cleanup_turn = _spy  # type: ignore[method-assign]

    await handler.ainvoke(
        {"messages": [HumanMessage(content="turn one")]},
        config={"configurable": {"thread_id": "isolation-thread"}},
    )
    handler._graph = build_tool_call_graph()  # type: ignore[attr-defined]
    await handler.ainvoke(
        {"messages": [HumanMessage(content="turn two")]},
        config={"configurable": {"thread_id": "isolation-thread"}},
    )

    assert len(seen_workflow_ids) == 2
    assert seen_workflow_ids[0] != seen_workflow_ids[1]


@pytest.mark.asyncio
async def test_second_turn_never_resolves_first_turns_registered_context() -> None:
    """After turn 1 completes (and its cleanup sweeps), the registry has zero
    entries; while turn 2 is IN FLIGHT, resolving any trace id the registry
    currently knows about must return turn 2's context, never turn 1's — the
    single-active fallback tier in particular would silently leak turn 1's
    identity into turn 2 if cleanup didn't actually run between them."""
    handler = _build_handler_with_core_runtime()
    registry = get_trace_registry(handler._core_runtime)  # type: ignore[arg-type]

    turn_one_workflow_ids: set[str] = set()
    handler._graph = build_tool_call_graph()  # type: ignore[attr-defined]
    original_register = registry.register

    def _capture_turn_one(trace_id: Any, ctx: Any) -> None:
        turn_one_workflow_ids.add(ctx.workflow_id)
        original_register(trace_id, ctx)

    registry.register = _capture_turn_one  # type: ignore[method-assign]
    await handler.ainvoke(
        {"messages": [HumanMessage(content="turn one")]},
        config={"configurable": {"thread_id": "isolation-thread-2"}},
    )
    registry.register = original_register  # type: ignore[method-assign]

    assert turn_one_workflow_ids, "turn one should have registered at least one activity"
    assert len(registry._by_trace) == 0, "turn one's cleanup must have swept everything"

    # Turn 2: capture whatever context IS resolvable mid-turn and assert it
    # never matches turn 1's workflow_id.
    resolved_mid_turn_two: list[Any] = []
    handler._graph = build_tool_call_graph()  # type: ignore[attr-defined]

    def _spy_register(trace_id: Any, ctx: Any) -> None:
        original_register(trace_id, ctx)
        resolved_mid_turn_two.append(registry.resolve(trace_id))

    registry.register = _spy_register  # type: ignore[method-assign]
    try:
        await handler.ainvoke(
            {"messages": [HumanMessage(content="turn two")]},
            config={"configurable": {"thread_id": "isolation-thread-2"}},
        )
    finally:
        registry.register = original_register  # type: ignore[method-assign]

    assert resolved_mid_turn_two, "turn two should have registered at least one activity"
    for ctx in resolved_mid_turn_two:
        assert ctx is not None
        assert ctx.workflow_id not in turn_one_workflow_ids


@pytest.mark.asyncio
async def test_abort_mark_from_first_turn_does_not_survive_into_second_turn() -> None:
    """A hook-style abort mark set during turn 1 (same technique as
    test_context_cleanup_on_error.py's abort-sweep test) must be gone by the
    time turn 2 runs — even when turn 2 reuses the EXACT SAME activity_type
    (`echo_tool`) turn 1 used, proving isolation is keyed correctly by
    workflow_id and not accidentally shared by activity_type/name alone."""
    handler = _build_handler_with_core_runtime()
    store = handler._core_runtime.context_store  # type: ignore[attr-defined]
    registry = get_trace_registry(handler._core_runtime)  # type: ignore[arg-type]
    original_register = registry.register

    turn_one_key: dict[str, tuple[str, str]] = {}

    def _mark_turn_one(trace_id: Any, ctx: Any) -> None:
        original_register(trace_id, ctx)
        if "key" not in turn_one_key and ctx.activity_type == "echo_tool":
            turn_one_key["key"] = (ctx.workflow_id, ctx.activity_id)
            store.mark_activity_aborted(ctx.workflow_id, ctx.activity_id)

    handler._graph = build_tool_call_graph()  # type: ignore[attr-defined]
    registry.register = _mark_turn_one  # type: ignore[method-assign]
    try:
        await handler.ainvoke(
            {"messages": [HumanMessage(content="turn one")]},
            config={"configurable": {"thread_id": "isolation-thread-3"}},
        )
    finally:
        registry.register = original_register  # type: ignore[method-assign]

    assert turn_one_key, "turn one never registered an echo_tool activity to abort"
    wf1, act1 = turn_one_key["key"]
    assert not store.is_activity_aborted(wf1, act1), "turn one's own cleanup should have cleared it"

    # Turn 2 reuses the same activity_type — a fresh workflow_id/activity_id
    # pair every time (event_run_id is LangGraph's own per-node UUID), but
    # even if it collided, is_activity_aborted is checked by the SAME key.
    seen_aborted_during_turn_two: list[bool] = []
    handler._graph = build_tool_call_graph()  # type: ignore[attr-defined]

    def _check_turn_two(trace_id: Any, ctx: Any) -> None:
        original_register(trace_id, ctx)
        if ctx.activity_type == "echo_tool":
            seen_aborted_during_turn_two.append(
                store.is_activity_aborted(ctx.workflow_id, ctx.activity_id)
            )

    registry.register = _check_turn_two  # type: ignore[method-assign]
    try:
        await handler.ainvoke(
            {"messages": [HumanMessage(content="turn two")]},
            config={"configurable": {"thread_id": "isolation-thread-3"}},
        )
    finally:
        registry.register = original_register  # type: ignore[method-assign]

    assert seen_aborted_during_turn_two, "turn two never registered an echo_tool activity"
    assert not any(seen_aborted_during_turn_two), (
        "turn two's echo_tool activity must never read as aborted from turn one's mark"
    )
    # And turn one's original key specifically stays clear.
    assert not store.is_activity_aborted(wf1, act1)


@pytest.mark.asyncio
async def test_abandoned_generator_from_first_turn_does_not_leak_into_second() -> None:
    """An `astream` caller that abandons turn 1 mid-stream (break + no
    explicit `aclose()` — relying on garbage collection, the way a caller
    who just stops iterating without a try/finally would) must not leave
    turn 1's bindings around to pollute turn 2's isolation. Exercises the
    SAME `aclosing`-wrapped delegation `test_context_cleanup_on_error.py`
    covers, but end to end across TWO turns on the SAME handler."""
    import gc

    handler = _build_handler_with_core_runtime()
    registry = get_trace_registry(handler._core_runtime)  # type: ignore[arg-type]

    handler._graph = build_tool_call_graph()  # type: ignore[attr-defined]
    gen = handler.astream(
        {"messages": [HumanMessage(content="turn one")]},
        config={"configurable": {"thread_id": "isolation-thread-4"}},
    )
    async for _ in gen:
        break  # abandon after the first chunk, no explicit aclose()
    del gen
    gc.collect()  # encourage prompt finalization in this test's own event loop

    handler._graph = build_tool_call_graph()  # type: ignore[attr-defined]
    resolved_mid_turn_two: list[Any] = []
    original_register = registry.register

    def _spy_register(trace_id: Any, ctx: Any) -> None:
        original_register(trace_id, ctx)
        resolved_mid_turn_two.append(registry.resolve(trace_id))

    registry.register = _spy_register  # type: ignore[method-assign]
    try:
        await handler.ainvoke(
            {"messages": [HumanMessage(content="turn two")]},
            config={"configurable": {"thread_id": "isolation-thread-4"}},
        )
    finally:
        registry.register = original_register  # type: ignore[method-assign]

    assert resolved_mid_turn_two, "turn two should have registered at least one activity"
    # No leftover entries from the abandoned turn one by the time turn two runs.
    assert len(registry._by_trace) == 0, "turn two's own cleanup should leave zero residue"
