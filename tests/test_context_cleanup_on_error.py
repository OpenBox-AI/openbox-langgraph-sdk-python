"""Turn-exit cleanup of the core-runtime dual-write bindings across all four
public entry points: `ainvoke`, `astream_governed`, `astream`, `astream_events`.

Each entry point wraps its stream loop in `try/finally` calling
`_cleanup_turn(workflow_id)`, which sweeps BOTH the trace registrations
`activity_context_binding.register_activity` wrote during the turn AND any
abort marks registered against that turn's `workflow_id` on the base
`ContextStore`. This must hold on: normal completion, a mid-stream exception,
an abandoned/early-closed generator, and (for `ainvoke` specifically) AFTER an
approved hook-approval retry completes — never before, since the retry keeps
using the pre-screen's identifiers and cleanup must not race it.
"""

from __future__ import annotations

from typing import Annotated, Any, TypedDict
from unittest.mock import AsyncMock, patch

import pytest
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

pytest.importorskip("openbox_core")

from openbox_langgraph.client import GovernanceClient
from openbox_langgraph.core_runtime import create_core_runtime, get_trace_registry
from openbox_langgraph.errors import GovernanceBlockedError
from openbox_langgraph.langgraph_handler import (
    OpenBoxLangGraphHandler,
    OpenBoxLangGraphHandlerOptions,
)
from openbox_langgraph.types import GovernanceVerdictResponse, Verdict
from tests.golden.fake_agent_graphs import build_tool_call_graph

# NOTE: the `_unconfigured_global_state` (autouse) and `_ensure_recording_tracer_provider`
# fixtures this module depends on live in `tests/conftest.py`. See that
# file's docstring for why both are required.

# Every core runtime built by the helper installs base instrumentation (the
# only hook runtime); close them all at teardown so no global hook state leaks
# into a later test in the same session.
_created_runtimes: list[Any] = []


@pytest.fixture(autouse=True)
def _close_created_runtimes() -> Any:
    yield
    while _created_runtimes:
        _created_runtimes.pop().close()


class _AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


class _AllowEverythingClient(GovernanceClient):
    """Always-ALLOW stub — isolates the test to cleanup timing, not verdicts."""

    def __init__(self) -> None:
        super().__init__(api_url="https://core.openbox.ai", api_key="obx_test_abc")

    async def evaluate_event(self, event: Any) -> GovernanceVerdictResponse | None:
        return GovernanceVerdictResponse(verdict=Verdict.ALLOW)


def _build_handler_with_core_runtime() -> OpenBoxLangGraphHandler:
    """Same technique as test_core_context_binding.py: injected client avoids
    building a core runtime in `__init__`; `_core_runtime` assigned directly
    afterward (registered for teardown close) so the registration + cleanup
    code paths run."""
    handler = OpenBoxLangGraphHandler(
        graph=None, options=OpenBoxLangGraphHandlerOptions(client=_AllowEverythingClient())
    )
    handler._core_runtime = create_core_runtime(  # type: ignore[attr-defined]
        handler._config,  # type: ignore[attr-defined]
        api_url="https://core.openbox.ai",
        api_key="obx_test_abc",
        governance_timeout=30.0,
    )
    _created_runtimes.append(handler._core_runtime)  # type: ignore[attr-defined]
    return handler


def _simple_graph() -> Any:
    """One plain node, no tool/LLM sub-events — enough to complete or raise a turn."""

    async def agent(state: _AgentState) -> dict[str, Any]:
        return {"messages": [AIMessage(content="done")]}

    graph = StateGraph(_AgentState)
    graph.add_node("agent", agent)
    graph.add_edge(START, "agent")
    graph.add_edge("agent", END)
    return graph.compile()


def _raising_node_graph(exc: Exception) -> Any:
    async def agent(state: _AgentState) -> dict[str, Any]:
        raise exc

    graph = StateGraph(_AgentState)
    graph.add_node("agent", agent)
    graph.add_edge(START, "agent")
    graph.add_edge("agent", END)
    return graph.compile()


def _registry_trace_count(handler: OpenBoxLangGraphHandler) -> int:
    """White-box read of the registry's internal trace map size — this
    project's ruff config does not enable SLF001, so no noqa is needed for
    the private-attribute access."""
    registry = get_trace_registry(handler._core_runtime)  # type: ignore[arg-type]
    return len(registry._by_trace)


@pytest.mark.asyncio
async def test_ainvoke_sweeps_trace_bindings_on_normal_completion() -> None:
    """After a successful ainvoke turn, no trace bindings from that turn
    remain registered — proves the outermost `finally` actually fires."""
    handler = _build_handler_with_core_runtime()
    handler._graph = _simple_graph()  # type: ignore[attr-defined]

    await handler.ainvoke(
        {"messages": [HumanMessage(content="hi")]},
        config={"configurable": {"thread_id": "cleanup-thread-1"}},
    )

    assert _registry_trace_count(handler) == 0


@pytest.mark.asyncio
async def test_ainvoke_sweeps_on_mid_stream_non_approval_exception() -> None:
    """A node raising a plain (non-GovernanceBlockedError) exception still
    triggers `_cleanup_turn` via the `finally` — cleanup is not conditioned on
    a specific exception type, unlike the approval-retry `except` branches."""
    handler = _build_handler_with_core_runtime()
    handler._graph = _raising_node_graph(RuntimeError("boom"))  # type: ignore[attr-defined]

    with pytest.raises(RuntimeError, match="boom"):
        await handler.ainvoke(
            {"messages": [HumanMessage(content="hi")]},
            config={"configurable": {"thread_id": "cleanup-thread-2"}},
        )

    assert _registry_trace_count(handler) == 0


@pytest.mark.asyncio
async def test_ainvoke_cleanup_runs_once_after_approved_retry_not_before() -> None:
    """Mirrors test_hook_approval_retry_baseline.py's flaky-node graph, but
    with a real core runtime wired: `_cleanup_turn` must be called EXACTLY
    ONCE, and only after the approved retry's `self._graph.ainvoke(...)` call
    has already returned — proving the `finally` sits at the outermost level
    of the `try/except/except/finally`, not inside either `except` block."""
    handler = _build_handler_with_core_runtime()
    call_count = {"n": 0}
    cleanup_calls: list[int] = []  # snapshot of call_count["n"] at each cleanup call

    async def flaky_node(state: _AgentState) -> dict[str, Any]:
        call_count["n"] += 1
        if call_count["n"] == 1:
            raise GovernanceBlockedError("require_approval", "needs approval", "tool")
        return {"messages": [AIMessage(content="succeeded on retry")]}

    graph = StateGraph(_AgentState)
    graph.add_node("agent", flaky_node)
    graph.add_edge(START, "agent")
    graph.add_edge("agent", END)
    handler._graph = graph.compile()  # type: ignore[attr-defined]

    original_cleanup = handler._cleanup_turn  # type: ignore[attr-defined]

    def _spy_cleanup(workflow_id: str) -> None:
        cleanup_calls.append(call_count["n"])
        original_cleanup(workflow_id)

    with (
        patch(
            "openbox_langgraph.langgraph_handler.poll_until_decision",
            new=AsyncMock(return_value=None),
        ) as mock_poll,
        patch.object(handler, "_cleanup_turn", side_effect=_spy_cleanup),
    ):
        result = await handler.ainvoke(
            {"messages": [HumanMessage(content="hi")]},
            config={"configurable": {"thread_id": "cleanup-thread-3"}},
        )

    mock_poll.assert_awaited_once()
    assert call_count["n"] == 2, "expected one blocked pass + one retry"
    assert result["messages"][-1].content == "succeeded on retry"
    # Exactly one cleanup call, and it happened AFTER the retry's node ran
    # (call_count was already 2 by the time cleanup fired).
    assert cleanup_calls == [2]
    assert _registry_trace_count(handler) == 0


@pytest.mark.asyncio
async def test_astream_governed_sweeps_on_normal_completion() -> None:
    handler = _build_handler_with_core_runtime()
    handler._graph = _simple_graph()  # type: ignore[attr-defined]

    async for _ in handler.astream_governed(
        {"messages": [HumanMessage(content="hi")]},
        config={"configurable": {"thread_id": "cleanup-thread-4"}},
    ):
        pass

    assert _registry_trace_count(handler) == 0


@pytest.mark.asyncio
async def test_astream_governed_sweeps_on_mid_stream_exception() -> None:
    handler = _build_handler_with_core_runtime()
    handler._graph = _raising_node_graph(RuntimeError("stream boom"))  # type: ignore[attr-defined]

    with pytest.raises(RuntimeError, match="stream boom"):
        async for _ in handler.astream_governed(
            {"messages": [HumanMessage(content="hi")]},
            config={"configurable": {"thread_id": "cleanup-thread-5"}},
        ):
            pass

    assert _registry_trace_count(handler) == 0


@pytest.mark.asyncio
async def test_astream_sweeps_via_delegation_to_astream_governed() -> None:
    """`astream` mints no turn of its own — this proves its `aclosing`-wrapped
    delegation still triggers `astream_governed`'s cleanup on normal exhaustion."""
    handler = _build_handler_with_core_runtime()
    handler._graph = _simple_graph()  # type: ignore[attr-defined]

    async for _ in handler.astream(
        {"messages": [HumanMessage(content="hi")]},
        config={"configurable": {"thread_id": "cleanup-thread-6"}},
    ):
        pass

    assert _registry_trace_count(handler) == 0


@pytest.mark.asyncio
async def test_astream_sweeps_on_abandoned_generator_early_break() -> None:
    """The critical case `astream`'s own `finally` exists for: the CALLER
    breaks out of the `async for` early, abandoning the generator. Without
    `contextlib.aclosing` wrapping the inner `astream_governed` generator,
    `GeneratorExit` on this outer generator's `aclose()` would NEVER reach the
    inner generator's `finally` (verified empirically — see astream's own
    code comment) and the turn's trace bindings would leak forever."""
    handler = _build_handler_with_core_runtime()
    handler._graph = _simple_graph()  # type: ignore[attr-defined]

    gen = handler.astream(
        {"messages": [HumanMessage(content="hi")]},
        config={"configurable": {"thread_id": "cleanup-thread-7"}},
    )
    async for _ in gen:
        break  # abandon after the first chunk
    await gen.aclose()

    assert _registry_trace_count(handler) == 0


@pytest.mark.asyncio
async def test_astream_events_sweeps_on_normal_completion() -> None:
    handler = _build_handler_with_core_runtime()
    handler._graph = _simple_graph()  # type: ignore[attr-defined]

    async for _ in handler.astream_events(
        {"messages": [HumanMessage(content="hi")]},
        config={"configurable": {"thread_id": "cleanup-thread-8"}},
    ):
        pass

    assert _registry_trace_count(handler) == 0


@pytest.mark.asyncio
async def test_astream_events_sweeps_on_mid_stream_exception() -> None:
    handler = _build_handler_with_core_runtime()
    handler._graph = _raising_node_graph(ValueError("events boom"))  # type: ignore[attr-defined]

    with pytest.raises(ValueError, match="events boom"):
        async for _ in handler.astream_events(
            {"messages": [HumanMessage(content="hi")]},
            config={"configurable": {"thread_id": "cleanup-thread-9"}},
        ):
            pass

    assert _registry_trace_count(handler) == 0


@pytest.mark.asyncio
async def test_abort_marks_set_during_a_turn_do_not_survive_ainvoke() -> None:
    """`mark_activity_aborted` calls made during a turn (driven by whichever
    hook runtime resolves context via this registry, not by this test) must
    not survive past `ainvoke` returning. The base `ContextStore` has no
    prefix-sweep of its own, so `_cleanup_turn` supplies it via
    `TraceContextRegistry`'s own tracked activity keys.

    Uses `build_tool_call_graph()` (shared golden-test fixture: a real
    `echo_tool` round-trip via `ToolNode`) so `on_tool_start` genuinely
    dual-writes an activity this test can then mark aborted mid-turn — a
    plain node function never fires `on_tool_start`, so there would be
    nothing registered to abort against.
    """
    handler = _build_handler_with_core_runtime()
    handler._graph = build_tool_call_graph()  # type: ignore[attr-defined]
    marked: dict[str, tuple[str, str]] = {}

    original_register = get_trace_registry(handler._core_runtime).register  # type: ignore[arg-type]

    def _mark_on_first_register(trace_id: Any, ctx: Any) -> None:
        original_register(trace_id, ctx)
        # Mark THIS activity's own registered (workflow_id, activity_id) —
        # matching how a hook runtime resolving context via `register()`
        # would mark aborted using that SAME activity_id, which is exactly
        # what `TraceContextRegistry._activity_keys` tracks for the per-turn
        # sweep to find.
        if "key" not in marked and ctx.activity_type == "echo_tool":
            marked["key"] = (ctx.workflow_id, ctx.activity_id)
            handler._core_runtime.context_store.mark_activity_aborted(  # type: ignore[attr-defined]
                ctx.workflow_id, ctx.activity_id
            )

    with patch.object(
        get_trace_registry(handler._core_runtime),  # type: ignore[arg-type]
        "register",
        side_effect=_mark_on_first_register,
    ):
        await handler.ainvoke(
            {"messages": [HumanMessage(content="hi")]},
            config={"configurable": {"thread_id": "cleanup-thread-10"}},
        )

    assert marked, "echo_tool's registration was never observed to abort against"
    workflow_id, activity_id = marked["key"]
    store = handler._core_runtime.context_store  # type: ignore[attr-defined]
    assert not store.is_activity_aborted(workflow_id, activity_id)
