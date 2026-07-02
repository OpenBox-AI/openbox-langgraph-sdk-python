"""Pre-migration baseline: the legacy hook-approval retry path is broken today.

Sequence (see openbox_langgraph/langgraph_handler.py OpenBoxLangGraphHandler.ainvoke,
openbox_langgraph/hook_governance.py, and openbox_langgraph/span_processor.py):

1. A hook (HTTP/DB/file/traced-function governance check) returns REQUIRE_APPROVAL.
   hook_governance sets an abort flag via WorkflowSpanProcessor.set_activity_abort
   and raises GovernanceBlockedError(verdict="require_approval").
2. ainvoke's except block catches it, polls until a decision via poll_until_decision,
   then re-runs the underlying graph directly with `self._graph.ainvoke(...)` —
   WITHOUT ever calling WorkflowSpanProcessor.clear_activity_abort.
3. WorkflowSpanProcessor.clear_activity_abort exists but has zero callers in the
   package — so the abort flag set in step 1 survives an approved retry.

WHAT THIS TEST ACTUALLY EXERCISES (read carefully before trusting it as an
end-to-end reproduction): it drives the REAL `OpenBoxLangGraphHandler.ainvoke`
against a real compiled LangGraph graph, but the `WorkflowSpanProcessor` here
is a STANDALONE instance constructed directly by the test, NOT the handler's
own `self._span_processor`. The handler only builds its own span processor
when the module-level global config has both `api_url` and `api_key` set
(langgraph_handler.py `__init__`, ~line 421-439: `if gc and gc.api_url and
gc.api_key: ... self._span_processor = WorkflowSpanProcessor() ...`) — this
test never calls `initialize()`, so `handler._span_processor` is actually
`None` throughout. Likewise, the abort flag here is set directly by the test's
flaky node (simulating what a hook would do), not by a real
`hook_governance.evaluate_sync`/`evaluate_async` call.

This was a deliberate choice, not an oversight: wiring through `initialize()`
so the handler builds a real `_span_processor` also runs
`setup_opentelemetry_for_governance`, which globally reconfigures the
`hook_governance` module singleton and registers a span processor with the
process-wide OTel `TracerProvider` — verified empirically that a SECOND
`trace.set_tracer_provider()` call in the same process is a silent no-op, so
every subsequent test that goes through that path in the same pytest session
would accumulate span processors on whichever provider was created first,
with no teardown. That cross-test pollution risk outweighs the benefit here.

What this test DOES prove, and is fully faithful to: the exact CONTROL FLOW
in `ainvoke`'s except block — catch GovernanceBlockedError(require_approval),
poll, retry via `self._graph.ainvoke(...)` — genuinely never touches
`clear_activity_abort` on ANY `WorkflowSpanProcessor` instance, standalone or
handler-owned, because that call simply does not exist anywhere on that code
path. The bug is in `ainvoke` itself, not in which span processor instance is
used, so this test's abort-flag observation is a valid stand-in for the real
`handler._span_processor` case without paying the OTel global-state cost.

PINS the current (broken) outcome: after an approved retry, the abort flag
set on the first pass is still present, and clear_activity_abort was never
invoked. A later fix that clears the flag on approval should update this test
alongside the fix — it exists to make that change visible and deliberate,
not to prevent it.
"""

from __future__ import annotations

from typing import Annotated, Any, TypedDict
from unittest.mock import AsyncMock, patch

import pytest
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

from openbox_langgraph.client import GovernanceClient
from openbox_langgraph.errors import GovernanceBlockedError
from openbox_langgraph.langgraph_handler import (
    OpenBoxLangGraphHandler,
    OpenBoxLangGraphHandlerOptions,
)
from openbox_langgraph.span_processor import WorkflowSpanProcessor
from openbox_langgraph.types import GovernanceVerdictResponse, Verdict

_WORKFLOW_KEY = "retry-baseline-workflow"
_ACTIVITY_KEY = "retry-baseline-activity"
_ABORT_REASON = "needs human approval before continuing"


class _AgentState(TypedDict):
    """Minimal LangGraph state for the retry-baseline graph."""

    messages: Annotated[list[BaseMessage], add_messages]


class _AllowEverythingClient(GovernanceClient):
    """GovernanceClient stub that always ALLOWs — isolates the test to the
    ainvoke retry path itself, not the ordinary lifecycle-event governance calls."""

    def __init__(self) -> None:
        super().__init__(api_url="https://core.openbox.ai", api_key="obx_test_abc")

    async def evaluate_event(self, event: Any) -> GovernanceVerdictResponse | None:
        return GovernanceVerdictResponse(verdict=Verdict.ALLOW)


def _build_flaky_node_graph(
    span_processor: WorkflowSpanProcessor, call_count: dict[str, int]
) -> Any:
    """One node that raises REQUIRE_APPROVAL on its first call, succeeds on retry.

    Mirrors what a hook does today: set the abort flag, then raise
    GovernanceBlockedError("require_approval", ...). Uses a closure counter
    (not graph state) to distinguish first-pass from retry, because the
    handler's retry re-invokes the graph with the ORIGINAL input — LangGraph
    state does not carry over between the failed pass and the direct
    `self._graph.ainvoke(...)` retry.
    """

    async def flaky_node(state: _AgentState) -> dict[str, Any]:
        call_count["n"] += 1
        if call_count["n"] == 1:
            span_processor.set_activity_abort(_WORKFLOW_KEY, _ACTIVITY_KEY, _ABORT_REASON)
            raise GovernanceBlockedError("require_approval", _ABORT_REASON, "tool-under-test")
        return {"messages": [AIMessage(content="succeeded on approved retry")]}

    graph = StateGraph(_AgentState)
    graph.add_node("agent", flaky_node)
    graph.add_edge(START, "agent")
    graph.add_edge("agent", END)
    return graph.compile()


@pytest.mark.asyncio
async def test_abort_flag_survives_an_approved_retry() -> None:
    """PINS the bug: the abort flag set before REQUIRE_APPROVAL is never cleared,
    even after poll_until_decision resolves the approval and the retry succeeds.

    Uses a standalone WorkflowSpanProcessor (not handler._span_processor, which
    stays None here — see module docstring for why) to observe the same
    ainvoke control-flow bug without triggering global OTel setup."""
    span_processor = WorkflowSpanProcessor()
    call_count = {"n": 0}
    graph = _build_flaky_node_graph(span_processor, call_count)
    handler = OpenBoxLangGraphHandler(
        graph, OpenBoxLangGraphHandlerOptions(client=_AllowEverythingClient())
    )

    with patch(
        "openbox_langgraph.langgraph_handler.poll_until_decision",
        new=AsyncMock(return_value=None),
    ) as mock_poll:
        result = await handler.ainvoke(
            {"messages": [HumanMessage(content="please run the tool")]},
            config={"configurable": {"thread_id": "retry-baseline-thread"}},
        )

    # Sanity: the retry path actually ran (first pass blocked, poll happened,
    # second pass via self._graph.ainvoke succeeded).
    assert call_count["n"] == 2, "expected exactly one blocked pass + one retry"
    mock_poll.assert_awaited_once()
    assert result["messages"][-1].content == "succeeded on approved retry"

    # The bug: nothing on the approved-retry path clears the abort flag.
    abort_after_retry = span_processor.get_activity_abort(_WORKFLOW_KEY, _ACTIVITY_KEY)
    assert abort_after_retry == _ABORT_REASON, (
        "baseline expects the abort flag to survive an approved retry — "
        "if this now fails, clear_activity_abort has gained a caller and "
        "this baseline test (and the bug it pins) should be revisited"
    )


@pytest.mark.asyncio
async def test_clear_activity_abort_is_never_called_on_the_retry_path() -> None:
    """PINS the bug from the other direction: clear_activity_abort has zero callers
    on the approval-retry path — spy on the method and assert it is never invoked.

    Same standalone WorkflowSpanProcessor caveat as test_abort_flag_survives_an_approved_retry —
    see module docstring."""
    span_processor = WorkflowSpanProcessor()
    call_count = {"n": 0}
    graph = _build_flaky_node_graph(span_processor, call_count)
    handler = OpenBoxLangGraphHandler(
        graph, OpenBoxLangGraphHandlerOptions(client=_AllowEverythingClient())
    )

    with (
        patch(
            "openbox_langgraph.langgraph_handler.poll_until_decision",
            new=AsyncMock(return_value=None),
        ),
        patch.object(
            span_processor, "clear_activity_abort", wraps=span_processor.clear_activity_abort
        ) as spy_clear,
    ):
        await handler.ainvoke(
            {"messages": [HumanMessage(content="please run the tool")]},
            config={"configurable": {"thread_id": "retry-baseline-thread-2"}},
        )

    spy_clear.assert_not_called()
